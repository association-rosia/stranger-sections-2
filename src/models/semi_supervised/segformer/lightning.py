import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as tm
import wandb
import math
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SamModel, SamImageProcessor

import src.data.semi_supervised.dataset as ssp_dataset
from src.data.processor import SS2ImageProcessor
from src.data.tiling import Tiler
from src.utils import func
from src.utils.cls import Config

torch.set_float32_matmul_precision('medium')


class SegFormerLightning(pl.LightningModule):
    def __init__(self, config: Config):
        super(SegFormerLightning, self).__init__()

        self.config = config
        self.class_labels = self.config.data.class_labels.to_dict()

        tiler = Tiler(config)
        self.labeled_tiles = tiler.build(labeled=True)
        self.unlabeled_tiles = tiler.build(labeled=False)
        self.processor = SS2ImageProcessor.get_huggingface_processor(config)

        self.student = load_student_model(self.config)
        self.teacher = load_teacher_model(self.config)
        self.sam, self.sam_processor = load_sam(self.config)

        self.delta_c, self.delta_s = None, None
        self.update_loss_weights()

        self.segmentation_loss_fct = self.configure_criterion()
        self.consistency_loss_fct = nn.CrossEntropyLoss()
        self.sam_loss_fct = nn.CrossEntropyLoss()
        self.metrics = self.configure_metrics()

        self.current_step = None
        self.current_batch_idx = None

    def forward(self, batch):
        segmentation_input, segmentation_image, consistency_inputs, consistency_image = batch
        segmentation_loss = self.segmentation_forward(segmentation_input)
        consistency_loss, consistency_logits_1 = self.consistency_forward(consistency_inputs)
        sam_loss = self.sam_forward(consistency_inputs, consistency_logits_1)
        loss = segmentation_loss + self.delta_c * consistency_loss + self.delta_s * sam_loss

        return loss

    def training_step(self, batch):
        self.update_loss_weights()
        self.current_step = 'training'
        loss = self.forward(batch)
        self.log('train/loss', loss, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self.current_step = 'validation'
        self.current_batch_idx = batch_idx
        loss = self.forward(batch)
        self.log('val/loss', loss, on_epoch=True, sync_dist=True)

        return loss

    def on_validation_epoch_end(self):
        metrics = self.metrics.compute()
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        self.metrics.reset()

    def configure_optimizers(self):
        optimizer = AdamW(params=self.student.parameters(), lr=self.config.lr)

        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.config.reduce_lr_on_plateau_factor,
                patience=self.config.reduce_lr_on_plateau_patience,
                verbose=True
            ),
            'monitor': 'val/loss',
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def configure_criterion(self):
        class_labels = self.config.data.class_labels.__dict__
        class_ordered = sorted([int(k) for k in class_labels.keys()])
        label_weights = self.config.data.label_weights.__dict__
        weight = torch.Tensor([label_weights[class_labels[str(i)]] for i in class_ordered])

        return nn.CrossEntropyLoss(weight=weight)

    def configure_metrics(self):
        num_labels = self.config.num_labels

        metrics = tm.MetricCollection({
            'val/dice-macro': tm.Dice(num_classes=num_labels, average='macro'),
            'val/dice-micro': tm.Dice(num_classes=num_labels, average='micro'),
            'val/iou-macro': tm.JaccardIndex(task='multiclass', num_classes=num_labels, average='macro'),
            'val/iou-micro': tm.JaccardIndex(task='multiclass', num_classes=num_labels, average='micro'),
        })

        return metrics

    def segmentation_forward(self, inputs):
        labels = self.reshape_labels(inputs)
        outputs = self.student(**inputs)
        logits = self.reshape_outputs(inputs, outputs)

        if self.current_step == 'validation':
            masks = self.reshape_outputs(inputs, outputs, return_mask=True)
            self.metrics.update(masks, labels)

            if self.current_batch_idx == 0:
                self.log_segmentation_images(inputs, labels, masks)

        loss = self.segmentation_loss_fct(logits, labels)
        self.log('val/segmentation_loss', loss, on_epoch=True, sync_dist=True)

        return loss

    def consistency_forward(self, inputs):
        inputs_1, inputs_2 = inputs

        outputs_1 = self.student(**inputs_1)
        logits_1 = self.reshape_outputs(inputs_1, outputs_1)

        outputs_2 = self.teacher(**inputs_2)
        logits_2 = self.reshape_outputs(inputs_2, outputs_2)
        mask_2 = self.logits_to_masks(logits_2)

        if self.current_step == 'validation' and self.current_batch_idx == 0:
            mask_1 = self.logits_to_masks(logits_1)
            self.log_consistency_images(inputs_1, mask_1, '1')
            self.log_consistency_images(inputs_2, mask_2, '2')

        loss = self.consistency_loss_fct(logits_1, mask_2)
        self.log('val/consistency_loss', loss, on_epoch=True, sync_dist=True)

        return loss, logits_1

    @torch.no_grad()
    def sam_forward(self, inputs, consistency_logits):
        inputs, _ = inputs
        consistency_masks = self.logits_to_masks(consistency_logits)

        flatten_inputs, values, idcs = self.create_flatten_inputs(consistency_masks, inputs)
        flatten_outputs = self.sam(**flatten_inputs)
        sam_masks = self.post_process_flatten_outputs(flatten_inputs, flatten_outputs, values, idcs)

        if self.current_step == 'validation' and self.current_batch_idx == 0:
            self.log_sam_images(inputs, consistency_masks, sam_masks)

        loss = self.sam_loss_fct(consistency_logits, sam_masks.long())
        self.log('val/sam_loss', loss, on_epoch=True, sync_dist=True)

        return loss

    def post_process_flatten_outputs(self, flatten_inputs, flatten_outputs, values, idcs):
        sam_masks = []
        unique_idcs = list(set(idcs))

        masks = self.sam_processor.post_process_masks(
            masks=flatten_outputs.pred_masks,
            original_sizes=flatten_inputs['original_sizes'],
            reshaped_input_sizes=flatten_inputs['reshaped_input_sizes'],
            binarize=False
        )

        masks = torch.cat(masks)
        masks = masks.squeeze(dim=1)

        for idx in unique_idcs:
            sam_mask = []
            post_processed_masks_idx = masks[torch.Tensor(idcs) == idx]
            mask_values = [values[i] for i in idcs if i == idx]

            replaced_class = 0
            for i in range(4):
                if i in mask_values:
                    sam_mask.append(post_processed_masks_idx[replaced_class])
                    replaced_class += 1
                elif i == 0:
                    sam_mask.append(1e-8 * torch.ones(masks.shape[-2:], device=masks.device, dtype=torch.float16))
                else:
                    sam_mask.append(torch.zeros(masks.shape[-2:], device=masks.device, dtype=torch.float16))

            sam_mask = torch.stack(sam_mask)
            sam_mask = sam_mask.argmax(dim=0)
            sam_masks.append(sam_mask)

        sam_masks = torch.stack(sam_masks)
        sam_masks = self.reshape_tensor(sam_masks, size=(self.config.tile_size, self.config.tile_size), is_3d=True)

        return sam_masks

    def create_flatten_inputs(self, consistency_masks, inputs):
        input_masks = []
        pixel_values = []
        values = []
        idcs = []
        device = consistency_masks.device

        for i in range(self.config.batch_size):
            # create prompt mask
            consistency_mask = consistency_masks[i]
            input_masks_i, values_i = self.get_sam_input_masks(consistency_mask)
            input_masks.append(input_masks_i)
            values += values_i

            # create input image
            num_masks = len(input_masks_i) if input_masks_i is not None else 1
            pixel_values_i = torch.stack([inputs['pixel_values'][i] for _ in range(num_masks)])
            pixel_values_i = self.reshape_tensor(pixel_values_i)
            pixel_values.append(pixel_values_i)
            idcs += [i for _ in range(num_masks)]

        input_masks = torch.cat(input_masks).unsqueeze(dim=1)
        pixel_values = torch.cat(pixel_values)
        flatten_inputs = self.sam_processor(images=pixel_values, return_tensors='pt')
        flatten_inputs['input_masks'] = input_masks
        flatten_inputs['multimask_output'] = False
        flatten_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in flatten_inputs.items()}

        return flatten_inputs, values, idcs

    @staticmethod
    def reshape_tensor(tensor, size=(1024, 1024), is_3d=False):
        if is_3d:
            tensor = tensor.unsqueeze(dim=1)

        tensor = tensor.float()
        tensor = F.interpolate(
            tensor,
            size=size,
            mode='bilinear',
            align_corners=False
        ).squeeze(dim=1).half()

        return tensor

    def get_sam_input_masks(self, consistency_mask):
        values = torch.unique(consistency_mask).tolist()
        input_masks = F.one_hot(consistency_mask.to(torch.int64))
        input_masks = torch.permute(input_masks, (2, 0, 1))
        input_masks = input_masks[values]

        if len(values) > 1 and 0 in values:
            input_masks = input_masks[1:]
            input_masks = self.reshape_tensor(input_masks, size=(256, 256), is_3d=True)
            values.remove(0)
        elif len(values) == 1 and 0 in values:
            input_masks = torch.zeros((1, 256, 256), device=consistency_mask.device, dtype=torch.float16)
            values = [-1]
        else:
            input_masks = self.reshape_tensor(input_masks, size=(256, 256), is_3d=True)

        return input_masks, values

    @staticmethod
    def logits_to_masks(logits):
        mask = logits.argmax(dim=1)

        return mask

    @staticmethod
    def reshape_labels(inputs):
        labels = inputs['labels'].unsqueeze(dim=1)

        labels = nn.functional.interpolate(
            labels,
            size=inputs['pixel_values'].shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        labels = labels.squeeze(dim=1)

        return labels

    @staticmethod
    def reshape_outputs(inputs, outputs, return_mask=False):
        logits = outputs.logits

        outputs = nn.functional.interpolate(
            logits,
            size=inputs['pixel_values'].shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        if return_mask:
            outputs = outputs.argmax(dim=1)

        return outputs

    def update_loss_weights(self):
        current_epoch = self.current_epoch + 1
        self.delta_c = 0.1 * math.exp(-5 * (1 - current_epoch / self.config.max_epochs))
        self.delta_s = 0.1 * math.exp(-5 * (current_epoch / self.config.max_epochs))

    @torch.no_grad()
    def update_teacher(self, teacher_momentum=0.994):
        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_param.data = teacher_momentum * teacher_param.data + (1 - teacher_momentum) * student_param.data

    def log_segmentation_images(self, inputs, labels, masks):
        inputs = torch.moveaxis(inputs['pixel_values'][0], 0, -1).numpy(force=True)
        labels = labels[0].numpy(force=True)
        masks = masks[0].numpy(force=True)

        wandb.log({
            'val/segmentation': wandb.Image(
                inputs,
                masks={
                    'labels': {
                        'mask_data': labels,
                        'class_labels': self.class_labels,
                    },
                    'predictions': {
                        'mask_data': masks,
                        'class_labels': self.class_labels,
                    }
                }
            )
        })

    def log_consistency_images(self, inputs, masks, num):
        inputs = torch.moveaxis(inputs['pixel_values'][0], 0, -1).numpy(force=True)
        masks = masks[0].numpy(force=True)

        wandb.log({
            f'val/consistency_{num}': wandb.Image(
                inputs,
                masks={
                    'mask': {
                        'mask_data': masks,
                        'class_labels': self.class_labels,
                    }
                }
            )
        })

    def log_sam_images(self, inputs, consistency_masks, sam_masks):
        inputs = torch.moveaxis(inputs['pixel_values'][0], 0, -1).numpy(force=True)
        consistency_masks = consistency_masks[0].numpy(force=True)
        sam_masks = sam_masks[0].numpy(force=True)

        wandb.log({
            'val/sam': wandb.Image(
                inputs,
                masks={
                    'consistency': {
                        'mask_data': consistency_masks,
                        'class_labels': self.class_labels,
                    },
                    'sam': {
                        'mask_data': sam_masks,
                        'class_labels': self.class_labels,
                    }
                }
            )
        })

    @staticmethod
    def get_original_mask(masks):
        output_mask = torch.zeros_like(masks[0])

        # Iterate through the stacked binary mask tensors
        for index, mask in enumerate(masks):
            # Find the indices where the mask is True
            true_indices = torch.nonzero(mask, as_tuple=False)
            # Update the output tensor with the corresponding indices
            output_mask[true_indices[:, 0], true_indices[:, 1]] = index + 1

        return output_mask

    def train_dataloader(self):
        return DataLoader(
            dataset=ssp_dataset.make_train_dataset(self.config, self.labeled_tiles, self.unlabeled_tiles),
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=ssp_dataset.make_val_dataset(self.config, self.labeled_tiles, self.unlabeled_tiles),
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )


def load_student_model(config: Config):
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name_or_path=config.model_id,
        num_labels=config.num_labels,
        ignore_mismatched_sizes=True
    )

    return model


def load_teacher_model(config: Config):
    model = load_student_model(config)

    for param in model.parameters():
        param.requires_grad = False

    return model


def load_sam(config: Config):
    model = SamModel.from_pretrained(
        config.sam_id
    )

    processor = SamImageProcessor.from_pretrained(
        config.sam_id,
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
        do_convert_rgb=False
    )

    for param in model.parameters():
        param.requires_grad = False

    return model, processor


def load_model(config: Config, map_location=None):
    if config.checkpoint is None:
        lightning = SegFormerLightning(config)
    else:
        path_checkpoint = os.path.join(config.path.models, config.checkpoint)
        lightning = SegFormerLightning.load_from_checkpoint(path_checkpoint, config=config, map_location=map_location)

    return lightning


def _debug():
    config = func.load_config('main')
    wandb_config = func.load_config('segformer', 'semi_supervised')
    config.update(wandb_config)
    model = load_model(config)

    return


if __name__ == '__main__':
    _debug()
