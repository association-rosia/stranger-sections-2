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

        self.input_image_sizes = None
        self.student = load_student_model(self.config)
        self.teacher = load_teacher_model(self.config)

        self.input_masks_sizes = (256, 256)
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
        self.input_image_sizes = segmentation_input['pixel_values'].shape[-2:]
        segmentation_loss = self.segmentation_forward(segmentation_input)
        consistency_loss, consistency_logits_s = self.consistency_forward(consistency_inputs)
        sam_loss = self.sam_forward(consistency_inputs, consistency_logits_s)
        loss = segmentation_loss + self.delta_c * consistency_loss + self.delta_s * sam_loss

        return loss

    def training_step(self, batch):
        self.update_loss_weights()
        self.current_step = 'training'
        loss = self.forward(batch)
        self.update_teacher()
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
        logits = self.reshape_outputs(outputs)

        if self.current_step == 'validation':
            masks = self.reshape_outputs(outputs, return_mask=True)
            self.metrics.update(masks, labels)

            if self.current_batch_idx == 0:
                self.log_segmentation_images(inputs, labels, masks)

        loss = self.segmentation_loss_fct(logits, labels)
        self.log('val/segmentation_loss', loss, on_epoch=True, sync_dist=True)

        return loss

    def consistency_forward(self, inputs):
        inputs_s, inputs_t = inputs

        outputs_s = self.student(**inputs_s)
        logits_s = self.reshape_outputs(outputs_s)

        outputs_t = self.teacher(**inputs_t)
        logits_t = self.reshape_outputs(outputs_t)
        mask_t = self.logits_to_masks(logits_t)

        if self.current_step == 'validation' and self.current_batch_idx == 0:
            mask_s = self.logits_to_masks(logits_s)
            self.log_consistency_images(inputs_s, mask_s, 'student')
            self.log_consistency_images(inputs_t, mask_t, 'teacher')

        loss = self.consistency_loss_fct(logits_s, mask_t)
        self.log('val/consistency_loss', loss, on_epoch=True, sync_dist=True)

        return loss, logits_s

    @torch.no_grad()
    def sam_forward(self, inputs, consistency_logits):
        inputs, _ = inputs
        consistency_masks = self.logits_to_masks(consistency_logits)
        flatten_inputs, classes, indices = self.get_flatten_inputs(consistency_masks, inputs)
        flatten_outputs = self.sam_predict(flatten_inputs)
        sam_masks = self.post_process_flatten_outputs(flatten_inputs, flatten_outputs, classes, indices)

        if self.current_step == 'validation' and self.current_batch_idx == 0:
            self.log_sam_images(inputs, consistency_masks, sam_masks)

        loss = self.sam_loss_fct(consistency_logits, sam_masks.long())
        self.log('val/sam_loss', loss, on_epoch=True, sync_dist=True)

        return loss

    def get_flatten_inputs(self, consistency_masks, inputs):
        input_masks, pixel_values, classes, indices = [], [], [], []

        for i in range(self.config.batch_size):
            consistency_mask = consistency_masks[i]
            input_masks, input_masks_i, values = self.create_input_masks(consistency_mask, input_masks, classes)
            pixel_values, indices = self.create_pixel_values(input_masks_i, inputs, i, pixel_values, indices)

        flatten_inputs = self.create_flatten_inputs(consistency_masks, input_masks, pixel_values)

        return flatten_inputs, classes, indices

    def create_input_masks(self, consistency_mask, input_masks, classes):
        classes_i = torch.unique(consistency_mask).tolist()
        input_masks_i = F.one_hot(consistency_mask.to(torch.int64))
        input_masks_i = torch.permute(input_masks_i, (2, 0, 1))
        input_masks_i = input_masks_i[classes_i]

        if len(classes_i) > 1 and 0 in classes_i:
            input_masks_i = input_masks_i[1:]
            classes_i = classes_i[1:]
        elif classes_i == [0]:
            input_masks_i = torch.zeros(
                size=(1, self.input_masks_sizes[0], self.input_masks_sizes[1]),
                device=consistency_mask.device,
                dtype=consistency_mask.dtype
            )
            classes_i = [-1]

        input_masks_i = self.reshape_tensor(input_masks_i, size=self.input_masks_sizes, is_3d=True)
        input_masks.append(input_masks_i)
        classes += classes_i

        return input_masks, input_masks_i, classes

    def create_pixel_values(self, input_masks_i, inputs, i, pixel_values, indices):
        num_masks = len(input_masks_i) if input_masks_i is not None else 1
        pixel_values_i = torch.stack([inputs['pixel_values'][i] for _ in range(num_masks)])
        pixel_values_i = self.reshape_tensor(pixel_values_i)
        pixel_values.append(pixel_values_i)
        indices += [i for _ in range(num_masks)]

        return pixel_values, indices

    def create_flatten_inputs(self, consistency_masks, input_masks, pixel_values):
        input_masks = torch.cat(input_masks).unsqueeze(dim=1)
        pixel_values = torch.cat(pixel_values)
        flatten_inputs = self.sam_processor(images=pixel_values, return_tensors='pt')
        flatten_inputs['input_masks'] = input_masks
        flatten_inputs['multimask_output'] = False
        flatten_inputs = {k: v.to(consistency_masks.device) if isinstance(v, torch.Tensor) else v
                          for k, v in flatten_inputs.items()}

        return flatten_inputs

    def sam_predict(self, flatten_inputs):
        pred_masks = []
        flatten_inputs_size = len(flatten_inputs['pixel_values'])

        for start_idx in range(0, flatten_inputs_size, self.config.sam_batch_size):
            end_idx = min(start_idx + self.config.sam_batch_size, flatten_inputs_size)
            sam_batch = self.extract_sam_batch(flatten_inputs, start_idx, end_idx)
            flatten_outputs = self.sam(**sam_batch)
            pred_masks.append(flatten_outputs.pred_masks)

        pred_masks = torch.cat(pred_masks)

        return pred_masks

    @staticmethod
    def extract_sam_batch(flatten_inputs, start_idx, end_idx):
        sam_batch = {}

        for key, value in flatten_inputs.items():
            if isinstance(value, torch.Tensor):
                sam_batch[key] = value[start_idx:end_idx]
            else:
                sam_batch[key] = value

        return sam_batch

    def post_process_flatten_outputs(self, flatten_inputs, pred_masks, classes, batch_idx):
        sam_masks = []

        masks = self.sam_processor.post_process_masks(
            masks=pred_masks,
            original_sizes=flatten_inputs['original_sizes'],
            reshaped_input_sizes=flatten_inputs['reshaped_input_sizes'],
            binarize=False
        )
        masks = F.softmax(torch.cat(masks).squeeze(dim=1))

        for idx in range(self.config.sam_batch_size):
            sam_mask = []
            mask_batch_idx = masks[torch.Tensor(batch_idx) == idx]
            mask_class_idx = [classes[i] for i in range(len(batch_idx)) if batch_idx[i] == idx]

            class_idx_replaced = 0
            for label in range(self.config.num_labels):
                if label in mask_class_idx:
                    mask = (mask_batch_idx[class_idx_replaced] > self.config.sam_threshold).to(dtype=torch.float16)
                    sam_mask.append(mask)
                    class_idx_replaced += 1
                elif label == 0:
                    sam_mask.append(1e-8 * torch.ones(masks.shape[-2:], device=masks.device, dtype=torch.float16))
                else:
                    sam_mask.append(torch.zeros(masks.shape[-2:], device=masks.device, dtype=torch.float16))

            sam_mask = torch.stack(sam_mask)
            sam_mask = sam_mask.argmax(dim=0)
            sam_masks.append(sam_mask)

        sam_masks = torch.stack(sam_masks)
        sam_masks = self.reshape_tensor(sam_masks, size=self.input_image_sizes, is_3d=True)

        return sam_masks

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

    @staticmethod
    def logits_to_masks(logits):
        mask = logits.argmax(dim=1)

        return mask

    def reshape_labels(self, inputs):
        labels = inputs['labels'].unsqueeze(dim=1)

        labels = nn.functional.interpolate(
            labels,
            size=self.input_image_sizes,
            mode='bilinear',
            align_corners=False
        )

        labels = labels.squeeze(dim=1)

        return labels

    def reshape_outputs(self, outputs, return_mask=False):
        logits = outputs.logits

        outputs = nn.functional.interpolate(
            logits,
            size=self.input_image_sizes,
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
    with torch.no_grad():
        model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path=config.model_id,
            num_labels=config.num_labels,
            ignore_mismatched_sizes=True
        )

    for param in model.parameters():
        param.requires_grad = False

    return model


def load_sam(config: Config):
    with torch.no_grad():
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
