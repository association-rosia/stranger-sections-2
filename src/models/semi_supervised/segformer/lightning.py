import math
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics as tm
import wandb
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

import src.data.semi_supervised.dataset as ssp_dataset
from src.data.processor import SS2ImageProcessor
from src.data.tiling import Tiler
from src.models.semi_supervised.sam import SamForSemiSupervised
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
        self.input_masks_sizes = (256, 256)

        self.student = load_student_model(config)
        self.teacher = load_teacher_model(config)
        self.sam = SamForSemiSupervised(config)

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
        labels = self.reshape_labels(inputs)  # TODO
        outputs = self.student(**inputs)
        logits = self.reshape_outputs(outputs)  # TODO

        loss = self.segmentation_loss_fct(logits, labels)

        if self.current_step == 'validation':
            self.log('val/segmentation_loss', loss, on_epoch=True, sync_dist=True)

            masks = self.reshape_outputs(outputs, return_mask=True)  # TODO
            self.metrics.update(masks, labels)

            if self.current_batch_idx == 0:
                self.log_segmentation_images(inputs, labels, masks)

        return loss

    def consistency_forward(self, inputs):
        inputs_s, inputs_t = inputs

        outputs_s = self.student(**inputs_s)
        logits_s = self.reshape_outputs(outputs_s)  # TODO

        outputs_t = self.teacher(**inputs_t)
        logits_t = self.reshape_outputs(outputs_t)  # TODO
        mask_t = self.logits_to_masks(logits_t)

        loss = self.consistency_loss_fct(logits_s, mask_t.long())

        if self.current_step == 'validation':
            self.log('val/consistency_loss', loss, on_epoch=True, sync_dist=True)

            if self.current_batch_idx == 0:
                mask_s = self.logits_to_masks(logits_s)
                self.log_consistency_images(inputs_s, mask_s, 'student')
                self.log_consistency_images(inputs_t, mask_t, 'teacher')

        return loss, logits_s

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

    def log_input_masks(self, inputs, input_masks_i, classes_i):
        inputs = self.reshape_tensor(inputs['pixel_values'][0], self.input_masks_sizes, is_3d=True)  # TODO
        inputs = torch.moveaxis(inputs, 0, -1).numpy(force=True)
        input_masks_i = input_masks_i.numpy(force=True)

        masks = {}
        class_idx_logged = 0
        for class_label in range(len(self.class_labels)):
            if class_label in classes_i:
                masks[f'input_masks_{class_label}'] = {'mask_data': input_masks_i[class_idx_logged]}
                class_idx_logged += 1
            else:
                masks[f'input_masks_{class_label}'] = {'mask_data': np.zeros(shape=self.input_masks_sizes)}

        masks = dict(sorted(masks.items()))

        wandb.log({
            'val/sam_input_masks': wandb.Image(
                inputs,
                masks=masks
            )
        })

    @staticmethod
    def log_output_masks(flatten_inputs, output_masks):
        inputs = torch.moveaxis(flatten_inputs['pixel_values'][0], 0, -1).numpy(force=True)
        output_masks = output_masks.numpy(force=True)
        masks = {f'output_masks_{i}': {'mask_data': output_masks[i]} for i in range(len(output_masks))}
        masks = dict(sorted(masks.items()))

        wandb.log({
            'val/sam_output_masks': wandb.Image(
                inputs,
                masks=masks
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
