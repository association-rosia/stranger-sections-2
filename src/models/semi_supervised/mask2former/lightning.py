import os
import math

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import Mask2FormerForUniversalSegmentation
import pytorch_lightning as pl

import torchmetrics as tm
import wandb
 
import src.data.semi_supervised.dataset as ssp_dataset
from src.data.processor import SS2ImageProcessor
from src.data import collate
from src.data.tiling import Tiler
from src.models.semi_supervised.mask2former.losses import SS2Mask2FormerLoss
from src.utils import func
from src.utils.cls import Config

torch.set_float32_matmul_precision('medium')


class Mask2FormerLightning(pl.LightningModule):
    def __init__(self, config: Config):
        super(Mask2FormerLightning, self).__init__()

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

        self.delta_c, self.delta_s = None, None
        self.update_loss_weights()

        self.segmentation_loss_fct = self.configure_criterion()
        self.consistency_loss_fct = self.configure_criterion() #SS2Mask2FormerLoss(self.student.config, )
        # self.sam_loss_fct = SS2Mask2FormerLoss(self.student.config)
        
        self.metrics = self.configure_metrics()

        self.current_step = None
        self.current_batch_idx = None

    def forward(self, segmentation_input, consistency_inputs):
        self.input_image_sizes = segmentation_input['pixel_values'].shape[-2:]

        segmentation_loss = self.segmentation_forward(segmentation_input)
        consistency_loss, consistency_outputs = self.consistency_forward(*consistency_inputs)
        # sam_loss = self.sam_forward(consistency_inputs, consistency_outputs)
        
        loss = segmentation_loss + self.delta_c * consistency_loss # + self.delta_s * sam_loss
        dict_loss = {
            'segmentation_loss': segmentation_loss,
            'consistency_loss': consistency_loss,
            'loss': loss
        }

        return dict_loss

    def training_step(self, batch):
        self.update_loss_weights()
        self.current_step = 'training'
        dict_loss = self.forward(*batch)
        self.update_teacher()

        dict_loss = {'train/' + k: v for k, v in dict_loss.items()}
        self.log_dict(dict_loss, on_epoch=True, sync_dist=True)

        return dict_loss['train/loss']

    def validation_step(self, batch, batch_idx):
        self.current_step = 'validation'
        self.current_batch_idx = batch_idx
        dict_loss = self.forward(*batch)
        dict_loss = {'val/' + k: v for k, v in dict_loss.items()}
        self.log_dict(dict_loss, on_epoch=True, sync_dist=True)

        return dict_loss['val/loss']

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

        return SS2Mask2FormerLoss(self.student.config, weight=weight)

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
        outputs = self.student.forward(
            pixel_values=inputs['pixel_values'],
            pixel_mask=inputs['pixel_mask'],
            output_auxiliary_logits=True
        )

        target_sizes = [self.config.tile_size] * self.config.batch_size
        outputs_processed = self.processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        masks = torch.stack(outputs_processed)
        labels = torch.stack(self.inverse_process_mask_labels(inputs))

        if self.current_step == 'validation':
            # masks = self.reshape_outputs(outputs, return_mask=True)
            self.metrics.update(masks, labels)

            if self.current_batch_idx == 0:
                self.log_segmentation_images(inputs, labels, masks)
        
        segmentation_loss = self.segmentation_loss_fct.forward(
            masks_queries_logits=outputs.masks_queries_logits,
            class_queries_logits=outputs.class_queries_logits,
            mask_labels=inputs['mask_labels'],
            class_labels=inputs['class_labels'],
            auxiliary_predictions=outputs.auxiliary_logits,
        )

        return segmentation_loss

    def consistency_forward(self, inputs_s, inputs_t):
        target_sizes = [self.config.tile_size] * self.config.batch_size

        outputs_s = self.student.forward(
            pixel_values=inputs_s['pixel_values'],
            pixel_mask=inputs_s['pixel_mask'],
            output_auxiliary_logits=True
        )

        outputs_t = self.teacher.forward(
            pixel_values=inputs_t['pixel_values'],
            pixel_mask=inputs_t['pixel_mask']
        )

        outputs_processed_t = self.processor.post_process_semantic_segmentation(outputs_t, target_sizes=target_sizes)
        masks_t = torch.stack(outputs_processed_t)

        if self.current_step == 'validation' and self.current_batch_idx == 0:
            outputs_processed_s = self.processor.post_process_semantic_segmentation(outputs_t, target_sizes=target_sizes)
            masks_s = torch.stack(outputs_processed_s)

            self.log_consistency_images(inputs_s, masks_s, 'student')
            self.log_consistency_images(inputs_t, masks_t, 'teacher')

        mask_labels, class_labels = [], []
        for mask_t in masks_t:
            device = mask_t.device
            mask_t = mask_t.numpy(force=True)
            binary_masks, labels = self.processor.convert_segmentation_map_to_binary_masks(mask_t)
            mask_labels.append(torch.from_numpy(binary_masks).to(device=device))
            class_labels.append(torch.from_numpy(labels).to(device=device))

        consistency_loss = self.consistency_loss_fct(
            masks_queries_logits=outputs_s.masks_queries_logits,
            class_queries_logits=outputs_s.class_queries_logits,
            mask_labels=mask_labels,
            class_labels=class_labels,
            auxiliary_predictions=outputs_s.auxiliary_logits,
        )

        return consistency_loss, outputs_s
    
    def inverse_process_mask_labels(self, inputs):
        mask_labels = inputs['mask_labels']
        class_labels = inputs['class_labels']
        reconstructed_labels = []
        for masks, labels in zip(mask_labels, class_labels):
            reconstructed_mask = masks[0].clone().detach()
            for binary_mask, label in zip(masks, labels):
                reconstructed_mask += binary_mask * label
            reconstructed_labels.append(reconstructed_mask.to(dtype=torch.int8))

        return reconstructed_labels

    def update_loss_weights(self):
        current_epoch = self.current_epoch + 1
        self.delta_c = 0.1 * math.exp(-5 * (1 - current_epoch / self.config.max_epochs))
        # self.delta_s = 0.1 * math.exp(-5 * (current_epoch / self.config.max_epochs))

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

    def log_input_masks(self, inputs, input_masks_i, classes_i):
        inputs = self.reshape_tensor(inputs['pixel_values'][0], self.input_masks_sizes, is_3d=True)
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
            collate_fn=collate.get_collate_fn_training(self.config)
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=ssp_dataset.make_val_dataset(self.config, self.labeled_tiles, self.unlabeled_tiles),
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate.get_collate_fn_training(self.config)
        )


def load_student_model(config: Config):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_model_name_or_path=config.model_id,
        num_labels=config.num_labels,
        ignore_mismatched_sizes=True
    )

    return model


def load_teacher_model(config: Config):
    with torch.no_grad():
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained_model_name_or_path=config.model_id,
            num_labels=config.num_labels,
            ignore_mismatched_sizes=True
        )

    for param in model.parameters():
        param.requires_grad = False

    return model


def load_model(config: Config, map_location=None):
    if config.checkpoint is None:
        lightning = Mask2FormerLightning(config)
    else:
        path_checkpoint = os.path.join(config.path.models, config.checkpoint)
        lightning = Mask2FormerLightning.load_from_checkpoint(path_checkpoint, config=config, map_location=map_location)

    return lightning


def _debug():
    config = func.load_config('main')
    wandb_config = func.load_config('segformer', 'semi_supervised')
    config.update(wandb_config)
    model = load_model(config)

    return


if __name__ == '__main__':
    _debug()
