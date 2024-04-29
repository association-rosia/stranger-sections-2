import os

import pytorch_lightning as pl
import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
from torch import nn

import src.data.semi_supervised.make_dataset as ssp_dataset
from src.data.processor import SS2ImageProcessor
from utils import func
from utils.cls import Config

from src.data import tiling


class SegFormerLightning(pl.LightningModule):
    def __init__(self, config: Config):
        super(SegFormerLightning, self).__init__()
        self.config = config

        self.student = load_student_model(self.config)
        self.teacher = load_teacher_model(self.config)
        self.sam = load_sam_model(self.config)

        # TODO: create our own loss function CrossEntropyLoss works with a logit and a mask (not 2 masks)
        self.consistency_loss_fct = nn.CrossEntropyLoss()
        self.sam_loss_fct = nn.CrossEntropyLoss()
        self.delta_c = 1
        self.delta_s = 1

        self.labeled_tiles = tiling.build(labeled=True, size_tile=self.config.size_tile)
        self.unlabeled_tiles = tiling.build(labeled=False, size_tile=self.config.size_tile)
        self.processor = SS2ImageProcessor.get_huggingface_processor(config)
        self.class_labels = {0: 'Background', 1: 'Inertinite', 2: 'Vitrinite', 3: 'Liptinite'}

    def forward(self, batch):
        segmentation_input, consistency_inputs = batch
        segmentation_loss = self.segmentation_forward(segmentation_input)
        consistency_loss, consistency_logits_1 = self.consistency_forward(consistency_inputs)
        sam_loss = self.sam_forward(consistency_inputs, consistency_logits_1)

        loss = segmentation_loss + self.delta_c * consistency_loss + self.delta_s * sam_loss

        return loss

    def training_step(self, batch):
        loss = self.forward(batch)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)

        # if batch_idx == 0:
        #     self.log_image(inputs, outputs)

        return loss

    def segmentation_forward(self, inputs):
        outputs = self.student(**inputs)

        # TODO: log the predicted mask
        # mask = self.outputs_to_mask(inputs, outputs)

        return outputs.loss  # CrossEntropyLoss

    def consistency_forward(self, inputs):
        inputs_1, inputs_2, = inputs

        outputs_1 = self.student(**inputs_1)
        logits_1 = self.reshape_outputs(inputs_1, outputs_1)

        outputs_2 = self.teacher(**inputs_2)
        logits_2 = self.reshape_outputs(inputs_2, outputs_2)
        mask_2 = self.logits_to_mask(logits_2)

        loss = self.consistency_loss_fct(logits_1, mask_2)

        return loss, logits_1

    def sam_forward(self, inputs, consistency_logits):
        consistency_mask = self.logits_to_mask(consistency_logits)

        inputs_1, inputs_2, = inputs

        loss = None

        return loss

    @staticmethod
    def logits_to_mask(logits):
        mask = logits.argmax(dim=1)

        return mask

    @staticmethod
    def reshape_outputs(inputs, outputs):
        logits = outputs.logits

        mask = nn.functional.interpolate(
            logits,
            size=inputs['pixel_values'].shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        return mask

    @torch.no_grad()
    def update_teacher(self, teacher_momentum=0.994):
        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_param.data = teacher_momentum * teacher_param.data + (1 - teacher_momentum) * student_param.data

    def log_image(self, inputs, outputs):
        pixel_values = torch.moveaxis(inputs['pixel_values'][0], 0, -1).numpy(force=True)
        outputs = self.processor.post_process_semantic_segmentation(outputs)
        outputs = outputs[0].numpy(force=True)
        ground_truth = self.get_original_mask(inputs['mask_labels'][0])
        ground_truth = ground_truth.numpy(force=True)

        wandb.log({
            'val/prediction': wandb.Image(pixel_values, masks={
                'predictions': {
                    'mask_data': outputs,
                    'class_labels': self.class_labels,
                },
                'ground_truth': {
                    'mask_data': ground_truth,
                    'class_labels': self.class_labels,
                }
            })
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


def load_sam_model(config: Config):
    return


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
