import math
import os

import pytorch_lightning as pl
import torch
import torchmetrics as tm
import wandb
from torch.nn import CrossEntropyLoss
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

        self.segmentation_loss_fct = self.configure_segmentation_loss_fct()
        self.consistency_loss_fct = CrossEntropyLoss()
        self.sam_loss_fct = CrossEntropyLoss()
        self.metrics = self.configure_metrics()

        self.student = load_student_model(config)
        self.teacher = load_teacher_model(config)
        self.sam = SamForSemiSupervised(config, class_labels=self.class_labels, loss_fct=self.sam_loss_fct)

        self.delta_c, self.delta_s = None, None
        self.update_loss_weights()

        self.current_step = None
        self.current_batch_idx = None

    def forward(self, batch):
        segmentation_input, consistency_inputs = batch
        self.input_image_sizes = segmentation_input['pixel_values'].shape[-2:]

        segmentation_loss = self.segmentation_forward(segmentation_input)
        consistency_loss, consistency_logits_student = self.consistency_forward(consistency_inputs)
        sam_loss = self.sam_forward(consistency_inputs, consistency_logits_student)

        loss = segmentation_loss + self.delta_c * consistency_loss + self.delta_s * sam_loss

        return loss

    def training_step(self, batch):
        self.update_loss_weights()
        self.current_step = 'training'
        loss = self.forward(batch)
        self.update_teacher()
        self.log('train/loss', loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self.current_step = 'validation'
        self.current_batch_idx = batch_idx
        loss = self.forward(batch)
        self.log('val/loss', loss, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        metrics = self.metrics.compute()
        self.log_dict(metrics, on_epoch=True)
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

    def configure_segmentation_loss_fct(self):
        class_labels = self.config.data.class_labels.__dict__
        class_ordered = sorted([int(k) for k in class_labels.keys()])
        label_weights = self.config.data.label_weights.__dict__
        weight = torch.Tensor([label_weights[class_labels[str(i)]] for i in class_ordered])

        return CrossEntropyLoss(weight=weight)

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
        outputs = self.student(**inputs)
        logits = func.reshape_tensor(outputs.logits, size=self.input_image_sizes)
        labels = func.reshape_tensor(inputs['labels'], size=self.input_image_sizes)

        loss = self.segmentation_loss_fct(logits, labels)

        if self.current_step == 'validation':
            self.log('val/segmentation_loss', loss, on_epoch=True)
            masks = func.logits_to_masks(logits)
            self.metrics.update(masks, labels)

            if self.current_batch_idx == 0:
                self.log_segmentation_images(inputs, labels, masks)

        return loss

    def consistency_forward(self, inputs):
        inputs_student, inputs_teacher = inputs

        outputs_student = self.student(**inputs_student)
        logits_student = func.reshape_tensor(outputs_student.logits, size=self.input_image_sizes)

        outputs_teacher = self.teacher(**inputs_teacher)
        logits_teacher = func.reshape_tensor(outputs_teacher.logits, size=self.input_image_sizes)
        mask_teacher = func.logits_to_masks(logits_teacher)

        loss = self.consistency_loss_fct(logits_student, mask_teacher.long())

        if self.current_step == 'validation':
            self.log('val/consistency_loss', loss, on_epoch=True)

            if self.current_batch_idx == 0:
                mask_student = func.logits_to_masks(logits_student)
                self.log_consistency_images(inputs_student, mask_student, 'student')
                self.log_consistency_images(inputs_teacher, mask_teacher, 'teacher')

        return loss, logits_student

    def sam_forward(self, inputs, logits_student):
        self.sam.current_step = self.current_step
        self.sam.current_batch_idx = self.current_batch_idx

        loss, consistency_masks, sam_masks = self.sam.forward(inputs[0], logits_student)

        if self.current_step == 'validation':
            self.log('val/sam_loss', loss, on_epoch=True)

            if self.current_batch_idx == 0:
                self.log_sam_images(inputs[0], consistency_masks, sam_masks)

        return loss

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
