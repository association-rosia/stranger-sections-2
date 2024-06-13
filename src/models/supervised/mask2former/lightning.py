import os

import pytorch_lightning as pl
import torch
import torchmetrics as tm
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import Mask2FormerForUniversalSegmentation

import src.data.supervised.dataset as spv_dataset
from src.data.collate import SS2Mask2formerCollateFn
from src.data.processor import SS2ImageProcessor
from src.utils import func
from src.utils.cls import Config


class Mask2FormerLightning(pl.LightningModule):
    def __init__(self, config: Config):
        super(Mask2FormerLightning, self).__init__()
        self.config = config
        self.model = _load_base_model(self.config)
        self.processor = SS2ImageProcessor.get_huggingface_processor(config)
        self.metrics = self.configure_metrics()
        self.class_labels = {0: 'Background', 1: 'Inertinite', 2: 'Vitrinite', 3: 'Liptinite'}
        self.best_metrics = {
            'best/dice-macro': 0,
            'best/dice-micro': 0,
            'best/iou-macro': 0,
            'best/iou-micro': 0,
        }

    def forward(self, inputs):
        outputs = self.model(**inputs)

        return outputs

    def training_step(self, batch):
        inputs = batch
        outputs = self.forward(inputs)
        loss = outputs['loss']
        self.log('train/loss', loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch
        outputs = self.forward(inputs)
        loss = outputs['loss']
        self.log('val/loss', loss, on_epoch=True)
        self.validation_log(inputs, outputs, batch_idx)

        return loss

    def on_validation_epoch_end(self) -> None:
        metrics = self.metrics.compute()
        self.best_metrics = {k.replace('val', 'best'): max(self.best_metrics[k.replace('val', 'best')], v) for k, v in metrics.items()}
        self.log_dict(self.best_metrics, on_epoch=True)
        self.log_dict(metrics, on_epoch=True)
        self.metrics.reset()

    def validation_log(self, inputs, outputs, batch_idx):
        target_sizes = [self.config.tile_size] * self.config.batch_size_supervised
        masks = self.processor.post_process_semantic_segmentation(outputs, target_sizes)
        ground_truth = self.inverse_process_mask_labels(inputs)

        if batch_idx == 0:
            self.log_image(inputs['pixel_values'][0], ground_truth[0], masks[0])

        self.metrics.update(torch.stack(masks), torch.stack(ground_truth))

    def log_image(self, pixel_values, ground_truth, mask):
        pixel_values = torch.moveaxis(pixel_values, 0, -1).numpy(force=True)
        mask = mask.numpy(force=True)
        ground_truth = ground_truth.numpy(force=True)

        wandb.log({
            'val/segmentation': wandb.Image(
                pixel_values,
                masks={
                    'labels': {
                        'mask_data': ground_truth,
                        'class_labels': self.class_labels,
                    },
                    'predictions': {
                        'mask_data': mask,
                        'class_labels': self.class_labels,
                    }
                }
            )
        })

    def inverse_process_mask_labels(self, inputs):
        mask_labels = inputs['mask_labels']
        class_labels = inputs['class_labels']
        reconstructed_labels = []
        for masks, labels in zip(mask_labels, class_labels):
            reconstructed_mask = torch.zeros(masks[0].shape, device=masks[0].device)
            for binary_mask, label in zip(masks, labels):
                reconstructed_mask += binary_mask * label
            reconstructed_labels.append(reconstructed_mask.to(dtype=torch.int8))

        return reconstructed_labels

    def configure_optimizers(self):
        optimizer = AdamW(params=self.model.parameters(), lr=self.config.lr_supervised)
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

    def configure_metrics(self):
        num_labels = self.config.num_labels

        metrics = tm.MetricCollection({
            'val/dice-macro': tm.Dice(num_classes=num_labels, average='macro'),
            'val/dice-micro': tm.Dice(num_classes=num_labels, average='micro'),
            'val/iou-macro': tm.JaccardIndex(task='multiclass', num_classes=num_labels, average='macro'),
            'val/iou-micro': tm.JaccardIndex(task='multiclass', num_classes=num_labels, average='micro'),
        })

        return metrics

    def train_dataloader(self):
        return DataLoader(
            dataset=spv_dataset.make_train_dataset(self.config),
            batch_size=self.config.batch_size_supervised,
            num_workers=self.config.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=SS2Mask2formerCollateFn(self.config, training=True)
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=spv_dataset.make_val_dataset(self.config),
            batch_size=self.config.batch_size_supervised,
            num_workers=self.config.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            collate_fn=SS2Mask2formerCollateFn(self.config, training=True)
        )


def _load_base_model(config: Config):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_model_name_or_path=config.model_id,
        num_labels=config.num_labels,
        ignore_mismatched_sizes=True
    )

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
    wandb_config = func.load_config('mask2former', 'segmentation')
    config.update(wandb_config)
    model = load_model(config)

    return


if __name__ == '__main__':
    _debug()
