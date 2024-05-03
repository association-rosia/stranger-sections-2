import wandb
import os
import torch
import torch.nn.functional as tF
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from transformers import SegformerForSemanticSegmentation
import torchmetrics as tm

import src.data.supervised.dataset as spv_dataset
import src.data.collate as spv_collate
from src.utils.cls import Config


class SegformerLightning(pl.LightningModule):
    def __init__(self, config):
        super(SegformerLightning, self).__init__()
        self.config = config
        self.model = _load_base_model(self.config)
        self.criterion = self.configure_criterion()
        self.metrics = self.configure_metrics()

    def forward(self, pixel_values, labels=None):
        outputs = self.model(pixel_values=pixel_values, labels=labels)

        upsampled_logits = tF.interpolate(
            outputs.logits,
            size=pixel_values.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        return upsampled_logits.squeeze(1)

    def training_step(self, batch):
        logits = self.forward(**batch)
        loss = self.criterion(logits, batch['labels'])
        self.log('train/loss', loss, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(**batch)
        loss = self.criterion(logits, batch['labels'])
        self.log('val/loss', loss, on_epoch=True, sync_dist=True)
        self.validation_log(batch, batch_idx, logits)

        return loss

    def on_validation_epoch_end(self):
        metrics = self.metrics.compute()
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        self.metrics.reset()

    def configure_optimizers(self):
        optimizer = AdamW(params=self.model.parameters(), lr=self.config.lr)
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
        label_weights = self.config.label_weights.__dict__
        weight = torch.Tensor([label_weights[class_labels[str(i)]] for i in class_ordered])
        
        return torch.nn.CrossEntropyLoss(weight=weight)

    
    def configure_metrics(self):
        num_labels = self.config.num_labels

        metrics = tm.MetricCollection({
            'val/dice-macro': tm.Dice(num_classes=num_labels, average='macro'),
            'val/dice-micro': tm.Dice(num_classes=num_labels, average='micro'),
            'val/iou-macro': tm.JaccardIndex(task='multiclass', num_classes=num_labels, average='macro'),
            'val/iou-micro': tm.JaccardIndex(task='multiclass', num_classes=num_labels, average='micro'),
        })

        return metrics

    def validation_log(self, batch, batch_idx, logits):
        pred_masks = tF.sigmoid(logits).argmax(dim=1).type(torch.uint8)
        batch['labels'] = torch.where(batch['labels'] == 255, 0, batch['labels'])
        self.metrics.update(pred_masks, batch['labels'])

        if batch_idx == 0:
            self.log_image(batch, pred_masks)

    def log_image(self, batch, pred_masks):
        image = torch.moveaxis(batch['pixel_values'][0], 0, -1).numpy(force=True)
        pred_mask = pred_masks[0].numpy(force=True)
        ground_truth = batch['labels'][0].numpy(force=True)
        class_labels = self.config.data.class_labels.__dict__
        class_labels = {int(k): v for k, v in class_labels.items()}

        wandb.log(
            {'val/prediction': wandb.Image(image, masks={
                'predictions': {
                    'mask_data': pred_mask,
                    'class_labels': class_labels,
                },
                'ground_truth': {
                    'mask_data': ground_truth,
                    'class_labels': class_labels,
                }
            })})

    def train_dataloader(self):
        return DataLoader(
            dataset=spv_dataset.make_train_dataset(self.config),
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=spv_collate.get_collate_fn_training(self.config)
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=spv_dataset.make_val_dataset(self.config),
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            collate_fn=spv_collate.get_collate_fn_training(self.config)
        )


def _load_base_model(config: Config):
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name_or_path=config.model_id,
        num_labels=config.num_labels,
        ignore_mismatched_sizes=True
    )

    return model


def load_model(config, map_location=None):
    if config.checkpoint is None:
        lightning = SegformerLightning(config)
    else:
        path_checkpoint = os.path.join(config.path.models, config.checkpoint)
        lightning = SegformerLightning.load_from_checkpoint(path_checkpoint, config=config, map_location=map_location)

    return lightning


def _debug():
    from utils import func
    config = func.load_config('main')
    wandb_config = func.load_config('segformer', 'supervised')
    config = Config(config, wandb_config)
    model = load_model(config)

    return


if __name__ == '__main__':
    _debug()