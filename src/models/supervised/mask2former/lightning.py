import os

import pytorch_lightning as pl
import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import Mask2FormerForUniversalSegmentation

import src.data.supervised.collate as spv_collate
import src.data.supervised.dataset as spv_dataset
from src.data.supervised.processor import SS2SupervisedProcessor
from utils import classes as uC


class Mask2FormerLightning(pl.LightningModule):
    def __init__(self, config: uC.Config):
        super(Mask2FormerLightning, self).__init__()
        self.config = config
        self.model = _load_base_model(self.config)
        self.processor = SS2SupervisedProcessor.get_huggingface_processor(config)
        self.class_labels = {0: 'Background', 1: 'Inertinite', 2: 'Vitrinite', 3: 'Liptinite'}

    def forward(self, inputs):
        outputs = self.model(**inputs)

        return outputs

    def training_step(self, batch):
        inputs = batch
        outputs = self.forward(inputs)
        loss = outputs['loss']
        self.log('train/loss', loss, on_epoch=True, sync_dist=True, batch_size=self.config.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch
        outputs = self.forward(inputs)
        loss = outputs['loss']
        self.log('val/loss', loss, on_epoch=True, sync_dist=True, batch_size=self.config.batch_size)

        if batch_idx == 0:
            self.log_image(inputs, outputs)

        return loss

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

        # Parcourez les tensors masques binaires empilés
        for index, mask in enumerate(masks):
            # Trouvez les indices où le masque est True
            true_indices = torch.nonzero(mask, as_tuple=False)
            # Mettez à jour le tensor de sortie avec les indices correspondants
            output_mask[true_indices[:, 0], true_indices[:, 1]] = index + 1

        return output_mask

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


def _load_base_model(config: uC.Config):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_model_name_or_path=config.model_id,
        num_labels=config.num_labels,
        # class_weight=1.0,
        # mask_weight=1.0,
        # dice_weight=10.0,
        ignore_mismatched_sizes=True
    )

    return model


def load_model(config, map_location=None):
    if config.checkpoint is None:
        lightning = Mask2FormerLightning(config)
    else:
        path_checkpoint = os.path.join(config.path.models, config.checkpoint)
        lightning = Mask2FormerLightning.load_from_checkpoint(path_checkpoint, config=config, map_location=map_location)

    return lightning


def _debug():
    config = utils.get_config()
    wandb_config = utils.load_config('mask2former', 'segmentation')
    config.update(wandb_config)
    model = load_model(config)

    return


if __name__ == '__main__':
    _debug()
