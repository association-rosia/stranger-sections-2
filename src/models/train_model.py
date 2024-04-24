import os

import pytorch_lightning as pl
import wandb

import src.models.supervised.mask2former.lightning as spv_m2f
from utils import func
from utils.cls import Config

import argparse


def main():
    model_name, mode = parse_args()
    config = func.load_config('main')
    wandb_config = func.init_wandb(model_name, mode)
    config = Config.merge(config, wandb_config)
    model = load_model(config)
    trainer = get_trainer(config)
    trainer.fit(model=model)
    wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()

    return args.model_name, args.mode


def load_model(config: Config, map_location=None):
    if config.mode == 'supervised':
        if config.model_name == 'mask2former':
            model = spv_m2f.load_model(config, map_location=map_location)

    if 'model' not in locals():
        raise ValueError(f"mode={config.mode} and model_name={config.model_name} doesn't exist.")

    return model


def get_trainer(config: Config):
    os.makedirs(config.path.models, exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor='val/loss',
        mode='min',
        dirpath=config.path.models,
        filename=f'{wandb.run.name}-{wandb.run.id}',
        auto_insert_metric_name=False,
        verbose=True
    )

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val/loss',
        patience=config.early_stopping_patience,
        verbose=True,
        mode='min'
    )

    if config.dry:
        trainer = pl.Trainer(
            max_epochs=2,
            logger=pl.loggers.WandbLogger(),
            callbacks=[checkpoint_callback],
            devices=config.devices,
            precision='16-mixed',
            limit_train_batches=3,
            limit_val_batches=3
        )
    elif len(config.devices) > 1:
        trainer = pl.Trainer(
            devices=config.devices,
            max_epochs=config.max_epochs,
            logger=pl.loggers.WandbLogger(),
            callbacks=[checkpoint_callback, early_stopping_callback],
            precision='16-mixed',
            strategy='ddp_find_unused_parameters_true',
            val_check_interval=config.val_check_interval
        )
    else:
        trainer = pl.Trainer(
            devices=config.devices,
            max_epochs=config.max_epochs,
            logger=pl.loggers.WandbLogger(),
            callbacks=[checkpoint_callback, early_stopping_callback],
            precision='16-mixed',
            val_check_interval=config.val_check_interval
        )

    return trainer


if __name__ == '__main__':
    main()
