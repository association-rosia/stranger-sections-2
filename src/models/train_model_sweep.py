import warnings

warnings.filterwarnings('ignore')

from dotenv import load_dotenv

load_dotenv()

import pytorch_lightning as pl
import wandb

import torch
from src.utils import func
from src.utils.cls import Config, TrainingMode
from src.models.train_model import load_model

torch.set_float32_matmul_precision('medium')


def main():
    wandb.init()
    base_config = func.load_config('main')
    spv_config = Config(base_config, wandb.config, {'checkpoint': None, 'mode': TrainingMode.SUPERVISED})
    
    model = load_model(spv_config)
    trainer = get_trainer_supervised(spv_config)
    trainer.fit(model=model)
    del model, trainer

    ckpt_config = Config(base_config, wandb.config, {'checkpoint': f'{wandb.run.name}-{wandb.run.id}-spv.ckpt', 'mode': TrainingMode.SUPERVISED})
    ckpt_model = load_model(ckpt_config)
    
    ssp_config = Config(base_config, wandb.config, {'checkpoint': None, 'mode': TrainingMode.SEMI_SUPERVISED})
    model = load_model(ssp_config)
    model.student = ckpt_model.model
    model.teacher = ckpt_model.model
    trainer = get_trainer_semi_supervised(ssp_config)
    trainer.fit(model=model)

    wandb.finish()


def get_trainer_supervised(config: Config):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor='val/iou-micro',
        mode='max',
        dirpath=config.path.models,
        filename=f'{wandb.run.name}-{wandb.run.id}-spv',
        auto_insert_metric_name=False,
        verbose=False
    )

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val/iou-micro',
        mode='max',
        patience=config.early_stopping_patience,
        verbose=False
    )

    if config.dry:
        trainer = pl.Trainer(
            max_epochs=2,
            accelerator='gpu',
            precision='16-mixed',
            logger=pl.loggers.WandbLogger(),
            callbacks=[checkpoint_callback],
            limit_train_batches=3,
            limit_val_batches=3,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=config.max_epochs,
            accelerator='gpu',
            precision='16-mixed',
            logger=pl.loggers.WandbLogger(),
            callbacks=[checkpoint_callback, early_stopping_callback],
        )

    return trainer


def get_trainer_semi_supervised(config: Config):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor='val/iou-micro',
        mode='max',
        dirpath=config.path.models,
        filename=f'{wandb.run.name}-{wandb.run.id}-ssp',
        auto_insert_metric_name=False,
        verbose=True
    )

    if config.dry:
        trainer = pl.Trainer(
            max_epochs=2,
            accelerator='gpu',
            precision='16-mixed',
            logger=pl.loggers.WandbLogger(),
            callbacks=[checkpoint_callback],
            limit_train_batches=3,
            limit_val_batches=3,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=config.max_epochs,
            accelerator='gpu',
            precision='16-mixed',
            logger=pl.loggers.WandbLogger(),
            callbacks=[checkpoint_callback],
        )

    return trainer

if __name__ == '__main__':
    main()
