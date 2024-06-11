import warnings

warnings.filterwarnings('ignore')

from dotenv import load_dotenv

load_dotenv()

import pytorch_lightning as pl
import wandb

import torch
from src.utils import func
from src.utils.cls import Config
from src.models.train_model import load_model

torch.set_float32_matmul_precision('medium')


def main():
    config = func.load_config('main')
    wandb.init()
    config = Config(config, wandb.config)
    model = load_model(config)
    trainer = get_trainer(config)
    trainer.fit(model=model)
    wandb.finish()


def get_trainer(config: Config):
    if config.dry:
        trainer = pl.Trainer(
            max_epochs=2,
            logger=pl.loggers.WandbLogger(),
            devices=config.devices,
            precision='16-mixed',
            limit_train_batches=3,
            limit_val_batches=3,
            enable_checkpointing=False
        )
    else:
        trainer = pl.Trainer(
            devices=config.devices,
            max_epochs=config.max_epochs,
            logger=pl.loggers.WandbLogger(),
            precision='16-mixed',
            enable_checkpointing=False
        )

    return trainer


if __name__ == '__main__':
    main()
