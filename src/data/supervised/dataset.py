import os
from glob import glob

from torch.utils.data import Dataset

import src.data.supervised.processor as spv_processor
import src.data.utils as dt_utils
from src import utils
import src.data.tiling


class SS2SupervisedDataset(Dataset):
    def __init__(self, config: uC.Config, tiles: list, processor: spv_processor.SS2SupervisedProcessor):
        super().__init__()
        self.config = config
        self.tiles = tiles
        self.processor = processor

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx):
        image = dt_utils._image_loader(self.images[idx])
        label = dt_utils._label_loader(self.labels[idx])
        inputs = self.processor.preprocess(image, label)

        return inputs


def make_train_dataset(config: uC.Config) -> SS2SupervisedDataset:
    tiles = tiling.get_tiles()
    train_tiles, _ = dt_utils.train_val_split_tiles(config, tiles)
    processor = spv_processor.make_training_processor(config)

    return SS2SupervisedDataset(config, tiles, processor)


def make_val_dataset(config: uC.Config) -> SS2SupervisedDataset:
    tiles = tiling.get_tiles()
    _, val_tiles = dt_utils.train_val_split_tiles(config, tiles)
    processor = spv_processor.make_eval_processor(config)

    return SS2SupervisedDataset(config, tiles, processor)


def _debug():
    from src import utils
    from torch.utils.data import DataLoader
    from src.data.supervised.collate import get_collate_fn_training

    config = utils.get_config()
    wandb_config = utils.load_config('mask2former.yml', 'segmentation')
    config.update(wandb_config)

    train_dataset = make_train_dataset(config)
    val_dataset = make_val_dataset(config)

    train_dataloader = DataLoader(train_dataset, batch_size=12, collate_fn=get_collate_fn_training(config))

    for batch in train_dataloader:
        break

    return


if __name__ == '__main__':
    _debug()