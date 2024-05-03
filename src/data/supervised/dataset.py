from torch.utils.data import Dataset

import src.data.processor as processor
from src.data import tiling
from src.utils import func
from src.utils.cls import Config


class SS2SupervisedDataset(Dataset):
    def __init__(self, config: Config, tiles: list, processor: processor.SS2ImageProcessor):
        super().__init__()
        self.config = config
        self.tiles = tiles
        self.processor = processor

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx):
        image = func.load_supervised_image(self.config, self.tiles[idx])
        label = func.load_label(self.config, self.tiles[idx])
        inputs = self.processor.preprocess(image, label)

        return inputs


def make_train_dataset(config: Config) -> SS2SupervisedDataset:
    tiles = tiling.build()
    train_tiles, _ = func.train_val_split_tiles(config, tiles)
    # processor = processor.make_training_processor(config)

    return SS2SupervisedDataset(config, train_tiles, processor)


def make_val_dataset(config: Config) -> SS2SupervisedDataset:
    tiles = tiling.build(config)
    _, val_tiles = func.train_val_split_tiles(config, tiles)
    # processor = processor.make_eval_processor(config)

    return SS2SupervisedDataset(config, val_tiles, processor)


def _debug():
    from torch.utils.data import DataLoader
    from src.data.collate import get_collate_fn_training

    config = func.load_config('main')
    wandb_config = func.load_config('mask2former', 'supervised')
    config = Config(config, wandb_config)

    train_dataset = make_train_dataset(config)
    val_dataset = make_val_dataset(config)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=12,
        collate_fn=get_collate_fn_training(config)
    )

    for batch in train_dataloader:
        break

    return


if __name__ == '__main__':
    _debug()
