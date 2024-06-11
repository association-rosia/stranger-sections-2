import torch
from torch.utils.data import Dataset

from src.data.processor import SS2ImageProcessor, AugmentationMode, PreprocessingMode
from src.data.tiling import Tiler
from src.utils import func
from src.utils.cls import Config


class SS2SupervisedDataset(Dataset):
    def __init__(self, config: Config, tiles: list):
        super().__init__()
        self.config = config
        self.tiles = tiles
        self.processor = SS2ImageProcessor(
            self.config,
            preprocessing_mode=PreprocessingMode.PHOTOMETRIC,
            augmentation_mode=AugmentationMode.NONE
        )

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx):
        image = func.load_supervised_image(self.config, self.tiles[idx])
        label = func.load_label(self.config, self.tiles[idx])
        inputs = self.processor.preprocess(image, label)

        inputs = self.adjust_shape(inputs)
        inputs['pixel_values'] = inputs['pixel_values'].to(dtype=torch.float16)

        return inputs

    @staticmethod
    def adjust_shape(inputs):
        inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}

        return inputs


def make_train_dataset(config: Config) -> SS2SupervisedDataset:
    tiler = Tiler(config)
    tiles = tiler.build(config)
    train_tiles, _ = func.train_val_split_tiles(config, tiles)

    return SS2SupervisedDataset(config, train_tiles)


def make_val_dataset(config: Config) -> SS2SupervisedDataset:
    tiler = Tiler(config)
    tiles = tiler.build(config)
    _, val_tiles = func.train_val_split_tiles(config, tiles)

    return SS2SupervisedDataset(config, val_tiles)


def _debug():
    from torch.utils.data import DataLoader
    from src.data.collate import SS2Mask2formerCollateFn

    config = func.load_config('main')
    wandb_config = func.load_config('segformer', 'supervised')
    config = Config(config, wandb_config)

    train_dataset = make_train_dataset(config)
    val_dataset = make_val_dataset(config)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=12,
        collate_fn=SS2Mask2formerCollateFn(config)
    )

    for batch in train_dataloader:
        break

    return


if __name__ == '__main__':
    _debug()
