from torch.utils.data import Dataset

import src.data.semi_supervised.processor as spv_processor
from src.data import tiling
from utils import classes as uC
from utils import functions as uF


class SS2SemiSupervisedDataset(Dataset):
    def __init__(self, config: uC.Config, tiles: list, processor: spv_processor.SS2SupervisedProcessor):
        super().__init__()
        self.config = config
        self.tiles = tiles
        self.processor = processor

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx):
        unlabeled_image = None

        labeled_image = uF.load_image(self.config, self.tiles[idx])
        label = uF.load_label(self.config, self.tiles[idx])
        labeled_inputs = self.processor.preprocess(labeled_image, label)

        inputs = None

        return inputs


def make_train_dataset(config: uC.Config) -> SS2SemiSupervisedDataset:
    labeled_tiles = tiling.main()
    labeled_train_tiles, _ = uF.train_val_split_tiles(config, labeled_tiles)
    processor = spv_processor.make_training_processor(config)

    return SS2SemiSupervisedDataset(config, labeled_train_tiles, processor)


def make_val_dataset(config: uC.Config) -> SS2SemiSupervisedDataset:
    labeled_tiles = tiling.main()
    _, labeled_val_tiles = uF.train_val_split_tiles(config, labeled_tiles)
    processor = spv_processor.make_eval_processor(config)

    return SS2SemiSupervisedDataset(config, labeled_val_tiles, processor)


def _debug():
    from torch.utils.data import DataLoader
    from src.data.supervised.collate import get_collate_fn_training

    config = uF.load_config('main')
    wandb_config = uF.load_config('segformer', 'semi_supervised')
    config = uC.Config.merge(config, wandb_config)

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
