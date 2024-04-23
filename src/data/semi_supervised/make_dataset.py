from torch.utils.data import Dataset

import src.data.processor as spv_processor
from src.data import tiling
from utils import classes as uC
from utils import func as uF


class SS2SemiSupervisedDataset(Dataset):
    def __init__(self,
                 config: uC.Config,
                 labeled_tiles: list,
                 unlabeled_tiles: list,
                 processor: spv_processor.SS2SupervisedProcessor
                 ):
        super().__init__()
        self.config = config
        self.labeled_tiles = labeled_tiles
        self.unlabeled_tiles = unlabeled_tiles
        self.processor = processor

    def __len__(self) -> int:
        return len(self.labeled_tiles)

    def __getitem__(self, idx):
        unlabeled_inputs = None

        labeled_image = uF.load_image(self.config, self.labeled_tiles[idx])
        label = uF.load_label(self.config, self.labeled_tiles[idx])
        labeled_inputs = self.processor.preprocess(labeled_image, label)

        unlabeled_idx = None

        return labeled_inputs, unlabeled_inputs


def make_train_dataset(config: uC.Config, labeled_tiles: list, unlabeled_tiles: list) -> SS2SemiSupervisedDataset:
    labeled_train_tiles, _ = uF.train_val_split_tiles(config, labeled_tiles)
    unlabeled_train_tiles, _ = uF.train_val_split_tiles(config, unlabeled_tiles)

    processor = spv_processor.make_training_processor(config)

    return SS2SemiSupervisedDataset(config, labeled_train_tiles, unlabeled_train_tiles, processor)


def make_val_dataset(config: uC.Config, labeled_tiles: list, unlabeled_tiles: list) -> SS2SemiSupervisedDataset:
    _, labeled_val_tiles = uF.train_val_split_tiles(config, labeled_tiles)
    _, unlabeled_val_tiles = uF.train_val_split_tiles(config, unlabeled_tiles)
    processor = spv_processor.make_eval_processor(config)

    return SS2SemiSupervisedDataset(config, labeled_val_tiles, unlabeled_val_tiles, processor)


def _debug():
    from torch.utils.data import DataLoader
    from src.data.collate import get_collate_fn_training

    config = uF.load_config('main')
    wandb_config = uF.load_config('segformer', 'semi_supervised')
    config = uC.Config.merge(config, wandb_config)

    labeled_tiles = tiling.main(labeled=True, size_tile=wandb_config.size_tile)
    unlabeled_tiles = tiling.main(labeled=False, size_tile=wandb_config.size_tile)

    train_dataset = make_train_dataset(config, labeled_tiles, unlabeled_tiles)
    val_dataset = make_val_dataset(config, labeled_tiles, unlabeled_tiles)

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
