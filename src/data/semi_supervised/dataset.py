import random

import torch
from torch.utils.data import Dataset

from src.data.processor import SS2ImageProcessor, AugmentationMode
from src.data.tiling import Tiler
from src.utils import func
from src.utils.cls import Config


class SS2SemiSupervisedDataset(Dataset):
    def __init__(self, config: Config, labeled_tiles: list, unlabeled_tiles: list):
        super().__init__()
        self.config = config
        self.labeled_tiles = labeled_tiles
        self.unlabeled_tiles = unlabeled_tiles
        self.processor = SS2ImageProcessor(self.config)

    def __len__(self) -> int:
        return len(self.labeled_tiles)

    def __getitem__(self, idx):
        supervised_input = self.get_supervised_input(idx)
        unsupervised_inputs = self.get_unsupervised_inputs()

        return supervised_input, unsupervised_inputs

    @staticmethod
    def adjust_shape(inputs):
        inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}

        return inputs

    def get_supervised_input(self, idx):
        images = func.load_supervised_image(self.config, self.labeled_tiles[idx])
        labels = func.load_label(self.config, self.labeled_tiles[idx])

        inputs = self.processor.preprocess(
            images=images,
            labels=labels,
            augmentation_mode=AugmentationMode.BOTH,
            apply_huggingface=True
        )
        inputs = self.adjust_shape(inputs)
        inputs['pixel_values'] = inputs['pixel_values'].to(dtype=torch.float16)

        return inputs

    def get_unsupervised_inputs(self):
        idx = random.randint(0, len(self.unlabeled_tiles) - 1)
        images = func.load_unsupervised_image(self.config, self.unlabeled_tiles[idx])

        images = self.processor.preprocess(
            images=images,
            augmentation_mode=AugmentationMode.GEOMETRIC,
            apply_huggingface=False,
        )

        inputs_student = self.processor.preprocess(
            images=images,
            augmentation_mode=AugmentationMode.PHOTOMETRIC,
            apply_huggingface=True,
        )
        inputs_student = self.adjust_shape(inputs_student)
        inputs_student['pixel_values'] = inputs_student['pixel_values'].to(dtype=torch.float16)

        inputs_teacher = self.processor.preprocess(
            images=images,
            augmentation_mode=AugmentationMode.PHOTOMETRIC,
            apply_huggingface=True,
        )
        inputs_teacher = self.adjust_shape(inputs_teacher)
        inputs_teacher['pixel_values'] = inputs_teacher['pixel_values'].to(dtype=torch.float16)

        return inputs_student, inputs_teacher


def make_train_dataset(config: Config, labeled_tiles: list, unlabeled_tiles: list) -> SS2SemiSupervisedDataset:
    labeled_train_tiles, _ = func.train_val_split_tiles(config, labeled_tiles)
    unlabeled_train_tiles, _ = func.train_val_split_tiles(config, unlabeled_tiles)

    return SS2SemiSupervisedDataset(config, labeled_train_tiles, unlabeled_train_tiles)


def make_val_dataset(config: Config, labeled_tiles: list, unlabeled_tiles: list) -> SS2SemiSupervisedDataset:
    _, labeled_val_tiles = func.train_val_split_tiles(config, labeled_tiles)
    _, unlabeled_val_tiles = func.train_val_split_tiles(config, unlabeled_tiles)

    return SS2SemiSupervisedDataset(config, labeled_val_tiles, unlabeled_val_tiles)


def _debug():
    from torch.utils.data import DataLoader
    from src.data.collate import SS2Mask2formerCollateFn

    config = func.load_config('main')
    wandb_config = func.load_config('mask2former', 'semi_supervised')
    config = Config(config, wandb_config)

    tiler = Tiler(config=config)
    labeled_tiles = tiler.build(labeled=True)
    unlabeled_tiles = tiler.build(labeled=False)

    train_dataset = make_train_dataset(config, labeled_tiles, unlabeled_tiles)
    val_dataset = make_val_dataset(config, labeled_tiles, unlabeled_tiles)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=12,
        collate_fn=SS2Mask2formerCollateFn(config, training=True)
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=12,
        collate_fn=SS2Mask2formerCollateFn(config, training=True)
    )

    for supervised_batch, unsupervised_batch in train_dataloader:
        continue

    for supervised_batch, unsupervised_batch in val_dataloader:
        continue

    return


if __name__ == '__main__':
    _debug()
