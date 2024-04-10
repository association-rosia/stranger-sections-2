import os
from glob import glob

from torch.utils.data import Dataset

from src import utils
import src.data.utils as dt_utils
import src.data.supervised.processor as spv_processor

class SupervisedDataset(Dataset):
    def __init__(self, config: dict, image_paths: list, label_paths: list, processor: spv_processor.SupervisedProcessor) -> None:
        super().__init__()
        self.config = config
        self.processor = processor
        self.images = image_paths
        self.labels = label_paths

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx):
        image = dt_utils._image_loader(self.images[idx])
        label = dt_utils._label_loader(self.labels[idx])

        inputs = self.processor.preprocess(image, label)

        return inputs


def make_train_dataset(config) -> SupervisedDataset:
    data_folder = utils.get_notebooks_path(os.path.join(config['path']['data']))

    pathname_images = os.path.join(data_folder, 'raw', 'train', 'labeled', '*.JPG')
    image_paths = sorted(glob(pathname_images))

    pathname_labels = os.path.join(data_folder, 'raw', 'train', 'labels', '*.npy')
    label_paths = sorted(glob(pathname_labels))

    train_image_paths, _, train_label_paths, _ = dt_utils.train_val_split(config, image_paths, label_paths)
    processor = spv_processor.make_training_processor(config)

    return SupervisedDataset(config, train_image_paths, train_label_paths, processor)


def make_val_dataset(config) -> SupervisedDataset:
    data_folder = utils.get_notebooks_path(os.path.join(config['path']['data']))
    
    pathname_images = os.path.join(data_folder, 'raw', 'train', 'labeled', '*.JPG')
    image_paths = sorted(glob(pathname_images))
   
    pathname_labels = os.path.join(data_folder, 'raw', 'train', 'labels', '*.npy')
    label_paths = sorted(glob(pathname_labels))
    
    _, val_image_paths, _, val_label_paths = dt_utils.train_val_split(config, image_paths, label_paths)
    processor = spv_processor.make_eval_processor(config)

    return SupervisedDataset(config, val_image_paths, val_label_paths, processor)


def _debug():
    from src import utils
    from torch.utils.data import DataLoader
    from src.data.supervised.collate import get_collate_fn

    config = utils.get_config()
    wandb_config = utils.load_config('mask2former.yml', 'segmentation')
    config.update(wandb_config)

    train_dataset = make_train_dataset(config)
    val_dataset = make_val_dataset(config)

    for batch in DataLoader(train_dataset, batch_size=12, collate_fn=get_collate_fn(config)):
        break

    return


if __name__ == '__main__':
    _debug()