import torchvision.transforms.v2.functional as tv2F
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def _image_loader(path: str) -> Image.Image:
    with open(path, mode='br') as f:
        return Image.open(f).convert('RGB')


def _label_loader(path: str) -> np.ndarray:
    with open(path, mode='br') as f:
        return np.load(f)


def train_val_split(config: dict, image_paths: list, label_paths: list):
    train_image_paths, val_image_paths, train_label_paths, val_label_paths = train_test_split(
        image_paths, label_paths,
        train_size=0.8,
        random_state=config['random_state']
    )

    return train_image_paths, val_image_paths, train_label_paths, val_label_paths
