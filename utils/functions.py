import os

import numpy as np
import wandb
import wandb.apis.public as wandb_api
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from utils import classes as uC


def get_notebooks_path(path: str) -> str:
    notebooks = os.path.join(os.pardir, path)
    new_path = path if os.path.exists(path) else notebooks

    return new_path


def load_config(yaml_file: str, mode: str = None, loading: str = 'object'):
    from utils.classes import Config

    if mode:
        root = os.path.join('configs', mode, f'{yaml_file}.yaml')
    else:
        root = os.path.join('configs', f'{yaml_file}.yaml')

    path = get_notebooks_path(root)

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    if loading == 'object':
        config = Config(config)

    return config


def init_wandb(yml_file: str, mode: str) -> dict:
    config = load_config('main')
    wandb_dir = get_notebooks_path(config['path']['logs'])
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ['WANDB_DIR'] = os.path.abspath(wandb_dir)
    wandb_config = load_config(yml_file, mode, loading='dict')

    wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
        config=wandb_config
    )

    return wandb.config


def get_run(run_id: str) -> wandb_api.Run:
    run = None

    if run_id:
        config = load_config('main')

        api = wandb.Api()
        run = wandb_api.Run(
            client=api.client,
            entity=config.wandb.entity,
            project=config.wandb.project,
            run_id=run_id,
        )

    return run


def load_image(config, tile: dict) -> Image.Image:
    path = os.path.join(config.path.data.raw.train.labeled, f'{tile["image"]}.JPG')

    with open(path, mode='br') as f:
        return Image.open(f).convert('RGB')  # TODO: crop using bbox


def load_label(config, tile: dict) -> np.ndarray:
    path = os.path.join(config.path.data.raw.train.labels, f'{tile["image"]}_gt.npy')

    with open(path, mode='br') as f:
        return np.load(f)  # TODO: crop using bbox


def train_val_split_tiles(config, tiles: list):
    images = list(set([tile['image'] for tile in tiles]))

    train_images, val_images = train_test_split(
        images,
        train_size=0.8,
        random_state=config.random_state,
    )

    train_tiles = [tile for tile in tiles if tile['image'] in train_images]
    val_tiles = [tile for tile in tiles if tile['image'] in val_images]

    return train_tiles, val_tiles
