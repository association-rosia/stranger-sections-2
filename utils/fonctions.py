import os

import wandb
import wandb.apis.public as wandb_api
import yaml

from utils.classes import Config


def get_notebooks_path(path: str) -> str:
    notebooks = os.path.join(os.pardir, path)
    new_path = path if os.path.exists(path) else notebooks

    return new_path


def load_config(yaml_file: str, mode: str = None, loading: str = 'object'):
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


def _image_loader(path: str) -> Image.Image:
    with open(path, mode='br') as f:
        return Image.open(f).convert('RGB')


def _label_loader(path: str) -> np.ndarray:
    with open(path, mode='br') as f:
        return np.load(f)


def train_val_split_tiles(config: uC.Config, tiles: list):
    train_tiles, val_tiles = train_test_split(
        tiles,
        train_size=0.8,
        random_state=config.random_state
    )

    return train_tiles, val_tiles
