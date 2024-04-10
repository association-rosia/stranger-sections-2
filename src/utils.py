import os

import wandb
import wandb.apis.public as wandb_api
import yaml


def get_notebooks_path(path: str) -> str:
    notebooks = os.path.join(os.pardir, path)
    new_path = path if os.path.exists(path) else notebooks

    return new_path


def load_config(yaml_file: str, sub_config: str = None):
    if sub_config:
        root = os.path.join('configs', sub_config, f'{yaml_file}.yaml')
    else:
        root = os.path.join('configs', f'{yaml_file}.yaml')

    path = get_notebooks_path(root)

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    config = Config(config)

    return config


class Config:
    def __init__(self, config_dict):
        if not isinstance(config_dict, dict):
            raise ValueError('Config must be initialized with a dictionary.')
        self.config = config_dict
        self._recursive_objectify(self.config)

    def _recursive_objectify(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = self._make_subconfig(value)

            setattr(self, key, value)

    @staticmethod
    def _make_subconfig(dictionary):
        subconfig = type('SubConfig', (), {})

        for k, v in dictionary.items():
            if isinstance(v, dict):
                v = Config._make_subconfig(v)

            setattr(subconfig, k, v)

        return subconfig

    def __repr__(self):
        return f'{self.__class__.__name__}({self.config})'


def get_config() -> dict:
    return load_config('config.yml')


def init_wandb(yml_file: str, mode: str) -> dict:
    config = get_config()
    wandb_dir = get_notebooks_path(config['path']['logs'])
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ['WANDB_DIR'] = os.path.abspath(wandb_dir)

    wandb_config = load_config(yml_file, mode)

    wandb.init(
        entity=config['wandb']['entity'],
        project=config['wandb']['project'],
        config=wandb_config
    )

    return wandb.config


def get_run(run_id: str) -> wandb_api.Run:
    run = None

    if run_id:
        project_config = get_config()

        api = wandb.Api()
        run = wandb_api.Run(
            client=api.client,
            entity=project_config['wandb']['entity'],
            project=project_config['wandb']['project'],
            run_id=run_id,
        )

    return run


class RunDemo:
    def __init__(self, config_file: str, id: str, name: str, sub_config: str = None) -> None:
        self.config = load_config(config_file, sub_config)
        self.name = name
        self.id = id
