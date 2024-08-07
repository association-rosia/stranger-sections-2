import os
from glob import glob
from itertools import product

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.submissions.model import SS2InferenceModel
from src.utils import func
from src.utils.cls import Config, TrainingMode


def main():
    base_config = func.load_config('main')
    wandb_run = func.get_run('kyxcee1p') #  jaom0qef
    submission_name = f'{wandb_run.name}-{wandb_run.id}'
    device = 'cuda:0'
    tile_sizes = [
        wandb_run.config['tile_size'],
        # 512
    ]
    checkpoint_types = [
        'spv-v1',
    ]
    tta_ks = [
        # 1,
        'max',
    ]

    for checkpoint_type, tile_size, tta_k in product(checkpoint_types, tile_sizes, tta_ks):
        wandb_run.config['checkpoint'] = f'{wandb_run.name}-{wandb_run.id}-{checkpoint_type}.ckpt'
        wandb_run.config['mode'] =  TrainingMode.SEMI_SUPERVISED
        config = Config(base_config, wandb_run.config)
        pathname = os.path.join(config.path.data.raw.test.unlabeled, '*.JPG')

        model = SS2InferenceModel(
            config,
            map_location=device,
            tile_size=tile_size,
            tta_k=tta_k
        )

        submission_folder = os.path.join(
            config.path.submissions,
            submission_name,
            f'{submission_name}-ckpt-{checkpoint_type}-tiling-{tile_size}-tta-{tta_k}'
        )

        os.makedirs(submission_folder, exist_ok=True)

        for image_path in tqdm(glob(pathname)):
            image = Image.open(image_path)
            mask_pred = model(image)
            mask_pred = mask_pred.numpy(force=True)
            mask_pred_name = os.path.basename(image_path).replace('.JPG', '_pred.npy')
            mask_pred_path = os.path.join(submission_folder, mask_pred_name)
            np.save(mask_pred_path, mask_pred)


if __name__ == '__main__':
    main()
