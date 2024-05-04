import os
from glob import glob
from itertools import product

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.submission.model import SS2InferenceModel
from src.utils import func
from src.utils.cls import Config


def main():
    base_config = func.load_config('main')
    wandb_run = func.get_run('ujjdbhrh')
    submission_name = f'{wandb_run.name}-{wandb_run.id}'

    for checkpoint_type, tiling in product(['metric', 'loss'], [True, False]):
        # for checkpoint_type, tiling in product(['metric'], [False]):
        wandb_run.config['checkpoint'] = f'{wandb_run.name}-{wandb_run.id}-{checkpoint_type}.ckpt'
        config = Config(base_config, wandb_run.config)
        pathname = os.path.join(config.path.data.raw.test.unlabeled, '*.JPG')

        model = SS2InferenceModel.load_from_config(config, map_location='cuda:0', tiling=tiling)
        submission_folder = os.path.join(config.path.submissions, submission_name,
                                         f'{submission_name}-ckpt-{checkpoint_type}-tiling-{tiling}')
        os.makedirs(submission_folder, exist_ok=True)

        for image_path in tqdm(glob(pathname)):
            image = Image.open(image_path)
            mask_pred = model(image)
            mask_pred_name = os.path.basename(image_path).replace('.JPG', '_pred.npy')
            mask_pred_path = os.path.join(submission_folder, mask_pred_name)
            np.save(mask_pred_path, mask_pred)


if __name__ == '__main__':
    main()
