import os
from glob import glob
from itertools import product

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data.processor import AugmentationMode
from src.submissions.model import SS2InferenceModel
from src.submissions.tta import TestTimeAugmenter
from src.utils import func
from src.utils.cls import Config


def main():
    base_config = func.load_config('main')
    wandb_run = func.get_run('mlmyc2ql') #  jaom0qef
    submission_name = f'{wandb_run.name}-{wandb_run.id}'
    device = 'cuda:1'
    tile_sizes = [
        wandb_run.config['tile_size'],
        # 512
    ]
    checkpoint_types = [
        'micro',
        'macro',
    ]
    tta_ks = [
        1,
        'max',
    ]

    for checkpoint_type, tile_size, k in product(checkpoint_types, tile_sizes, tta_ks):
        # for checkpoint_type, tiling in product(['metric'], [False]):
        wandb_run.config['checkpoint'] = f'{wandb_run.name}-{wandb_run.id}-{checkpoint_type}.ckpt'
        config = Config(base_config, wandb_run.config)
        pathname = os.path.join(config.path.data.raw.test.unlabeled, '*.JPG')

        test_time_augmenter = TestTimeAugmenter(
            k=k,
            random_state=42
        )

        model = SS2InferenceModel(
            config,
            map_location=device,
            tile_size=tile_size,
            test_time_augmenter=test_time_augmenter
        )

        submission_folder = os.path.join(
            config.path.submissions,
            submission_name,
            f'{submission_name}-ckpt-{checkpoint_type}-tiling-{tile_size}-tta-{k}'
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
