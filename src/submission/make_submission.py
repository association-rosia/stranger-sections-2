import os
from glob import glob
from tqdm import tqdm

import numpy as np
from PIL import Image

from src.submission.model import InferenceModel
from utils import functions as uF
from utils import classes as uC

import warnings

warnings.filterwarnings('ignore')


def main():
    config = uF.load_config('main')
    wandb_run = uF.get_run('ujjdbhrh')
    checkpoint_type = 'loss' # metric, loss
    
    wandb_run.config['checkpoint'] = f'{wandb_run.name}-{wandb_run.id}-{checkpoint_type}.ckpt'
    # wandb_run.config['checkpoint'] = f'{wandb_run.name}-{wandb_run.id}-loss.ckpt'
    config = uC.Config(config, wandb_run.config)
    model = InferenceModel.load_from_config(config, 'cuda:0')
    pathname = os.path.join(config.path.data.raw.test.unlabeled, '*.JPG')
    submission_name = f'{wandb_run.name}-{wandb_run.id}-{checkpoint_type}'
    submission_folder = os.path.join(config.path.submissions, submission_name)
    os.makedirs(submission_folder, exist_ok=True)

    for image_path in tqdm(glob(pathname)):
        image = Image.open(image_path)
        mask_pred = model(image)
        mask_pred_name = os.path.basename(image_path).replace('.JPG', '_pred.npy')
        mask_pred_path = os.path.join(submission_folder, mask_pred_name)
        np.save(mask_pred_path, mask_pred.numpy(force=True))


if __name__ == '__main__':
    main()
