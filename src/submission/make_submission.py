import os
from glob import glob
from PIL import Image
import numpy as np
from src import utils
from src.submission.model import InferenceModel

def main():
    config = utils.get_config()
    wandb_run = utils.get_run('cgtv3aow')
    model = InferenceModel.load_from_wandb_run(config, wandb_run, 'cuda:0')
    pathname = os.path.join(config['path']['data'], 'raw', 'test', 'unlabeled', '*.JPG')
    submission_folder = os.path.join(config['path']['submissions'], f'{wandb_run.name}-{wandb_run.id}')
    os.makedirs(submission_folder, exist_ok=True)

    for image_path in glob(pathname):
        image = Image.open(image_path)
        mask_pred = model(image)
        mask_pred_name = os.path.basename(image_path).replace('.JPG', '_pred.npy')
        mask_pred_path = os.path.join(submission_folder, mask_pred_name)
        np.save(mask_pred_path, mask_pred.numpy(force=True))

if __name__ == '__main__':
    main()