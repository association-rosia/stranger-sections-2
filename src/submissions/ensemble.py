import os
import numpy as np
from src.utils.cls import Config
from tqdm.autonotebook import tqdm
from glob import glob
from src.utils import func

def main():
    config = func.load_config('main', loading='object')
    submissions = [
        'submissions/fluent-water-971-mlmyc2ql/fluent-water-971-mlmyc2ql-ckpt-micro-tiling-384-tta-max',
        'submissions/atomic-sweep-62-iecskwiv/atomic-sweep-62-iecskwiv-ckpt-spv-v1-tiling-384-tta-max',
        'submissions/avid-sweep-31-kyxcee1p/avid-sweep-31-kyxcee1p-ckpt-spv-v1-tiling-384-tta-max'
    ]

    weights = [
        0.5471,
        0.5238,
        0.5032,
    ]
    # weights = None

    
    ensembler = Ensembler(config, submissions, weights)
    ensembler.build()

class Ensembler:
    def __init__(self, config: Config, submission_folders: list[str], weights: list[float] = None):
        self.config = config
        self.submission_folders = submission_folders
        if weights is None:
            weights = [1.] * len(self.submission_folders)
        self.weights = weights

    def __call__(self, pred_masks: list[np.ndarray], weights: list[float] = None) -> np.ndarray:
        if weights is None:
            weights = [1.] * len(pred_masks)

        processed_masks = [self.process_mask(mask, weight) for mask, weight in zip(pred_masks, weights)]
        ensemble_mask = self.merge_masks(processed_masks)

        return ensemble_mask

    def build(self):
        pathname = os.path.join(self.config.path.data.raw.test.unlabeled, '*.JPG')
        submission_name = '-'.join([
            os.path.basename(submission_folder).split('-')[3]
            for submission_folder in self.submission_folders
        ])
        submission_folder = os.path.join(self.config.path.submissions, submission_name)
        os.makedirs(submission_folder, exist_ok=True)

        for image_path in tqdm(glob(pathname)):
            mask_pred_name = os.path.basename(image_path).replace('.JPG', '_pred.npy')
            submission_path = os.path.join(submission_folder, mask_pred_name)
            pred_masks = []
            for prediction_folder, weight in zip(self.submission_folders, self.weights):
                mask_path = os.path.join(prediction_folder, mask_pred_name)
                mask = np.load(mask_path)
                processed_mask = self.process_mask(mask, weight)
                pred_masks.append(processed_mask)
            
            ensemble_mask = self.merge_masks(pred_masks)
            np.save(submission_path, ensemble_mask)


    def process_mask(self, mask: np.ndarray, weight: float) -> np.ndarray:
        classes_mask = np.eye(self.config.data.num_labels)[mask]
        prob_mask = classes_mask * weight

        return prob_mask

    def merge_masks(self, pred_masks: list[np.ndarray]) -> np.ndarray:
        merged_mask = np.zeros_like(pred_masks[0])
        for mask in pred_masks:
            merged_mask += mask

        merged_mask = np.argmax(merged_mask, axis=-1)

        return merged_mask


if __name__ == '__main__':
    main()