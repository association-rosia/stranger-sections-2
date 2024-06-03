import os.path
from enum import Enum

import numpy as np
import torch
import torchvision.transforms.v2 as tv2T
from PIL import Image
from torchvision import tv_tensors
from transformers import Mask2FormerImageProcessor, SegformerImageProcessor

from src.utils import func
from src.utils.cls import Config


class AugmentationMode(Enum):
    NONE = -1
    GEOMETRIC = 0
    PHOTOMETRIC = 1
    BOTH = 2


class SS2ImageProcessor:
    def __init__(self, config: Config, augmentation_mode: AugmentationMode = AugmentationMode.NONE) -> None:
        self.config = config
        self.augmentation_mode = augmentation_mode
        self.huggingface_processor = self.get_huggingface_processor(config)
        self.transforms = self._get_transforms(augmentation_mode)

    def preprocess(self,
                   images: np.ndarray | list[np.ndarray],
                   labels: np.ndarray | list[np.ndarray] = None,
                   augmentation_mode: AugmentationMode = None,
                   apply_huggingface: bool = True
                   ):

        if augmentation_mode is None:
            transforms = self.transforms
        else:
            transforms = self._get_transforms(augmentation_mode)

        images = self._numpy_to_list(images)
        labels = self._numpy_to_list(labels)

        if labels is not None:
            inputs, labels = self._preprocess_images_masks(images, labels, transforms)
        else:
            inputs = self._preprocess_images_only(images, transforms)

        if apply_huggingface:
            inputs = self.huggingface_processor.preprocess(inputs, segmentation_maps=labels, return_tensors='pt')

        return inputs

    @staticmethod
    def _numpy_to_list(array):
        if not isinstance(array, list) and array is not None:
            array = [array]

        return array

    @staticmethod
    def _preprocess_images_only(images, transforms):
        images_processed = []

        for image in images:
            image_processed = transforms(tv_tensors.Image(image))
            image_processed = torch.clamp(image_processed, min=0, max=1)
            image_processed = image_processed.to(dtype=torch.float16)
            images_processed.append(image_processed)

        return images_processed

    @staticmethod
    def _preprocess_images_masks(images, masks, transforms):
        images_processed, masks_processed = [], []

        for image, mask in zip(images, masks):
            image_processed, mask_processed = transforms(
                tv_tensors.Image(image),
                tv_tensors.Mask(mask)
            )

            image_processed = torch.clamp(image_processed, min=0, max=1)
            image_processed = image_processed.to(dtype=torch.float16)
            images_processed.append(image_processed)

            mask_processed = mask_processed.to(dtype=torch.uint8)
            masks_processed.append(mask_processed)

        return images_processed, masks_processed

    @staticmethod
    def get_huggingface_processor(config: Config):
        if config.model_name == 'mask2former':
            processor = Mask2FormerImageProcessor.from_pretrained(
                pretrained_model_name_or_path=config.model_id,
                do_rescale=False,
                do_normalize=config.do_normalize,
                reduce_labels=True,
                do_pad=False,
                do_resize=True,
                image_mean=config.data.mean,
                image_std=config.data.std,
                num_labels=config.num_labels,
            )
        elif config.model_name == 'segformer':
            processor = SegformerImageProcessor.from_pretrained(
                pretrained_model_name_or_path=config.model_id,
                do_rescale=False,
                do_normalize=config.do_normalize,
                do_reduce_labels=False,
                do_pad=False,
                do_resize=True,
                image_mean=config.data.mean,
                image_std=config.data.std,
                num_labels=config.num_labels,
            )
        else:
            raise ValueError(f"Unknown model_name: {config.model_name}")

        return processor

    @staticmethod
    def _get_none_transforms():
        transforms = [
            tv2T.Lambda(lambda x: x)
        ]

        return transforms

    @staticmethod
    def _get_geometric_transforms():
        transforms = [
            tv2T.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.05), scale=(0.95, 1.05), shear=(-8, 8)),
            tv2T.RandomHorizontalFlip(p=0.5),
            tv2T.RandomVerticalFlip(p=0.1),
            tv2T.RandomPerspective(distortion_scale=0.2, p=0.2),
            # tv2T.ElasticTransform(alpha=1, sigma=0.1)
        ]

        return transforms

    @staticmethod
    def _get_photometric_transforms():
        transforms = [
            tv2T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            tv2T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
            # tv2T.RandomGrayscale(p=0.05),
            # tv2T.RandomInvert(p=0.05),
            # tv2T.RandomPosterize(bits=5, p=0.1),
            # tv2T.RandomSolarize(threshold=0.75, p=0.1),
            # tv2T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
            # tv2T.RandomAutocontrast(p=0.2),
            # tv2T.RandomEqualize(p=0.2)
        ]

        return transforms

    def _get_both_transforms(self):
        geometric_transforms = self._get_geometric_transforms()
        photometric_transforms = self._get_photometric_transforms()

        return [*geometric_transforms, *photometric_transforms]

    def _get_transforms(self, augmentation_mode: AugmentationMode) -> tv2T.Compose:
        transforms = [tv2T.ToDtype(dtype=torch.float32, scale=True)]

        if augmentation_mode == AugmentationMode.NONE:
            transforms.extend(self._get_none_transforms())
        elif augmentation_mode == AugmentationMode.GEOMETRIC:
            transforms.extend(self._get_geometric_transforms())
        elif augmentation_mode == AugmentationMode.PHOTOMETRIC:
            transforms.extend(self._get_photometric_transforms())
        elif augmentation_mode == AugmentationMode.BOTH:
            transforms.extend(self._get_both_transforms())
        else:
            raise ValueError(f"Unknown augmentation_mode: {augmentation_mode}")

        return tv2T.Compose(transforms)


def make_training_processor(config: Config):
    return SS2ImageProcessor(config, AugmentationMode.NONE)


def make_eval_processor(config: Config):
    return SS2ImageProcessor(config, AugmentationMode.NONE)


def make_inference_processor(config: Config):
    return SS2ImageProcessor(config, AugmentationMode.NONE)


def _debug():
    config = func.load_config('main')
    wandb_config = func.load_config('mask2former', 'supervised')
    config = Config(config, wandb_config)

    train_preprocessor = make_training_processor(config)
    eval_preprocessor = make_eval_processor(config)
    inf_preprocessor = make_inference_processor(config)

    path_img = os.path.join(config.path.data.raw.train.labeled, '17gw5j.JPG')
    img = np.array(Image.open(path_img).convert('RGB'))

    path_mask = os.path.join(config.path.data.raw.train.labels, '17gw5j_gt.npy')
    mask = np.load(path_mask)

    t_output = train_preprocessor.preprocess(img, mask)
    e_output = eval_preprocessor.preprocess(img, mask)
    i_output = inf_preprocessor.preprocess(img)


if __name__ == '__main__':
    _debug()
