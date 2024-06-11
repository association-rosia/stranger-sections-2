import os.path
from enum import Enum

import numpy as np
import torch
import torchvision.transforms.v2 as tv2T
import torchvision.transforms.v2.functional as tv2F
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


class PreprocessingMode(Enum):
    NONE = -1
    PHOTOMETRIC = 0


class SS2ImageProcessor:
    def __init__(self,
                 config: Config,
                 augmentation_mode: AugmentationMode = AugmentationMode.NONE,
                 preprocessing_mode: PreprocessingMode = PreprocessingMode.NONE
                 ) -> None:
        self.config = config
        self.augmentation_mode = augmentation_mode
        self.preprocessing_mode = preprocessing_mode
        self.huggingface_processor = self.get_huggingface_processor(config)
        self.augmentation_transforms = self.get_augmentation_transforms(augmentation_mode)
        self.preprocessing_transforms = self.get_preprocessing_transforms(preprocessing_mode)

    def preprocess(self,
                   images: np.ndarray | list[np.ndarray],
                   labels: np.ndarray | list[np.ndarray] = None,
                   augmentation_mode: AugmentationMode = None,
                   apply_huggingface: bool = True
                   ):

        if augmentation_mode is None:
            transforms = self.augmentation_transforms
        else:
            transforms = self.get_augmentation_transforms(augmentation_mode)

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

    def _preprocess_images_only(self, images, transforms):
        images_processed = []

        for image in images:
            image_processed = tv_tensors.Image(image)
            image_processed = self.preprocessing_transforms(image_processed)
            image_processed = transforms(image_processed)
            image_processed /= 255
            image_processed = image_processed.to(dtype=torch.float16)
            images_processed.append(image_processed)

        return images_processed

    def _preprocess_images_masks(self, images, masks, transforms):
        images_processed, masks_processed = [], []

        for image, mask in zip(images, masks):
            image = tv_tensors.Image(image)
            mask = tv_tensors.Mask(mask)
            image_preprocessed, mask_preprocessed = self.preprocessing_transforms(image, mask)
            image_processed, mask_processed = transforms(image_preprocessed, mask_preprocessed)
            # * To always keep multiple labels on a mask
            if len(torch.unique(mask_processed)) == 1:
                image_processed = image_preprocessed
                mask_processed = mask_preprocessed
            
            image_processed /= 255
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
                reduce_labels=False,
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

    def _get_constant_photometric_transforms(self):
        transforms = [
            tv2T.Lambda(lambda x: tv2F.adjust_contrast_image(x, self.config.contrast_factor), tv_tensors.Image),
            tv2T.Lambda(lambda x: tv2F.adjust_brightness_image(x, self.config.contrast_factor), tv_tensors.Image),
            tv2T.Lambda(lambda x: tv2F.adjust_gamma_image(x, self.config.gamma_factor), tv_tensors.Image),
            tv2T.Lambda(lambda x: tv2F.adjust_hue_image(x, self.config.hue_factor), tv_tensors.Image),
            tv2T.Lambda(lambda x: tv2F.adjust_sharpness_image(x, self.config.sharpness_factor), tv_tensors.Image),
            tv2T.Lambda(lambda x: tv2F.adjust_saturation_image(x, self.config.saturation_factor), tv_tensors.Image)
        ]

        return transforms

    @staticmethod
    def _get_geometric_transforms():
        transforms = [
            tv2T.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.05), scale=(0.95, 1.05), shear=(-8, 8)),
            tv2T.RandomHorizontalFlip(p=0.5),
            tv2T.RandomVerticalFlip(p=0.1),
            tv2T.RandomPerspective(distortion_scale=0.2, p=0.2)
        ]

        return transforms

    @staticmethod
    def _get_photometric_transforms():
        transforms = [
            tv2T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            tv2T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))
        ]

        return transforms

    def _get_both_transforms(self):
        geometric_transforms = self._get_geometric_transforms()
        photometric_transforms = self._get_photometric_transforms()

        return [*geometric_transforms, *photometric_transforms]
    
    def get_preprocessing_transforms(self,
                       preprocessing_mode: PreprocessingMode
                        ) -> tv2T.Compose:
        transforms = [tv2T.ToDtype(dtype=torch.float32, scale=False)]

        if preprocessing_mode == PreprocessingMode.NONE:
            pass
        elif preprocessing_mode == PreprocessingMode.PHOTOMETRIC:
            transforms.extend(self._get_constant_photometric_transforms())
        else:
            raise ValueError(f"Unknown preprocessing_mode: {preprocessing_mode}")
        
        return tv2T.Compose(transforms)

    def get_augmentation_transforms(self,
                       augmentation_mode: AugmentationMode,
                       ) -> tv2T.Compose:

        transforms = [tv2T.Identity()]

        if augmentation_mode == AugmentationMode.NONE:
            pass
        elif augmentation_mode == AugmentationMode.GEOMETRIC:
            transforms.extend(self._get_geometric_transforms())
        elif augmentation_mode == AugmentationMode.PHOTOMETRIC:
            transforms.extend(self._get_photometric_transforms())
        elif augmentation_mode == AugmentationMode.BOTH:
            transforms.extend(self._get_both_transforms())
        else:
            raise ValueError(f"Unknown augmentation_mode: {augmentation_mode}")

        return tv2T.Compose(transforms)


def _debug():
    config = func.load_config('main')
    wandb_config = func.load_config('mask2former', 'supervised')
    config = Config(config, wandb_config)

    train_preprocessor = SS2ImageProcessor(config, AugmentationMode.NONE)
    eval_preprocessor = SS2ImageProcessor(config, AugmentationMode.NONE)
    inf_preprocessor = SS2ImageProcessor(config, AugmentationMode.NONE)

    path_img = os.path.join(config.path.data.raw.train.labeled, '17gw5j.JPG')
    img = np.array(Image.open(path_img).convert('RGB'))

    path_mask = os.path.join(config.path.data.raw.train.labels, '17gw5j_gt.npy')
    mask = np.load(path_mask)

    t_output = train_preprocessor.preprocess(img, mask)
    e_output = eval_preprocessor.preprocess(img, mask)
    i_output = inf_preprocessor.preprocess(img)


if __name__ == '__main__':
    _debug()
