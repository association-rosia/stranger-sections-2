import os.path
from enum import Enum

import numpy as np
import torchvision.transforms.v2 as tv2T
from PIL import Image
from torchvision import tv_tensors
from transformers import Mask2FormerImageProcessor, SegformerImageProcessor

from src.utils import func
from src.utils.cls import Config


class AugmentationMode(Enum):
    NONE = -1
    SPATIAL = 0
    COLORIMETRIC = 1
    BOTH = 2


class SS2ImageProcessor:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.huggingface_processor = self.get_huggingface_processor(config)
        self.transforms = None

    def preprocess(self,
                   image: np.ndarray,
                   mask: np.ndarray = None,
                   augmentation_mode: AugmentationMode = 0,
                   apply_huggingface: bool = True
                   ):
        self.transforms = self._get_transforms(augmentation_mode)
        image = self._numpy_to_list(image)
        mask = self._numpy_to_list(mask)

        if image and mask:
            inputs, mask = self._preprocess_image_mask(image, mask)
        elif image and not mask:
            inputs = self._preprocess_image_only(image)
        else:
            raise NotImplementedError

        if apply_huggingface:
            inputs = self.huggingface_processor.preprocess(inputs, segmentation_maps=mask, return_tensors='pt')

        return inputs

    @staticmethod
    def _numpy_to_list(array):
        if not isinstance(array, list) and array is not None:
            array = [array]

        return array

    def _preprocess_image_only(self, images):
        return [self.transforms(tv_tensors.Image(image)) for image in images]

    def _preprocess_image_mask(self, images, masks):
        images_processed = []
        masks_processed = []

        for image, mask in zip(images, masks):
            image_processed, mask_processed = self.transforms(
                tv_tensors.Image(image),
                tv_tensors.Mask(mask)
            )

            images_processed.append(image_processed)
            masks_processed.append(mask_processed)

        return images_processed, masks_processed

    @staticmethod
    def get_huggingface_processor(config: Config):
        if config.model_name == 'mask2former':
            processor = Mask2FormerImageProcessor.from_pretrained(
                pretrained_model_name_or_path=config.model_id,
                do_rescale=False,
                do_normalize=True,
                reduce_labels=True,
                do_pad=False,
                do_resize=True,
                image_mean=config.data.mean,
                image_std=config.data.std,
                num_labels=config.num_labels
            )
        elif config.model_name == 'segformer':
            processor = SegformerImageProcessor.from_pretrained(
                pretrained_model_name_or_path=config.model_id,
                do_rescale=False,
                do_normalize=False,
                do_reduce_labels=False,
                do_pad=False,
                do_resize=False,
                # image_mean=config.data.mean,
                # image_std=config.data.std,
                num_labels=config.num_labels
            )
        else:
            raise ValueError(f"Unknown model_name: {config.model_name}")

        return processor

    @staticmethod
    def _get_none_transforms():
        transforms = tv2T.Compose([
            tv2T.Lambda(lambda x: x)
        ])

        return transforms

    @staticmethod
    def _get_spatial_transforms():
        transforms = tv2T.Compose([
            tv2T.Lambda(lambda x: x)
            # tv2T.Lambda(
            #     partial(tv2F.adjust_contrast, contrast_factor=self.config.contrast_factor']),
            #     tv_tensors.Image
            # ),
        ])

        return transforms

    @staticmethod
    def _get_colorimetric_transforms():
        transforms = tv2T.Compose([
            tv2T.Lambda(lambda x: x)
            # tv2T.Lambda(
            #     partial(tv2F.adjust_contrast, contrast_factor=self.config.contrast_factor']),
            #     tv_tensors.Image
            # ),
        ])

        return transforms

    def _get_both_transforms(self):
        spatial_transforms = self._get_spatial_transforms()
        colorimetric_transforms = self._get_colorimetric_transforms()

        transforms = tv2T.Compose([
            *spatial_transforms.transforms,
            *colorimetric_transforms.transforms
        ])

        return transforms

    def _get_transforms(self, augmentation_mode):
        if augmentation_mode == AugmentationMode.NONE:
            transforms = self._get_none_transforms()
        elif augmentation_mode == AugmentationMode.SPATIAL:
            transforms = self._get_spatial_transforms()
        elif augmentation_mode == AugmentationMode.COLORIMETRIC:
            transforms = self._get_colorimetric_transforms()
        elif augmentation_mode == AugmentationMode.BOTH:
            transforms = self._get_both_transforms()
        else:
            raise ValueError(f"Unknown augmentation_mode: {augmentation_mode}")

        return transforms


def _debug():
    config = func.load_config('main')
    wandb_config = func.load_config('segformer', 'semi_supervised')
    config = Config(config, wandb_config)

    processor = SS2ImageProcessor(config)

    path_img = os.path.join(config.path.data.raw.train.labeled, '17gw5j.JPG')
    img = np.array(Image.open(path_img).convert('RGB'))

    path_mask = os.path.join(config.path.data.raw.train.labels, '17gw5j_gt.npy')
    mask = np.load(path_mask)

    output = processor.preprocess(img, mask)


if __name__ == '__main__':
    _debug()
