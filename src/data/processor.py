import os.path
from enum import Enum
from typing import overload

import numpy as np
import torch
import torchvision.transforms.v2 as tv2T
from PIL import Image
from torchvision import tv_tensors
from transformers import Mask2FormerImageProcessor, SegformerImageProcessor

from utils import classes as uC
from utils import func as uF


class AugmentationMode(Enum):
    NONE = -1
    SPATIAL = 0
    COLORIMETRIC = 1
    BOTH = 2


class SS2SupervisedProcessor:
    def __init__(self, config: uC.Config) -> None:
        self.config = config
        self.hf_processor = self.get_huggingface_processor(config)

    def preprocess(self, image: np.ndarray, mask: np.ndarray = None, augmentation_mode: AugmentationMode = 0):
        self.transforms = self._get_transforms(augmentation_mode)

        image = self._numpy_to_list(image)
        mask = self._numpy_to_list(mask)

        if mask is not None:
            image, mask = self._preprocess_image_label(image, mask)
        else:
            image = self._preprocess_image(image)

        inputs = self.hf_processor.preprocess(image, segmentation_maps=mask, return_tensors='pt')

        return inputs

    @staticmethod
    def _numpy_to_list(image):
        if not isinstance(image, list) and image is not None:
            image = [image]

        return image

    def _preprocess_image(self, images):
        return [self.transforms(tv_tensors.Image(image)) for image in images]

    def _preprocess_image_label(self, images, masks):
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
    def get_huggingface_processor(config: uC.Config):
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
                num_labels=config.num_labels,
                ignore_index=255
            )
        elif config.model_name == 'segformer':
            processor = SegformerImageProcessor.from_pretrained(
                pretrained_model_name_or_path=config.model_id,
                do_rescale=False,
                do_normalize=True,
                reduce_labels=True,
                do_pad=False,
                do_resize=True,
                image_mean=config.data.mean,
                image_std=config.data.std,
                num_labels=config.num_labels,
                ignore_index=255
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


# def make_unsupervised_processor(config: uC.Config):
#     return SS2SupervisedProcessor(config, ProcessorMode.UNSUPERVISED)
#
#
# def make_supervised_processor(config: uC.Config):
#     return SS2SupervisedProcessor(config, ProcessorMode.SUPERVISED)
#
#
# def make_eval_processor(config: uC.Config):
#     return SS2SupervisedProcessor(config, ProcessorMode.EVAL)
#
#
# def make_inference_processor(config: uC.Config):
#     return SS2SupervisedProcessor(config, ProcessorMode.INFERENCE)


def _debug():
    config = uF.load_config('main')
    wandb_config = uF.load_config('mask2former', 'supervised')
    config = uC.Config(config, wandb_config)

    unsupervised_preprocessor = make_unsupervised_processor(config)
    supervised_preprocessor = make_supervised_processor(config)
    eval_preprocessor = make_eval_processor(config)
    inf_preprocessor = make_inference_processor(config)

    path_img = os.path.join(config.path.data.raw.train.labeled, '17gw5j.JPG')
    img = np.array(Image.open(path_img).convert('RGB'))

    path_mask = os.path.join(config.path.data.raw.train.labels, '17gw5j_gt.npy')
    mask = np.load(path_mask)

    u_output = unsupervised_preprocessor.preprocess(img, mask)
    s_output = supervised_preprocessor.preprocess(img, mask)
    e_output = eval_preprocessor.preprocess(img, mask)
    i_output = inf_preprocessor.preprocess(img)


if __name__ == '__main__':
    _debug()
