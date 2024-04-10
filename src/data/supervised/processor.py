from enum import Enum
from typing import overload

import numpy as np
import torch
import torchvision.transforms.v2 as tv2T
from PIL import Image
from torchvision import tv_tensors
from transformers import Mask2FormerImageProcessor

from utils import classes as uC


class ProcessorMode(Enum):
    TRAINING = 0
    EVAL = 1
    INFERENCE = 2


class SS2SupervisedProcessor:
    def __init__(self, config: uC.Config, processor_mode: ProcessorMode) -> None:
        self.config = config
        self.processor_mode = processor_mode
        self.hf_processor = self.get_huggingface_processor(config)
        self.transforms = self._get_transforms()

    @overload
    def preprocess(self, images: Image.Image, masks: np.ndarray = None) -> torch.Tensor:
        ...

    @overload
    def preprocess(self, images: list[Image.Image], masks: list[np.ndarray] = None) -> torch.Tensor:
        ...

    def preprocess(self, images: Image.Image | list[Image.Image],
                   masks: np.ndarray | list[np.ndarray] = None) -> torch.Tensor:
        if not isinstance(images, list):
            images = [images]

        if self.processor_mode in [ProcessorMode.TRAINING, ProcessorMode.EVAL]:
            if not isinstance(masks, list):
                masks = [masks]
            images, masks = self._preprocess_image_label(images, masks)
        elif self.processor_mode == ProcessorMode.INFERENCE:
            images = self._preprocess_image(images)
            masks = None

        return self.hf_processor.preprocess(images, segmentation_maps=masks, return_tensors='pt')

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
            )

        return processor

    @staticmethod
    def _get_training_transforms():
        transforms = tv2T.Compose([
            tv2T.ToDtype(dtype=torch.float32, scale=True),
            # tv2T.Lambda(
            #     partial(tv2F.adjust_contrast, contrast_factor=self.config.contrast_factor']),
            #     tv_tensors.Image
            # ),
            # tv2T.Resize(
            #     self.config.size'],
            #     interpolation=tv2F.InterpolationMode.BICUBIC
            # ),
            # tv2T.Normalize(mean=self.config.data']['mean'], std=self.config.data']['std'])
        ])

        return transforms

    @staticmethod
    def _get_eval_transforms():
        transforms = tv2T.Compose([
            tv2T.ToDtype(dtype=torch.float32),
            # tv2T.Lambda(
            #     partial(tv2F.adjust_contrast, contrast_factor=self.config.contrast_factor']),
            #     tv_tensors.Image
            # ),
            # tv2T.Resize(
            #     self.config.size'],
            #     interpolation=tv2F.InterpolationMode.BICUBIC
            # ),
            # tv2T.Normalize(mean=self.config.data']['mean'], std=self.config.data']['std'])
        ])

        return transforms

    def _get_transforms(self):
        if self.processor_mode == ProcessorMode.TRAINING:
            return self._get_training_transforms()
        elif self.processor_mode in [ProcessorMode.EVAL, ProcessorMode.INFERENCE]:
            return self._get_eval_transforms()


def make_training_processor(config: uC.Config):
    return SS2SupervisedProcessor(config, ProcessorMode.TRAINING)


def make_eval_processor(config: uC.Config):
    return SS2SupervisedProcessor(config, ProcessorMode.EVAL)


def make_infering_processor(config: uC.Config):
    return SS2SupervisedProcessor(config, ProcessorMode.INFERENCE)


def _debug():
    from src import utils
    config = utils.load_config('main')
    wandb_config = utils.load_config('mask2former', 'supervised')
    config.update(wandb_config)
    train_preprocessor = make_training_processor(config)
    eval_preprocessor = make_eval_processor(config)
    inf_preprocessor = make_infering_processor(config)
    img = Image.open('data/raw/train/labeled/17gw5j.JPG').convert('RGB')
    mask = np.load('data/raw/train/labels/17gw5j_gt.npy')
    t_output = train_preprocessor.preprocess(img, mask)
    e_output = eval_preprocessor.preprocess(img, mask)
    i_output = inf_preprocessor.preprocess(img)


if __name__ == '__main__':
    _debug()
