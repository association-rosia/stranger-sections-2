from typing import overload
from functools import partial

import numpy as np
import torch
import torchvision.transforms.v2.functional as tv2F
import torchvision.transforms.v2 as tv2T
from  torchvision import tv_tensors
from PIL import Image
from enum import Enum


from transformers import Mask2FormerImageProcessor

class ProcessorMode(Enum):
    """Processor modes
    Available modes are ``training`` and ``eval``.
    """
    TRAINING = 0
    EVAL = 1
    INFERING = 2


class SupervisedProcessor:
    def __init__(self, config, processor_mode: ProcessorMode) -> None:
        self.config = config
        self.processor_mode = processor_mode
        self.hf_processor = self.get_hugingface_processor(config)
        self.transforms = self._get_transforms()

    @overload
    def preprocess(self, images: Image.Image, masks: np.ndarray=None) -> torch.Tensor:
        ...

    @overload
    def preprocess(self, images: list[Image.Image], masks: list[np.ndarray]=None) -> torch.Tensor:
        ...

    def preprocess(self, images: Image.Image | list[Image.Image], masks: np.ndarray | list[np.ndarray]=None) -> torch.Tensor:
        if not isinstance(images, list):
            images = [images]
        
        if self.processor_mode in [ProcessorMode.TRAINING, ProcessorMode.EVAL]:
            if not isinstance(masks, list):
                masks = [masks]
            images, masks = self._preprocess_image_label(images, masks)
        elif self.processor_mode == ProcessorMode.INFERING:
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
    def get_hugingface_processor(config: dict):
        if config['model_name'] == 'mask2former':
            processor = Mask2FormerImageProcessor.from_pretrained(
                pretrained_model_name_or_path=config['model_id'],
                do_rescale=False,
                do_normalize=True,
                reduce_labels=True,
                do_pad=False,
                do_resize=True,
                image_mean=config['data']['mean'],
                image_std=config['data']['std'],
                num_labels=config['num_labels'],
            )

        return processor

    def _get_training_transforms(self):
        transforms = tv2T.Compose([
            tv2T.ToDtype(dtype=torch.float32, scale=True),
            # tv2T.Lambda(
            #     partial(tv2F.adjust_contrast, contrast_factor=self.config['contrast_factor']),
            #     tv_tensors.Image
            # ),
            # tv2T.Resize(
            #     self.config['size'],
            #     interpolation=tv2F.InterpolationMode.BICUBIC
            # ),
            # tv2T.Normalize(mean=self.config['data']['mean'], std=self.config['data']['std'])
        ])

        return transforms

    def _get_eval_transforms(self):
        transforms = tv2T.Compose([
            tv2T.ToDtype(dtype=torch.float32),
            # tv2T.Lambda(
            #     partial(tv2F.adjust_contrast, contrast_factor=self.config['contrast_factor']),
            #     tv_tensors.Image
            # ),
            # tv2T.Resize(
            #     self.config['size'],
            #     interpolation=tv2F.InterpolationMode.BICUBIC
            # ),
            # tv2T.Normalize(mean=self.config['data']['mean'], std=self.config['data']['std'])
        ])

        return transforms

    def _get_transforms(self):
        if self.processor_mode == ProcessorMode.TRAINING:
            return self._get_training_transforms()
        elif self.processor_mode in [ProcessorMode.EVAL, ProcessorMode.INFERING]:
            return self._get_eval_transforms()


def make_training_processor(config):
    return SupervisedProcessor(config, ProcessorMode.TRAINING)


def make_eval_processor(config):
    return SupervisedProcessor(config, ProcessorMode.EVAL)


def make_infering_processor(config):
    return SupervisedProcessor(config, ProcessorMode.INFERING)

def _debug():
    from src import utils

    config = utils.get_config()
    wandb_config = utils.load_config('mask2former.yml', 'supervised')
    config.update(wandb_config)
    train_preprocessor = make_training_processor(config)
    eval_preprocessor = make_eval_processor(config)
    inf_preprocessor = make_infering_processor(config)
    img = Image.open('data/raw/train/labeled/17gw5j.JPG').convert('RGB')
    mask = np.load('data/raw/train/labels/17gw5j_gt.npy')
    t_output = train_preprocessor.preprocess(img, mask)
    e_output = eval_preprocessor.preprocess(img, mask)
    i_output = inf_preprocessor.preprocess(img)

    return

if __name__ == '__main__':
    _debug()