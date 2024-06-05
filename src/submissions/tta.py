import itertools
import math

import torch
import torchvision.transforms.v2.functional as tv2F
from src.data.processor import AugmentationMode, SS2ImageProcessor
from torchvision.transforms.v2 import Compose
import torchvision.transforms.v2 as tv2T

import numpy as np
from collections import deque


class GeometricAugmentation:
    def __init__(self, params):
        self.params = params
        super().__init__()

    def augment(self, image: torch.Tensor, *args, **params):
        raise NotImplementedError

    def deaugment(self, mask: torch.Tensor, *args, **params):
        raise NotImplementedError


class TestTimeAugmenter:
    def __init__(self, augmentation_mode: AugmentationMode, k: int | str, return_probs: bool, random_state: int = None):
        self.photometric_transforms = self._get_photometric_transforms(augmentation_mode)
        self.geometric_transform = self._get_geometric_transforms()
        self.params = [t.params for t in self.geometric_transform]
        self.product = list(itertools.product(*self.params))
        self.delist = self.geometric_transform[::-1]
        self.k = k
        self.return_probs = return_probs
        self.random_state = random_state
        self.numpy_random = np.random.RandomState(seed=random_state)
        self.queue = deque()

    def _get_photometric_transforms(self, augmentation_mode: AugmentationMode) -> Compose:
        transforms = [tv2T.Lambda(lambda x: x)]
        
        if augmentation_mode in [AugmentationMode.PHOTOMETRIC, AugmentationMode.BOTH]:
            transforms = SS2ImageProcessor._get_photometric_transforms()
        
        return Compose(transforms)
    
    def _get_geometric_transforms(augmentation_mode: AugmentationMode) -> list[GeometricAugmentation]:
        return [HorizontalFlip(), VerticalFlip(), Rotate()]

    def _select_parameters(self) -> list:
        if isinstance(self.k, int):
            random_parameters = self.numpy_random.choice(
                np.arange(1, len(self.product)),
                size=self.k-1,
                replace=False
            )
            parameters = [self.product[index_parameter] for index_parameter in random_parameters]
        elif self.k == 'max':
            parameters = self.product
        else:
            raise ValueError(f'{self.k}')

        return parameters
        
    def _merge_augmentations(self, masks: list[torch.Tensor]) -> torch.Tensor:
        tta_mask = torch.zeros(size=masks[0].shape, device=masks[0].device)
        for mask in masks:
            tta_mask += mask

        if not self.return_probs:
            tta_mask = torch.argmax(tta_mask, dim=0)

        return tta_mask

    def augment(self, image: torch.Tensor):
        
        tta_parameters = self._select_parameters()
        self.queue.append(tta_parameters)
        augmented_images = [image]

        for parameters in tta_parameters:
            augmented_image = torch.clone(image)
            augmented_image = self.photometric_transforms(augmented_image)
            for augmentation, parameter in zip(self.geometric_transform, parameters):
                augmented_image = augmentation.augment(augmented_image, parameter)
            augmented_images.append(augmented_image)

        return augmented_images
    
    def deaugment(self, masks: list):
        tta_parameters = self.queue.popleft()
        deaugmented_masks = [masks.pop(0)]
        for deaugmented_mask, parameters in zip(masks, tta_parameters):
            for augmentation, parameter in zip(self.geometric_transform[::-1], parameters[::-1]):
                deaugmented_mask = augmentation.deaugment(deaugmented_mask, parameter)
            deaugmented_masks.append(deaugmented_mask)

        tta_mask = self._merge_augmentations(deaugmented_masks)

        return tta_mask


class HorizontalFlip(GeometricAugmentation):
    def __init__(self):
        super().__init__([False, True])  # /!\ always start with the identity value

    def augment(self, image: torch.Tensor, apply=False, **kwargs):
        if apply:
            image = tv2F.hflip(image)

        return image

    def deaugment(self, mask: torch.Tensor, apply=False, **kwargs):
        if apply:
            mask = tv2F.hflip(mask)

        return mask


class VerticalFlip(GeometricAugmentation):
    def __init__(self):
        super().__init__([False, True])  # /!\ always start with the identity value

    def augment(self, image: torch.Tensor, apply=False, **kwargs):
        if apply:
            image = tv2F.vflip(image)

        return image

    def deaugment(self, mask: torch.Tensor, apply=False, **kwargs):
        if apply:
            mask = tv2F.vflip(mask)

        return mask


class Rotate(GeometricAugmentation):
    def __init__(self, angles: list = [0, 90, 180, 270]):
        allowed_angles = [0, 90, 180, 270]
        angles = list(set([0] + angles))  # /!\ always start with the identity value

        for angle in angles:
            if angle not in allowed_angles:
                raise ValueError(f'angles must be equal to 0, 90, 180 or 270')

        super().__init__(angles)

    def augment(self, image: torch.Tensor, angle=0, **kwargs):
        return tv2F.rotate(image, angle=angle)

    def deaugment(self, mask: torch.Tensor, angle=0, **kwargs):
        return tv2F.rotate(mask, angle=-angle)
