import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import Compose

from src.data import collate
from src.data.processor import SS2ImageProcessor, AugmentationMode, PreprocessingMode
from src.data.tiling import Tiler
from src.models.train_model import load_model
from src.submissions.tta import TestTimeAugmenter
from src.utils.cls import Config, ModelName
from transformers import Mask2FormerImageProcessor, SegformerImageProcessor


class SS2InferenceModel(torch.nn.Module):
    def __init__(
            self,
            config: Config,
            map_location: str,
            tile_size: int,
            test_time_augmenter: TestTimeAugmenter
    ) -> None:
        super().__init__()
        self.config = config
        self.map_location = map_location
        self.model = self._get_model()
        self.processor, self.transforms = self._get_processor_transforms()
        self.base_model_forward = self._get_base_model_forward()
        self.collate = self._get_collate()
        self.tiler = Tiler(self.config)
        self.tile_size = tile_size
        self.test_time_augmenter = test_time_augmenter

    def _get_model(self):
        model = load_model(self.config, map_location=self.map_location)
        if hasattr(model, 'model'):
            model = model.model
        else:
            model = model.student
        model = model.eval()
        model = model.to(device=self.map_location)

        return model

    def _get_processor_transforms(self) -> tuple[Mask2FormerImageProcessor | SegformerImageProcessor, Compose]:

        # TODO: remove in the future to keep photometric only
        if hasattr(self.config, 'brightness_factor'):
            preprocessing_mode = PreprocessingMode.PHOTOMETRIC
        else:
            preprocessing_mode = PreprocessingMode.NONE

        ss2_processor = SS2ImageProcessor(
            self.config,
            AugmentationMode.NONE,
            preprocessing_mode
        )
        
        return ss2_processor.huggingface_processor, ss2_processor.transforms
        
    def _get_base_model_forward(self):
        if self.config.model_name == ModelName.MASK2FORMER:
            return self._mask2former_forward
        if self.config.model_name == ModelName.SEGFORMER:
            return self._segformer_forward
        else:
            raise NotImplementedError

    def _get_collate(self):
        if self.config.model_name == ModelName.MASK2FORMER:
            return collate.SS2Mask2formerCollateFn(self.config, training=False)

    @torch.inference_mode
    def forward(self, image: Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image = np.moveaxis(np.asarray(image), -1, 0)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        image = self.transforms(image)
        tta_image = self.test_time_augmenter.augment(image)
        tta_tiled_image = [self.tiler.tile(image, self.tile_size) for image in tta_image]

        tta_mask = []
        for tiled_image in tta_tiled_image:
            inputs = self.processor.preprocess(tiled_image, return_tensors='pt')
            inputs.to(device=self.map_location)
            tiled_mask = self.base_model_forward(inputs)
            # pred_masks = [pred_mask.numpy(force=True) for pred_mask in pred_masks]
            tiled_mask = [mask.cpu() for mask in tiled_mask]
            mask = self.tiler.untile(tiled_mask)

            tta_mask.append(mask)

        pred_mask = self.test_time_augmenter.deaugment(tta_mask)

        return pred_mask

    def _mask2former_forward(self, inputs) -> torch.Tensor:
        # TODO: debug for mask2former
        outputs = self.model(**inputs)
        pred_masks = self.processor.post_process_semantic_segmentation(
            outputs, 
            target_sizes=[self.tile_size] * inputs['pixel_values'].shape[0]
        )

        return pred_masks

    def _segformer_forward(self, inputs) -> torch.Tensor:
        outputs = self.model(pixel_values=inputs['pixel_values'])
        resized_logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=self.tile_size,
            mode="bilinear",
            align_corners=False
        )
        resized_proba = resized_logits.softmax(dim=1)

        return resized_proba
