import numpy as np
from src.data import collate, processor
import torch
from PIL import Image
from typing_extensions import Self

from src.data.tiling import Tiler
from src.models.train_model import load_model
from src.utils.cls import Config


class SS2InferenceModel(torch.nn.Module):
    def __init__(
            self,
            config: Config,
            base_model: torch.nn.Module,
            map_location: str,
            tiling: bool = False,
            tile_size: int = None
    ) -> None:
        super().__init__()
        self.config = config
        self.map_location = map_location
        self.model = self._get_model(base_model)
        self.processor = self._get_processor()
        self.base_model_forward = self._get_base_model_forward()
        self.collate = self._get_collate()
        self.tiler = Tiler(self.config)
        self.tiling = tiling
        self.tile_size = self._get_tile_size(tile_size)

    @classmethod
    def load_from_config(
            cls,
            config: Config,
            map_location: str,
            tiling: bool = False,
            tile_size: int = None
    ) -> Self:
        model = load_model(config, map_location=map_location)
        if hasattr(model, 'model'):
            model = model.model
        else:
            model = model.student

        self = cls(config, model, map_location=map_location, tiling=tiling, tile_size=tile_size)

        return self

    def _get_model(self, base_model: torch.nn.Module):
        base_model = base_model.eval()
        base_model = base_model.to(device=self.map_location)

        return base_model

    def _get_processor(self) -> processor.SS2ImageProcessor:
        return processor.make_inference_processor(self.config)
        
    def _get_base_model_forward(self):
        if self.config.model_name == 'mask2former':
            return self._mask2former_forward
        if self.config.model_name == 'segformer':
            return self._segformer_forward
        else:
            raise ValueError(f"model_name expected 'mask2former' but received {self.config.model_name}")

    def _get_collate(self):
        return collate.get_collate_fn_inference(self.config)
        
    def _get_tile_size(self, tile_size):
        if self.tiling:
            if tile_size is None:
                tile_size = self.config.tile_size
            else:
                tile_size = tile_size
        else:
            tile_size = (self.config.data.size_h, self.config.data.size_w)

        return tile_size

    def _get_images(self, image: np.ndarray):
        if self.tiling:
            images = self.tiler.tile(image, self.tile_size)
        else:
            images = [image]

        return images

    @torch.inference_mode
    def forward(self, image: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, Image.Image):
            image = np.moveaxis(np.asarray(image), -1, 0)
        images = self._get_images(image)

        inputs = self.processor.preprocess(images)
        inputs.to(device=self.map_location)
        pred_masks = self.base_model_forward(inputs)
        pred_masks = [pred_mask.numpy(force=True) for pred_mask in pred_masks]

        if self.tiling:
            pred_mask = self.tiler.untile(pred_masks, self.tile_size)
        else:
            pred_mask = pred_masks[0]

        pred_mask = pred_mask.argmax(axis=0)

        return pred_mask

    def _mask2former_forward(self, inputs) -> torch.Tensor:
        # TODO: debug for mask2former
        outputs = self.model(inputs)
        pred_masks = self.processor.huggingface_processor.post_process_semantic_segmentation(outputs)

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
