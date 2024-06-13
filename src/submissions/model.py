import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import tv_tensors

from src.data import collate
from src.data.processor import SS2ImageProcessor, AugmentationMode, PreprocessingMode
from src.data.tiling import Tiler
from src.models.train_model import load_model
from src.submissions.tta import TestTimeAugmenter
from src.utils.cls import Config, ModelName


class SS2InferenceModel(torch.nn.Module):
    def __init__(
            self,
            config: Config,
            map_location: str,
            tile_size: int,
            tta_k: int | str,
    ) -> None:
        super().__init__()
        self.config = config
        self.map_location = map_location
        self.model = self._get_model()
        self.processor = self._get_processor()
        self.base_model_forward = self._get_base_model_forward()
        self.collate = self._get_collate()
        self.tiler = Tiler(self.config)
        self.tile_size = tile_size
        self.test_time_augmenter = TestTimeAugmenter(
            k=tta_k,
            random_state=self.config.random_state
        )

    def _get_model(self):
        model = load_model(self.config, map_location=self.map_location)
        if hasattr(model, 'model'):
            model = model.model
        else:
            model = model.student
        model = model.eval()
        model = model.to(device=self.map_location)

        return model

    def _get_processor(self) -> SS2ImageProcessor:
        ss2_processor = SS2ImageProcessor(
            self.config,
            AugmentationMode.NONE,
            PreprocessingMode.PHOTOMETRIC
        )
        
        return ss2_processor
        
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

            image = tv_tensors.Image(image)

        tiles = self.tiler.tile(image, self.tile_size)
        tiles_tta = [self.test_time_augmenter.augment(tile) for tile in tiles]

        tile_masks = []
        for tile_tta in tiles_tta:
            inputs = self.processor.preprocess(tile_tta)
            inputs.to(device=self.map_location)
            tta_mask = self.base_model_forward(inputs)
            tta_mask = [(
                    F.one_hot(mask, num_classes=self.config.num_labels)
                    .permute(2, 0, 1)
                )
                for mask in tta_mask
            ]
            tta_mask = [mask.cpu() for mask in tta_mask]
            tile_mask = self.test_time_augmenter.deaugment(tta_mask)
            tile_masks.append(tile_mask)
        
        mask = self.tiler.untile(tile_masks)
        mask = torch.argmax(mask, dim=0)

        return mask

    def _mask2former_forward(self, inputs) -> torch.Tensor:
        outputs = self.model(**inputs)
        pred_masks = self.processor.huggingface_processor.post_process_semantic_segmentation(
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
        resized_proba = torch.softmax(resized_logits, dim=1)
        pred_masks = torch.argmax(resized_proba, dim=1)

        return pred_masks
