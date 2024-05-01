import torch
import torch.nn.functional as tF
import wandb.apis.public as wandb_api
from PIL import Image
from typing_extensions import Self

import src.data.supervised.collate as spv_collate
import src.data.supervised.processor as spv_processor
from src.models.train_model import load_model
from utils import classes as uC


class InferenceModel(torch.nn.Module):
    def __init__(
            self,
            config: uC.Config,
            base_model: torch.nn.Module,
            map_location: str
    ) -> None:
        super().__init__()
        self.config = config
        self.model = base_model
        self.model.eval()
        self.model.to(device=map_location)
        self.map_location = map_location
        self.processor = self._get_processor()
        self.base_model_forward = self._get_base_model_forward()
        self.collate = self._get_collate()

    @classmethod
    def load_from_config(
            cls,
            config: uC.Config,
            map_location: str
    ) -> Self:
        model = load_model(config, map_location=map_location)
        self = cls(config, model.model, map_location=map_location)

        return self

    def _get_processor(self) -> spv_processor.SS2SupervisedProcessor:
        if self.config.mode == 'supervised':
            return spv_processor.make_inference_processor(self.config)
        else:
            raise ValueError(f"mode expected 'supervised' but received {self.config.mode}")

    def _get_base_model_forward(self):
        if self.config.model_name == 'mask2former':
            return self._mask2former_forward
        if self.config.model_name == 'segformer':
            return self._segformer_forward
        else:
            raise ValueError(f"model_name expected 'mask2former' but received {self.config.model_name}")

    def _get_collate(self):
        if self.config.mode == 'supervised':
            return spv_collate.get_collate_fn_inference(self.config)
        else:
            raise ValueError(f"mode expected 'supervised' but received {self.config.mode}")

    @torch.inference_mode
    def forward(self, image: Image.Image) -> torch.Tensor:
        # if isinstance(images, Image.Image):
        images = [image]
        inputs = self.processor.preprocess(images)
        inputs.to(device=self.map_location)
        pred_masks = self.base_model_forward(inputs)

        return pred_masks

    def _mask2former_forward(self, inputs) -> torch.Tensor:
        outputs = self.model(inputs)
        pred_masks = self.processor.hf_processor.post_process_semantic_segmentation(outputs)

        return pred_masks.squeeze()
    
    def _segformer_forward(self, inputs) -> torch.Tensor:
        outputs = self.model(pixel_values=inputs['pixel_values'])
        # pred_masks = tF.sigmoid(logits).argmax(dim=1).type(torch.uint8)
        pred_masks = self.processor.hf_processor.post_process_semantic_segmentation(
            outputs,
            [(self.config.data.size_h, self.config.data.size_w)]
        )

        return pred_masks[0]
