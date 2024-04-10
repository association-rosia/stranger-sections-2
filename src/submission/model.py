import torch
import wandb.apis.public as wandb_api
from PIL import Image
from typing_extensions import Self

import src.data.supervised.collate as spv_collate
import src.data.supervised.processor as spv_processor
from utils import classes as uC
from src.models.train_model import load_model


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
    def load_from_wandb_run(
            cls,
            config: uC.Config,
            wandb_run: wandb_api.Run | uC.RunDemo,
            map_location: str
    ) -> Self:

        config = uC.Config.merge(config, wandb_run.config)
        model = load_model(config, map_location=map_location)
        self = cls(config, model, map_location=map_location)

        return self

    def _get_processor(self) -> spv_processor.SS2SupervisedProcessor:
        if self.config.mode == 'supervised':
            return spv_processor.make_infering_processor(self.config)
        else:
            raise ValueError(f"mode expected 'supervised' but received {self.config.mode}")

    def _get_base_model_forward(self):
        if self.config.model_name == 'mask2former':
            return self._mask2former_forward
        else:
            raise ValueError(f"model_name expected 'mask2former' but received {self.config.model_name}")

    def _get_collate(self):
        if self.config.mode == 'supervised':
            return spv_collate.get_collate_fn_inference(self.config)
        else:
            raise ValueError(f"mode expected 'supervised' but received {self.config.mode}")

    @torch.inference_mode
    def forward(self, image: Image.Image) -> torch.Tensor:
        return self.base_model_forward(image)[0]

    def _mask2former_forward(self, images: Image.Image | list[Image.Image]) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = [images]

        inputs = self.processor.preprocess(images)
        # inputs = self.collate(inputs)
        inputs.to(device=self.map_location)
        outputs = self.model(inputs)
        mask_pred = self.processor.hf_processor.post_process_semantic_segmentation(outputs)

        return mask_pred
