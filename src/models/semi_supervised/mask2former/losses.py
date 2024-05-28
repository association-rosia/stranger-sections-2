from torch import Tensor
import torch
from transformers.models.mask2former.configuration_mask2former import Mask2FormerConfig
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss
from transformers.models.mask2former.image_processing_mask2former import Mask2FormerImageProcessor

class SS2Mask2FormerLoss(Mask2FormerLoss):
    def __init__(self,
                 config: Mask2FormerConfig,
                 weight=None):
        self.weight_dict: dict[str, float] = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }
        super().__init__(config, self.weight_dict)

        # Replace the empty_weight from Mask2FormerLoss by our config weight
        self.eos_coef = config.no_object_weight
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        if weight is not None:
            empty_weight[:-1] = weight
        self.register_buffer("empty_weight", empty_weight)

    def forward(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
        auxiliary_predictions: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        loss_dict: dict[str, Tensor] = super().forward(
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            mask_labels=mask_labels,
            class_labels=class_labels,
            auxiliary_predictions=auxiliary_predictions,
        )

        # weight each loss by `self.weight_dict[<LOSS_NAME>]` including auxiliary losses
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight

        return sum(loss_dict.values())

    
     