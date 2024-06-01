from torch import Tensor
import torch
from transformers.models.mask2former.configuration_mask2former import Mask2FormerConfig
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss

class SS2Mask2FormerLoss(torch.nn.Module):
    def __init__(self,
                 config: Mask2FormerConfig,
                 weight=None):
        super(SS2Mask2FormerLoss, self).__init__()
        weight_dict: dict[str, float] = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }
        self.mask2former_loss = Mask2FormerLoss(config, weight_dict)
        self.weight_dict = weight_dict
        # Replace the empty_weight from Mask2FormerLoss by our config weight
        self.eos_coef = config.no_object_weight
        empty_weight = torch.ones(self.mask2former_loss.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        
        if weight is not None:
            empty_weight[:-1] = weight
        self.mask2former_loss.register_buffer("empty_weight", empty_weight)

    def forward(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
        auxiliary_predictions: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        loss_dict: dict[str, Tensor] = self.mask2former_loss.forward(
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            mask_labels=mask_labels,
            class_labels=class_labels,
            auxiliary_predictions=auxiliary_predictions
        )

        # weight each loss by `self.weight_dict[<LOSS_NAME>]` including auxiliary losses
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight

        return sum(loss_dict.values())

    
     