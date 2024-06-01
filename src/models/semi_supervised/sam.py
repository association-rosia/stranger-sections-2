import torch
from transformers import SamModel, SamImageProcessor

from src.utils import func
from src.utils.cls import Config

import torch.nn.functional as F
import wandb


class SamForSemiSupervised:
    def __init__(self, config: Config, class_labels, loss_fct):
        self.config = config
        self.class_labels = class_labels
        self.loss_fct = loss_fct

        self.model, self.processor = self.load_model_processor()

        self.current_step = None
        self.current_batch_idx = None

    @torch.no_grad()
    def forward(self, inputs, consistency_logits):
        consistency_masks = func.logits_to_masks(consistency_logits)
        flatten_inputs, classes, indices = self.get_flatten_inputs(consistency_masks, inputs[0])
        flatten_outputs = self.sam_predict(flatten_inputs)
        sam_masks = self.post_process_flatten_outputs(flatten_inputs, flatten_outputs, classes, indices)
        loss = self.loss_fct(consistency_logits, sam_masks.long())

        return loss, consistency_masks, sam_masks

    def get_flatten_inputs(self, consistency_masks, inputs):
        input_masks, pixel_values, classes, indices = [], [], [], []

        for i in range(self.config.batch_size):
            input_masks, input_masks_i, classes, classes_i = self.create_input_masks(
                index=i,
                consistency_masks=consistency_masks,
                input_masks=input_masks,
                classes=classes
            )

            pixel_values, indices = self.create_pixel_values(
                index=i,
                input_masks_i=input_masks_i,
                inputs=inputs,
                pixel_values=pixel_values,
                indices=indices
            )

            if self.current_step == 'validation' and self.current_batch_idx == 0 and i == 0:
                self.log_input_masks(inputs, input_masks_i, classes_i)

        flatten_inputs = self.create_flatten_inputs(consistency_masks, input_masks, pixel_values)

        return flatten_inputs, classes, indices

    def create_input_masks(self, index, consistency_masks, input_masks, classes):
        consistency_masks_i = consistency_masks[index]
        classes_i = torch.unique(consistency_masks_i).tolist()
        consistency_masks_i = func.reshape_tensor(consistency_masks_i, size=self.input_masks_sizes)  # TODO
        input_masks_i = F.one_hot(consistency_masks_i.long(), num_classes=self.config.num_labels)
        input_masks_i = input_masks_i.to(dtype=torch.float16)
        input_masks_i = torch.permute(input_masks_i, (2, 0, 1))
        input_masks_i = input_masks_i[classes_i]

        if len(classes_i) > 1 and 0 in classes_i:
            input_masks_i = input_masks_i[1:]
            classes_i = classes_i[1:]
        elif classes_i == [0]:
            input_masks_i = torch.zeros(
                size=(1, self.input_masks_sizes[0], self.input_masks_sizes[1]),
                device=consistency_masks_i.device,
                dtype=consistency_masks_i.dtype
            )
            classes_i = [-1]

        input_masks.append(input_masks_i)
        classes += classes_i

        return input_masks, input_masks_i, classes, classes_i

    def create_pixel_values(self, index, input_masks_i, inputs, pixel_values, indices):
        num_masks = len(input_masks_i) if input_masks_i is not None else 1
        pixel_values_i = torch.stack([inputs['pixel_values'][index] for _ in range(num_masks)])
        pixel_values_i = func.reshape_tensor(pixel_values_i, size=())  # TODO
        pixel_values.append(pixel_values_i)
        indices += [index for _ in range(num_masks)]

        return pixel_values, indices

    def create_flatten_inputs(self, consistency_masks, input_masks, pixel_values):
        input_masks = torch.cat(input_masks).unsqueeze(dim=1)
        pixel_values = torch.cat(pixel_values)
        flatten_inputs = self.processor(images=pixel_values, return_tensors='pt')
        flatten_inputs['input_masks'] = input_masks
        flatten_inputs['multimask_output'] = False
        flatten_inputs = {k: v.to(consistency_masks.device) if isinstance(v, torch.Tensor) else v
                          for k, v in flatten_inputs.items()}

        return flatten_inputs

    def sam_predict(self, flatten_inputs):
        pred_masks = []
        flatten_inputs_size = len(flatten_inputs['pixel_values'])

        for start_idx in range(0, flatten_inputs_size, self.config.sam_batch_size):
            end_idx = min(start_idx + self.config.sam_batch_size, flatten_inputs_size)
            sam_batch = self.extract_sam_batch(flatten_inputs, start_idx, end_idx)
            flatten_outputs = self.model(**sam_batch)
            pred_masks.append(flatten_outputs.pred_masks)

        pred_masks = torch.cat(pred_masks)

        return pred_masks

    @staticmethod
    def extract_sam_batch(flatten_inputs, start_idx, end_idx):
        sam_batch = {}

        for key, value in flatten_inputs.items():
            if isinstance(value, torch.Tensor):
                sam_batch[key] = value[start_idx:end_idx]
            else:
                sam_batch[key] = value

        return sam_batch

    def post_process_flatten_outputs(self, flatten_inputs, pred_masks, classes, batch_idx):
        sam_masks = []

        masks = self.processor.post_process_masks(
            masks=pred_masks,
            original_sizes=flatten_inputs['original_sizes'],
            reshaped_input_sizes=flatten_inputs['reshaped_input_sizes'],
            binarize=False
        )
        masks = F.softmax(torch.cat(masks).squeeze(dim=1))

        for idx in range(self.config.sam_batch_size):
            sam_mask = []
            mask_batch_idx = masks[torch.Tensor(batch_idx) == idx]
            mask_class_idx = [classes[i] for i in range(len(batch_idx)) if batch_idx[i] == idx]

            class_idx_replaced = 0
            for label in range(self.config.num_labels):
                if label in mask_class_idx:
                    mask = (mask_batch_idx[class_idx_replaced] > self.config.sam_threshold).to(dtype=torch.float16)
                    sam_mask.append(mask)
                    class_idx_replaced += 1
                elif label == 0:
                    sam_mask.append(1e-8 * torch.ones(masks.shape[-2:], device=masks.device, dtype=torch.float16))
                else:
                    sam_mask.append(torch.zeros(masks.shape[-2:], device=masks.device, dtype=torch.float16))

            sam_mask = torch.stack(sam_mask)

            if self.current_step == 'validation' and self.current_batch_idx == 0 and idx == 0:
                self.log_output_masks(flatten_inputs, sam_mask)

            sam_mask = sam_mask.argmax(dim=0)
            sam_masks.append(sam_mask)

        sam_masks = torch.stack(sam_masks)
        sam_masks = func.reshape_tensor(sam_masks, size=self.input_image_sizes)  # TODO

        return sam_masks

    def log_input_masks(self, inputs, input_masks_i, classes_i):
        inputs = self.reshape_tensor(inputs['pixel_values'][0], self.input_image_sizes)
        inputs = torch.moveaxis(inputs, 0, -1).numpy(force=True)
        input_masks_i = input_masks_i.numpy(force=True)

        masks = {}
        class_idx_logged = 0
        for class_label in range(len(self.class_labels)):
            if class_label in classes_i:
                masks[f'input_masks_{class_label}'] = {'mask_data': input_masks_i[class_idx_logged]}
                class_idx_logged += 1
            else:
                masks[f'input_masks_{class_label}'] = {'mask_data': np.zeros(shape=self.input_image_sizes)}

        masks = dict(sorted(masks.items()))

        wandb.log({
            'val/sam_input_masks': wandb.Image(
                inputs,
                masks=masks
            )
        })

    @staticmethod
    def log_output_masks(flatten_inputs, output_masks):
        inputs = torch.moveaxis(flatten_inputs['pixel_values'][0], 0, -1).numpy(force=True)
        output_masks = output_masks.numpy(force=True)
        masks = {f'output_masks_{i}': {'mask_data': output_masks[i]} for i in range(len(output_masks))}
        masks = dict(sorted(masks.items()))

        wandb.log({
            'val/sam_output_masks': wandb.Image(
                inputs,
                masks=masks
            )
        })

    def load_model_processor(self):
        with torch.no_grad():
            model = SamModel.from_pretrained(
                self.config.sam_id
            )

        processor = SamImageProcessor.from_pretrained(
            self.config.sam_id,
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            do_convert_rgb=False
        )

        for param in model.parameters():
            param.requires_grad = False

        return model, processor
