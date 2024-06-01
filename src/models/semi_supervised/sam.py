import torch
from transformers import SamModel, SamImageProcessor

from src.utils import func
from src.utils.cls import Config

import torch.nn.functional as F
import wandb

import random
import numpy as np


class SamForSemiSupervised:
    def __init__(self, config: Config, class_labels, loss_fct):
        self.config = config
        self.class_labels = class_labels
        self.loss_fct = loss_fct

        self.model, self.processor = self.load_model_processor()
        self.pixel_values_sizes = (1024, 1024)

        self.current_step = None
        self.current_batch_idx = None

    @torch.no_grad()
    def forward(self, inputs, logits):
        masks = func.logits_to_masks(logits)
        inputs, classes, indices = self.get_inputs(masks, inputs)
        outputs = self.batch_predict(inputs)
        pred_masks = self.post_process_outputs(inputs, outputs, classes, indices)
        loss = self.loss_fct(logits, pred_masks.long())

        return loss, masks, pred_masks

    def get_inputs(self, consistency_masks, inputs):
        pixel_values, input_points, input_labels, classes, indices = [], [], [], [], []

        for i in range(self.config.batch_size):
            input_masks_i, input_points_i, input_labels_i, classes_i = self.get_input_points_labels(
                index=i,
                consistency_masks=consistency_masks
            )
            input_points.append(input_points_i)
            input_labels.append(input_labels_i)
            classes += classes_i

            pixel_values_i, indices_i = self.get_pixel_values(
                index=i,
                classes=classes_i,
                inputs=inputs
            )
            pixel_values.append(pixel_values_i)
            indices += indices_i

            if self.current_step == 'validation' and self.current_batch_idx == 0 and i == 0:
                self.log_input_masks(inputs, input_masks_i, classes_i)

        inputs = self.build_inputs(pixel_values, input_points, input_labels)

        return inputs, classes, indices

    def get_pixel_values(self, index, classes, inputs):
        num_masks = len(classes)
        pixel_values = torch.stack([inputs['pixel_values'][index] for _ in range(num_masks)])
        pixel_values = func.reshape_tensor(pixel_values, size=self.pixel_values_sizes)
        indices = [index for _ in range(num_masks)]

        return pixel_values, indices

    def build_inputs(self, pixel_values, input_points, input_labels):
        inputs = self.processor(images=torch.cat(pixel_values), return_tensors='pt')
        inputs['input_points'] = torch.cat(input_points)
        inputs['input_labels'] = torch.cat(input_labels)
        inputs['multimask_output'] = False

        inputs = {
            k: v.to(device=pixel_values[0].device)
            if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        return inputs

    def get_input_points_labels(self, index, consistency_masks):
        consistency_masks = consistency_masks[index]
        classes = torch.unique(consistency_masks).tolist()
        consistency_masks = func.reshape_tensor(consistency_masks, size=self.pixel_values_sizes)
        input_masks = self.get_input_masks(consistency_masks, classes)
        input_masks, classes = self.update_input_masks_classes(input_masks, classes)
        input_points, input_labels = self.build_input_points_labels(input_masks)

        return input_masks, input_points, input_labels, classes

    def build_input_points_labels(self, input_masks):
        input_points, input_labels = [], []

        for i in range(len(input_masks)):
            rate_of_ones = self.calculate_rate_of_ones(input_masks[i])
            num_points_1 = int(rate_of_ones * self.config.sam_num_input_points)
            num_points_0 = self.config.sam_num_input_points - num_points_1

            input_points_1, input_labels_1 = self.get_random_points(
                input_masks[i],
                num_points_1,
                labels=1
            )

            input_points_0, input_labels_0 = self.get_random_points(
                input_masks[i],
                num_points_0,
                labels=0
            )

            input_points.append(torch.cat([input_points_1, input_points_0]))
            input_labels.append(torch.cat([input_labels_1, input_labels_0]))

        input_points = torch.stack(input_points).unsqueeze(dim=1)
        input_labels = torch.stack(input_labels).unsqueeze(dim=1)

        return input_points, input_labels

    @staticmethod
    def calculate_rate_of_ones(input_masks):
        total_elements = input_masks.numel()
        ones_count = (input_masks == 1).sum().item()
        rate = ones_count / total_elements

        return rate

    @staticmethod
    def get_random_points(input_masks, num_points, labels: int):
        if num_points > 0:
            all_valid_points = torch.nonzero(input_masks == labels)
            random_indices = random.sample(range(len(all_valid_points)), num_points)
            random_points = torch.stack([all_valid_points[i] for i in random_indices])
            labels_points = labels * torch.ones(num_points, device=input_masks.device)
        else:
            random_points = torch.empty(0, device=input_masks.device)
            labels_points = torch.empty(0, device=input_masks.device)

        return random_points, labels_points

    def get_input_masks(self, consistency_masks, classes):
        input_masks = F.one_hot(consistency_masks.long(), num_classes=self.config.num_labels)
        input_masks = torch.permute(input_masks, (2, 0, 1))
        input_masks = input_masks[classes]

        return input_masks

    def update_input_masks_classes(self, input_masks, classes):
        if len(classes) > 1 and 0 in classes:
            input_masks = input_masks[1:]
            classes = classes[1:]
        elif classes == [0]:
            input_masks = torch.zeros(
                size=self.pixel_values_sizes,
                device=input_masks.device,
                dtype=input_masks.dtype
            ).unsqueeze(dim=0)
            classes = [-1]

        return input_masks, classes

    def batch_predict(self, inputs):
        pred_masks = []
        inputs_size = len(inputs['pixel_values'])
        self.model = self.model.to(dtype=inputs['pixel_values'].dtype, device=inputs['pixel_values'].device)

        for start_idx in range(0, inputs_size, self.config.sam_batch_size):
            end_idx = min(start_idx + self.config.sam_batch_size, inputs_size)
            sam_batch = self.extract_sam_batch(inputs, start_idx, end_idx)
            outputs = self.model(**sam_batch)
            pred_masks.append(outputs.pred_masks)

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

    def post_process_outputs(self, inputs, outputs, classes, batch_idx):
        pred_masks = []

        masks = self.processor.post_process_masks(
            masks=outputs,
            original_sizes=inputs['original_sizes'],
            reshaped_input_sizes=inputs['reshaped_input_sizes'],
            binarize=False
        )
        masks = F.softmax(torch.cat(masks).squeeze(dim=1))

        for idx in range(self.config.sam_batch_size):
            pred_mask = []
            mask_batch_idx = masks[torch.Tensor(batch_idx) == idx]
            mask_class_idx = [classes[i] for i in range(len(batch_idx)) if batch_idx[i] == idx]

            class_idx_replaced = 0
            for label in range(self.config.num_labels):
                if label in mask_class_idx:
                    mask = (mask_batch_idx[class_idx_replaced] > self.config.sam_threshold).to(dtype=torch.float16)
                    pred_mask.append(mask)
                    class_idx_replaced += 1
                elif label == 0:
                    pred_mask.append(1e-8 * torch.ones(masks.shape[-2:], device=masks.device, dtype=torch.float16))
                else:
                    pred_mask.append(torch.zeros(masks.shape[-2:], device=masks.device, dtype=torch.float16))

            pred_mask = torch.stack(pred_mask)

            if self.current_step == 'validation' and self.current_batch_idx == 0 and idx == 0:
                self.log_output_masks(inputs, pred_mask)

            pred_masks.append(pred_mask.argmax(dim=0))

        sam_masks = torch.stack(pred_masks)
        sam_masks = func.reshape_tensor(sam_masks, size=(self.config.tile_size, self.config.tile_size))

        return sam_masks

    def log_input_masks(self, inputs, input_masks_i, classes_i):
        inputs = func.reshape_tensor(inputs['pixel_values'][0], size=self.pixel_values_sizes)
        inputs = torch.moveaxis(inputs, 0, -1).numpy(force=True)
        input_masks_i = input_masks_i.numpy(force=True)

        masks = {}
        class_idx_logged = 0
        for class_label in range(len(self.class_labels)):
            if class_label in classes_i:
                masks[f'input_masks_{class_label}'] = {'mask_data': input_masks_i[class_idx_logged]}
                class_idx_logged += 1
            else:
                masks[f'input_masks_{class_label}'] = {'mask_data': np.zeros(shape=self.pixel_values_sizes)}

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
