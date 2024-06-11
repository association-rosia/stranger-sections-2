import io
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import wandb
from PIL import Image
from transformers import SamModel, SamProcessor

from src.utils import func
from src.utils.cls import Config


class SamForSemiSupervised(nn.Module):
    def __init__(self, config: Config, class_labels):
        super(SamForSemiSupervised, self).__init__()

        self.config = config
        self.class_labels = class_labels

        self.model, self.processor = self.load_model_processor()
        self.images_sizes = (1024, 1024)
        self.value_to_label = {0: 0, 1: 1}  # we can have 0: -1
        self.log_pixel_values = None

        self.current_step = None
        self.current_batch_idx = None

    @torch.no_grad()
    def forward(self, pixel_values, template_masks):
        image_embeddings, input_masks, classes, indices = self.get_inputs(pixel_values, template_masks)
        input_points, input_labels = self.build_input_points_labels(input_masks)
        self.log_input_masks(pixel_values, input_masks, input_points, input_labels, classes, indices)
        huggingface_inputs = self.build_huggingface_inputs(pixel_values, image_embeddings, input_points, input_labels)
        outputs = self.model(**huggingface_inputs)
        pred_masks = self.post_process_outputs(huggingface_inputs, outputs, classes, indices)
        self.log_pred_mask(pixel_values, template_masks, pred_masks)

        return pred_masks

    def get_image_embeddings(self, pixel_values):
        pixel_values = func.reshape_tensor(pixel_values, size=self.images_sizes)
        image_embeddings = self.model.get_image_embeddings(pixel_values)

        return image_embeddings

    def get_inputs(self, pixel_values, masks):
        image_embeddings = self.get_image_embeddings(pixel_values)
        image_embeddings_list, input_masks_list, classes, indices = [], [], [], []

        for i in range(self.config.batch_size):
            input_masks_i, classes_i = self.stack_input_masks(
                masks=masks,
                index=i
            )
            input_masks_list.append(input_masks_i)
            classes += classes_i

            image_embeddings_i, indices_i = self.stack_image_embeddings(
                image_embeddings=image_embeddings,
                classes=classes_i,
                index=i,

            )
            image_embeddings_list.append(image_embeddings_i)
            indices += indices_i

        image_embeddings = torch.cat(image_embeddings_list)
        input_masks = torch.cat(input_masks_list)

        return image_embeddings, input_masks, classes, indices

    @staticmethod
    def stack_image_embeddings(image_embeddings, classes, index):
        images = torch.stack([image_embeddings[index]] * len(classes))
        indices = [index] * len(classes)

        return images, indices

    def build_huggingface_inputs(self, pixel_values, image_embeddings, input_points, input_labels):
        inputs = self.processor(
            images=pixel_values,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors='pt'
        )
        inputs['multimask_output'] = False
        inputs.pop('pixel_values', None)
        inputs.update({'image_embeddings': image_embeddings})
        inputs.update({'original_sizes': inputs['original_sizes'][0].repeat(len(image_embeddings), 1)})
        inputs.update({'reshaped_input_sizes': inputs['reshaped_input_sizes'][0].repeat(len(image_embeddings), 1)})

        inputs = {
            k: v.to(device=image_embeddings[0].device)
            if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        return inputs

    def stack_input_masks(self, masks, index):
        masks = masks[index]
        classes = torch.unique(masks).tolist()
        masks = func.reshape_tensor(masks, size=self.images_sizes)
        input_masks = self.get_input_masks(masks, classes)
        input_masks, classes = self.update_input_masks_classes(input_masks, classes)

        return input_masks, classes

    def build_input_points_labels(self, input_masks):
        valid_points_zeros, valid_points_ones, num_points_zeros, num_points_ones = self.get_valid_points(input_masks)

        input_points_zeros, input_points_ones, input_labels_zeros, input_labels_ones = self.get_input_points_labels(
            valid_points_zeros=valid_points_zeros,
            valid_points_ones=valid_points_ones,
            num_points_zeros=num_points_zeros,
            num_points_ones=num_points_ones
        )

        input_points = [[input_points_zeros[i] + input_points_ones[i]] for i in range(len(input_masks))]
        input_labels = [[input_labels_zeros[i] + input_labels_ones[i]] for i in range(len(input_masks))]

        return input_points, input_labels

    def get_valid_points(self, input_masks):
        valid_points_zeros = self.get_coordinates(input_masks, num_layers=10, value=0)
        valid_points_ones = self.get_coordinates(input_masks, num_layers=10, value=1)
        num_points_zeros, num_points_ones = self.get_num_points(input_masks, valid_points_zeros, valid_points_ones)

        return valid_points_zeros, valid_points_ones, num_points_zeros, num_points_ones

    def get_input_points_labels(self, valid_points_zeros, valid_points_ones, num_points_zeros, num_points_ones):
        input_points_zeros, input_labels_zeros = self.get_random_points(
            valid_points_zeros,
            num_points_zeros,
            value=0
        )

        input_points_ones, input_labels_ones = self.get_random_points(
            valid_points_ones,
            num_points_ones,
            value=1
        )

        return input_points_zeros, input_points_ones, input_labels_zeros, input_labels_ones

    def get_random_points(self, valid_points, num_points, value):
        random_points, labels_points = [], []

        for i in range(len(num_points)):
            if num_points[i] > 0:
                random_points_i = random.sample(valid_points[i], k=num_points[i])
                random_points.append(random_points_i)
                labels_points_i = [self.value_to_label[value] for _ in range(len(random_points_i))]
                labels_points.append(labels_points_i)
            else:
                random_points.append([])
                labels_points.append([])

        return random_points, labels_points

    @staticmethod
    def get_coordinates(input_masks, num_layers, value):
        device = input_masks.device
        mask = (input_masks == value).float()
        kernel = torch.ones((1, 1, num_layers, num_layers), device=device)

        mask = mask.unsqueeze(1)
        mask = F.conv2d(mask, kernel, padding=1)
        mask = (mask == kernel.sum()).float()
        mask = mask.squeeze(1)

        coordinates_list = []
        for i in range(len(input_masks)):
            coordinates = (mask[i] == 1).nonzero(as_tuple=False)
            coordinates = coordinates[:, [1, 0]].tolist()
            coordinates_list.append(coordinates)

        return coordinates_list

    def get_num_points(self, input_masks, valid_points_zeros, valid_points_ones):
        num_points_zeros_list, num_points_ones_list = [], []

        for i in range(len(input_masks)):
            count_ones = (input_masks[i] == 1).sum().item()
            count_zeros = (input_masks[i] == 0).sum().item()

            if count_ones < count_zeros:
                num_points_ones = min(
                    int(self.config.sam_input_points_rate_ones * self.config.sam_input_points_num),
                    len(valid_points_ones[i])
                )
                num_points_ones_list.append(num_points_ones)
                num_points_zeros_list.append(self.config.sam_input_points_num - num_points_ones)
            else:
                num_points_zeros = min(
                    int((1 - self.config.sam_input_points_rate_ones) * self.config.sam_input_points_num),
                    len(valid_points_zeros[i])
                )
                num_points_zeros_list.append(num_points_zeros)
                num_points_ones_list.append(self.config.sam_input_points_num - num_points_zeros)

        return num_points_zeros_list, num_points_ones_list

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
                size=self.images_sizes,
                device=input_masks.device,
                dtype=input_masks.dtype
            ).unsqueeze(dim=0)
            classes = [-1]

        return input_masks, classes

    @staticmethod
    def extract_sam_batch(flatten_inputs, start_idx, end_idx):
        sam_batch = {}

        for key, value in flatten_inputs.items():
            if isinstance(value, torch.Tensor):
                sam_batch[key] = value[start_idx:end_idx]
            else:
                sam_batch[key] = value

        return sam_batch

    def post_process_outputs(self, inputs, outputs, classes, indices):
        final_masks = []

        masks = self.processor.image_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs['original_sizes'],
            reshaped_input_sizes=inputs['reshaped_input_sizes']
        )
        masks = torch.cat(masks).squeeze(dim=1)

        for i in list(set(indices)):
            stack_mask = []
            mask_i = masks[torch.Tensor(indices) == i]
            classes_i = [classes[j] for j in range(len(indices)) if indices[j] == i]
            stack = 0

            for label in range(self.config.num_labels):
                if label == 0:
                    stack_mask.append(0.1 * torch.ones(masks.shape[-2:], device=masks.device, dtype=masks.dtype))
                elif label in classes_i:
                    stack_mask.append((outputs.iou_scores[stack] * mask_i[stack]))
                    stack += 1
                else:
                    stack_mask.append(torch.zeros(masks.shape[-2:], device=masks.device, dtype=masks.dtype))

            stack_mask = torch.stack(stack_mask)
            self.log_output_masks(stack_mask, i)
            stack_mask = stack_mask.argmax(dim=0)
            final_masks.append(stack_mask)

        final_masks = torch.stack(final_masks)
        final_masks = func.reshape_tensor(final_masks, size=(self.config.tile_size, self.config.tile_size))

        return final_masks

    def log_input_masks(self, pixel_values, input_masks, input_points, input_labels, classes, indices):
        if self.current_step == 'validation' and self.current_batch_idx == 0:
            pixel_values = func.reshape_tensor(pixel_values[0], size=self.images_sizes)
            self.log_pixel_values = torch.moveaxis(pixel_values, 0, -1).numpy(force=True)

            input_masks = input_masks.numpy(force=True)
            classes_i = [classes[i] for i in range(len(indices)) if indices[i] == 0]

            stack = 0
            sam_input_masks = {}
            sam_input_points = []

            for label in range(self.config.num_labels):
                if label in classes_i:
                    sam_input_masks[f'input_masks_{label}'] = {
                        'mask_data': input_masks[stack],
                        'class_labels': self.class_labels
                    }

                    sam_input_points.append(
                        self.show_points_on_image(
                            input_points=np.array(input_points[stack][0]),
                            input_labels=np.array(input_labels[stack][0])
                        )
                    )

                    stack += 1
                else:
                    sam_input_masks[f'input_masks_{label}'] = {
                        'mask_data': np.zeros(shape=self.images_sizes),
                        'class_labels': self.class_labels
                    }

                    sam_input_points.append(
                        self.show_points_on_image(
                            input_points=None,
                            input_labels=None
                        )
                    )

            wandb.log({
                'val/sam_input_masks': wandb.Image(
                    self.log_pixel_values,
                    masks=dict(sorted(sam_input_masks.items()))
                )
            })

            wandb.log({
                'val/sam_input_points': sam_input_points
            })

    def show_points_on_image(self, input_points, input_labels):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.log_pixel_values.astype(np.float32))

        if input_points is not None and input_labels is not None:
            pos_points = input_points[input_labels == self.value_to_label[1]]
            neg_points = input_points[input_labels == self.value_to_label[0]]

            marker_size = 400
            plt.scatter(pos_points[:, 0], pos_points[:, 1], color='#40E0D0', marker='o', s=marker_size)
            plt.scatter(neg_points[:, 0], neg_points[:, 1], color='#FFC0CA', marker='o', s=marker_size)

        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        return wandb.Image(Image.open(buf))

    def log_output_masks(self, output_masks, i):
        if self.current_step == 'validation' and self.current_batch_idx == 0 and i == 0:
            output_masks = func.reshape_tensor(output_masks, size=self.images_sizes)
            output_masks = output_masks.numpy(force=True)

            sam_output_masks = {
                f'output_masks_{j}': {
                    'mask_data': output_masks[j] > 0.1,
                    'class_labels': self.class_labels
                } for j in range(len(output_masks))
            }

            wandb.log({
                'val/sam_output_masks': wandb.Image(
                    self.log_pixel_values,
                    masks=dict(sorted(sam_output_masks.items()))
                )
            })

    def log_pred_mask(self, pixel_values, template_masks, pred_masks):
        if self.current_step == 'validation' and self.current_batch_idx == 0:
            inputs = torch.moveaxis(pixel_values[0], 0, -1).numpy(force=True)
            template_masks = template_masks[0].numpy(force=True)
            pred_masks = pred_masks[0].numpy(force=True)

            wandb.log({
                'val/sam': wandb.Image(
                    inputs,
                    masks={
                        'consistency': {
                            'mask_data': template_masks,
                            'class_labels': self.class_labels
                        },
                        'sam': {
                            'mask_data': pred_masks,
                            'class_labels': self.class_labels
                        }
                    }
                )
            })

    def load_model_processor(self):
        with torch.no_grad():
            model = SamModel.from_pretrained(self.config.sam_model_id).half()

        processor = SamProcessor.from_pretrained(self.config.sam_model_id, do_rescale=False)

        for param in model.parameters():
            param.requires_grad = False

        return model, processor
