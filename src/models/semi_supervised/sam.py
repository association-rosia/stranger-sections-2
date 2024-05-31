import torch
from transformers import SamModel, SamImageProcessor

from src.utils.cls import Config


class SamForSemiSupervised:
    def __init__(self, config: Config):
        self.config = config

        self.loss_fct = nn.CrossEntropyLoss()

    @torch.no_grad()
    def forward(self, inputs, consistency_logits):
        inputs, _ = inputs
        consistency_masks = self.logits_to_masks(consistency_logits)

        flatten_inputs, classes, indices = self.get_flatten_inputs(consistency_masks, inputs)
        flatten_outputs = self.sam_predict(flatten_inputs)

        sam_masks = self.post_process_flatten_outputs(flatten_inputs, flatten_outputs, classes, indices)

        loss = self.sam_loss_fct(consistency_logits, sam_masks.long())

        if self.current_step == 'validation':
            self.log('val/sam_loss', loss, on_epoch=True, sync_dist=True)

            if self.current_batch_idx == 0:
                self.log_sam_images(inputs, consistency_masks, sam_masks)

        return loss

    def get_flatten_inputs(self, consistency_masks, inputs):
        input_masks, pixel_values, classes, indices = [], [], [], []

        for i in range(self.config.batch_size):
            input_masks, input_masks_i, classes, classes_i = self.create_input_masks(i, consistency_masks, input_masks,
                                                                                     classes)
            pixel_values, indices = self.create_pixel_values(i, input_masks_i, inputs, pixel_values, indices)

            if self.current_step == 'validation' and self.current_batch_idx == 0 and i == 0:
                self.log_input_masks(inputs, input_masks_i, classes_i)

        flatten_inputs = self.create_flatten_inputs(consistency_masks, input_masks, pixel_values)

        return flatten_inputs, classes, indices

    def create_input_masks(self, i, consistency_masks, input_masks, classes):
        consistency_masks_i = consistency_masks[i]
        classes_i = torch.unique(consistency_masks_i).tolist()
        consistency_masks_i = self.reshape_tensor(consistency_masks_i, size=self.input_masks_sizes, is_2d=True)  # TODO
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

    def create_pixel_values(self, i, input_masks_i, inputs, pixel_values, indices):
        num_masks = len(input_masks_i) if input_masks_i is not None else 1
        pixel_values_i = torch.stack([inputs['pixel_values'][i] for _ in range(num_masks)])
        pixel_values_i = self.reshape_tensor(pixel_values_i)  # TODO
        pixel_values.append(pixel_values_i)
        indices += [i for _ in range(num_masks)]

        return pixel_values, indices

    def create_flatten_inputs(self, consistency_masks, input_masks, pixel_values):
        input_masks = torch.cat(input_masks).unsqueeze(dim=1)
        pixel_values = torch.cat(pixel_values)
        flatten_inputs = self.sam_processor(images=pixel_values, return_tensors='pt')
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
            flatten_outputs = self.sam(**sam_batch)
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

        masks = self.sam_processor.post_process_masks(
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
        sam_masks = self.reshape_tensor(sam_masks, size=self.input_image_sizes, is_3d=True)  # TODO

        return sam_masks

    def load_sam(self):
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
