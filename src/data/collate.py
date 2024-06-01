import torch
from src.utils.cls import Config, TrainingMode

class SS2Mask2formerCollateFn:
    def __init__(self, config: Config, training: bool) -> None:
        self.training = training
        self.config = config
        self.collate_fn = self._get_collate_fn()

    def __call__(self, batch):
        return self.collate_fn(batch)
    
    def _get_collate_fn(self):
        if self.training:
            if self.config.mode == TrainingMode.SEMI_SUPERVISED:
                return self.train_semi_supervised_collate_fn
            elif self.config.mode == TrainingMode.SUPERVISED:
                return self.train_supervised_collate_fn
            else:
                raise NotImplementedError
        else:
            return self.inference_collate_fn
            
    def train_supervised_collate_fn(self, batch):
        class_labels = []
        mask_labels = []
        pixel_values = []
        pixel_mask = []
        for el in batch:
            class_labels.append(el['class_labels'])
            mask_labels.append(el['mask_labels'])
            pixel_values.append(el['pixel_values'])
            pixel_mask.append(el['pixel_mask'])

        return {
            'pixel_values': torch.stack(pixel_values),
            'pixel_mask': torch.stack(pixel_mask),
            'class_labels': class_labels,
            'mask_labels': mask_labels
        }
    
    def train_semi_supervised_collate_fn(self, batch):
        batch_segmentation = []
        batch_consistency_student = []
        batch_consistency_teacher = []
        for el in batch:
            batch_segmentation.append(el[0])
            batch_consistency_student.append(el[1][0])
            batch_consistency_teacher.append(el[1][1])

        segmentation_inputs = self.train_supervised_collate_fn(batch_segmentation)
        consistency_student_inputs = self.inference_collate_fn(batch_consistency_student)
        consistency_teacher_inputs = self.inference_collate_fn(batch_consistency_teacher)

        return segmentation_inputs, (consistency_student_inputs, consistency_teacher_inputs)

    def inference_collate_fn(self, batch):
        pixel_values = []
        pixel_mask = []
        for el in batch:
            pixel_values.append(el['pixel_values'])
            pixel_mask.append(el['pixel_mask'])

        return {
            'pixel_values': torch.stack(pixel_values),
            'pixel_mask': torch.stack(pixel_mask)
        }


def segformer_collate_fn_training(batch):
    pixel_values = []
    labels = []
    for el in batch:
        pixel_values.append(el['pixel_values'])
        labels.append(el['labels'])

    return {
        'pixel_values': torch.concat(pixel_values),
        'labels': torch.concat(labels)
    }


def segformer_collate_fn_inference(batch):
    pixel_values = [el['pixel_values'] for el in batch]

    return {'pixel_values': torch.concat(pixel_values)}


def get_collate_fn_training(config):
    if config.model_name == 'mask2former':
        return SS2Mask2formerCollateFn(config=config, training=True)
    elif config.model_name == 'segformer':
        return segformer_collate_fn_training
    else:
        raise ValueError(f"model_name expected 'mask2former' but received {config.model_name}")


def get_collate_fn_inference(config):
    if config.model_name == 'mask2former':
        return SS2Mask2formerCollateFn(config=config, training=False)
    elif config.model_name == 'segformer':
        return segformer_collate_fn_inference
    else:
        raise ValueError(f"model_name expected 'mask2former' but received {config.model_name}")
