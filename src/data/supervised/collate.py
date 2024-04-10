import torch

def mask2former_collate_fn_training(batch):
    class_labels = []
    mask_labels = []
    pixel_values = []
    pixel_mask = []
    for el in batch:
        class_labels.extend(el['class_labels'])
        mask_labels.extend(el['mask_labels'])
        pixel_values.append(el['pixel_values'])
        pixel_mask.append(el['pixel_mask'])

    return {
        'pixel_values': torch.concat(pixel_values),
        'pixel_mask': torch.concat(pixel_mask),
        'class_labels': class_labels,
        'mask_labels': mask_labels
    }


def mask2former_collate_fn_inference(batch):
    pixel_values = []
    pixel_mask = []
    for el in batch:
        pixel_values.append(el['pixel_values'])
        pixel_mask.append(el['pixel_mask'])

    return {
        'pixel_values': torch.concat(pixel_values),
        'pixel_mask': torch.concat(pixel_mask)
    }


def get_collate_fn_training(config):
    if config['model_name'] == 'mask2former':
        return mask2former_collate_fn_training
    else:
        raise ValueError(f"model_name expected 'mask2former' but received {config['model_name']}")


def get_collate_fn_inference(config):
    if config['model_name'] == 'mask2former':
        return mask2former_collate_fn_inference
    else:
        raise ValueError(f"model_name expected 'mask2former' but received {config['model_name']}")

