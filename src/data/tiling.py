import math
import os

import numpy as np

from src.utils import func
from src.utils.cls import Config


# TODO: Create a class Tiler
def build(config: Config, labeled: bool = True):
    num_h_tiles, overlap_h, num_w_tiles, overlap_w = get_num_tiles(config)
    bboxes = get_coords_tile(config, num_h_tiles, overlap_h, num_w_tiles, overlap_w)

    if labeled:
        tiles = get_labeled_tiles(config, bboxes)
    else:
        tiles = get_unlabeled_tiles(config, bboxes)

    return tiles


def get_labeled_tiles(config: Config, bboxes: list):
    tiles = []
    path_labels = config.path.data.raw.train.labels
    path_labels = func.get_notebooks_path(path_labels)

    npy_files = [file for file in os.listdir(path_labels) if file.endswith('.npy')]

    for npy_file in npy_files:
        npy_data = np.load(os.path.join(path_labels, npy_file))
        image = npy_file.split('_')[0]

        for bbox in bboxes:
            x0, y0, x1, y1 = bbox
            cropped_npy_data = npy_data[x0:x1, y0:y1]

            if len(np.unique(cropped_npy_data).tolist()) > 1:
                tiles.append({'image': image, 'bbox': bbox})

    return tiles


def get_unlabeled_tiles(config: Config, bboxes: list):
    tiles = []
    path_images = config.path.data.raw.train.unlabeled
    path_images = func.get_notebooks_path(path_images)

    files = [file for file in os.listdir(path_images) if file.endswith('.jpg')]

    for file in files:
        image = file.split('.')[0]

        for bbox in bboxes:
            tiles.append({'image': image, 'bbox': bbox})

    return tiles


def get_num_tiles(config: Config):
    size_h = config.data.size_h
    size_w = config.data.size_w
    num_h_tiles = config.data.size_h / config.tile_size
    num_w_tiles = config.data.size_w / config.tile_size

    overlap_h = math.ceil(math.ceil(config.tile_size * math.ceil(num_h_tiles) - size_h) / math.floor(num_h_tiles))
    overlap_w = math.ceil(math.ceil(config.tile_size * math.ceil(num_w_tiles) - size_w) / math.floor(num_w_tiles))
    num_h_tiles = math.ceil(num_h_tiles)
    num_w_tiles = math.ceil(num_w_tiles)

    return num_h_tiles, overlap_h, num_w_tiles, overlap_w


def get_coords_tile(config: Config, num_h_tiles: int, overlap_h: int, num_w_tiles: int, overlap_w: int):
    size_h = config.data.size_h
    size_w = config.data.size_w
    coords_tile = []

    for i in range(num_h_tiles):
        for j in range(num_w_tiles):
            x0 = max(0, i * (config.tile_size - overlap_h))
            y0 = max(0, j * (config.tile_size - overlap_w))
            x1 = min(size_h, x0 + config.tile_size)
            y1 = min(size_w, y0 + config.tile_size)

            if x1 + 1 == size_h:
                x0 += 1
                x1 += 1

            if y1 + 1 == size_w:
                y0 += 1
                y1 += 1

            coords_tile.append((x0, y0, x1, y1))

    return coords_tile


if __name__ == "__main__":
    config = func.load_config('main', loading='object')
    tiles = build(config=config, labeled=False)
