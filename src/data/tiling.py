import math
import os

import numpy as np

from utils import classes as uC
from utils import functions as uF


def get_tiles(size_tile: int = 384):
    tiles = []
    config = uF.load_config('main')
    path_labels = config.path.data.raw.train.labels

    num_h_tiles, overlap_h, num_w_tiles, overlap_w = get_num_tiles(config, size_tile)
    bboxes = get_coords_tile(config, size_tile, num_h_tiles, overlap_h, num_w_tiles, overlap_w)
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


def get_num_tiles(config: uC.Config, size_tile: int):
    size_h = config.data.size_h
    size_w = config.data.size_w
    num_h_tiles = config.data.size_h / size_tile
    num_w_tiles = config.data.size_w / size_tile

    overlap_h = math.ceil(math.ceil(size_tile * math.ceil(num_h_tiles) - size_h) / math.floor(num_h_tiles))
    overlap_w = math.ceil(math.ceil(size_tile * math.ceil(num_w_tiles) - size_w) / math.floor(num_w_tiles))
    num_h_tiles = math.ceil(num_h_tiles)
    num_w_tiles = math.ceil(num_w_tiles)

    return num_h_tiles, overlap_h, num_w_tiles, overlap_w


def get_coords_tile(config: uC.Config, size_tile: int, num_h_tiles: int, overlap_h: int, num_w_tiles: int,
                    overlap_w: int):
    size_h = config.data.size_h
    size_w = config.data.size_w
    coords_tile = []

    for i in range(num_h_tiles):
        for j in range(num_w_tiles):
            x0 = max(0, i * (size_tile - overlap_h))
            y0 = max(0, j * (size_tile - overlap_w))
            x1 = min(size_h, x0 + size_tile)
            y1 = min(size_w, y0 + size_tile)

            if x1 + 1 == size_h:
                x0 += 1
                x1 += 1

            if y1 + 1 == size_w:
                y0 += 1
                y1 += 1

            coords_tile.append((x0, y0, x1, y1))

    return coords_tile


if __name__ == "__main__":
    tiles = get_tiles()
    print()
