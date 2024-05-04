import math
import os

import numpy as np

from src.utils import func
from src.utils.cls import Config


class Tiler:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.bboxes = self._build_bboxes()

    def build(self, labeled: bool = True):
        bboxes = self._build_bboxes()

        if labeled:
            tiles = self._get_labeled_tiles(bboxes)
        else:
            tiles = self._get_unlabeled_tiles(bboxes)

        return tiles

    def tile(self, image: np.ndarray) -> list[np.ndarray]:
        tiles = []
        bboxes = self._build_bboxes()

        for x0, y0, x1, y1 in bboxes:
            tiles.append(image[:, x0:x1, y0:y1])

        return tiles

    def untile(self, tiles: list[np.ndarray]) -> np.ndarray:
        bboxes = self._build_bboxes()
        num_labels = self.config.num_labels
        size_h = self.config.data.size_h
        size_w = self.config.data.size_w
        image = np.zeros((num_labels, size_h, size_w), np.float16)

        for (x0, y0, x1, y1), tile in zip(bboxes, tiles):
            image[:, x0:x1, y0:y1] += tile

        return image

    def _build_bboxes(self):
        if self.config.tile_size is None:
            self.config.tile_size = self.config.self.config.tile_size
            bboxes = self.bboxes
        else:
            num_h_tiles, overlap_h, num_w_tiles, overlap_w = self._get_num_tiles()
            bboxes = self._get_coords_tile(num_h_tiles, overlap_h, num_w_tiles, overlap_w)

        return bboxes

    def _get_labeled_tiles(self, bboxes: list):
        tiles = []
        path_labels = self.config.path.data.raw.train.labels
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

    def _get_unlabeled_tiles(self, bboxes: list):
        tiles = []
        path_images = self.config.path.data.raw.train.unlabeled
        path_images = func.get_notebooks_path(path_images)

        files = [file for file in os.listdir(path_images) if file.endswith('.jpg')]

        for file in files:
            image = file.split('.')[0]

            for bbox in bboxes:
                tiles.append({'image': image, 'bbox': bbox})

        return tiles

    def _get_num_tiles(self):
        size_h = self.config.data.size_h
        size_w = self.config.data.size_w
        num_h_tiles = self.config.data.size_h / self.config.tile_size
        num_w_tiles = self.config.data.size_w / self.config.tile_size

        overlap_h = math.ceil(math.ceil(self.config.tile_size * math.ceil(num_h_tiles) - size_h) / math.floor(num_h_tiles))
        overlap_w = math.ceil(math.ceil(self.config.tile_size * math.ceil(num_w_tiles) - size_w) / math.floor(num_w_tiles))
        num_h_tiles = math.ceil(num_h_tiles)
        num_w_tiles = math.ceil(num_w_tiles)

        return num_h_tiles, overlap_h, num_w_tiles, overlap_w

    def _get_coords_tile(self, num_h_tiles: int, overlap_h: int, num_w_tiles: int,
                         overlap_w: int):
        size_h = self.config.data.size_h
        size_w = self.config.data.size_w
        coords_tile = []

        for i in range(num_h_tiles):
            for j in range(num_w_tiles):
                x0 = max(0, i * (self.config.tile_size - overlap_h))
                y0 = max(0, j * (self.config.tile_size - overlap_w))
                x1 = min(size_h, x0 + self.config.tile_size)
                y1 = min(size_w, y0 + self.config.tile_size)

                if x1 + 1 == size_h:
                    x0 += 1
                    x1 += 1

                if y1 + 1 == size_w:
                    y0 += 1
                    y1 += 1

                coords_tile.append((x0, y0, x1, y1))

        return coords_tile


def _debug():
    from PIL import Image
    main_config = func.load_config('main', loading='dict')
    wandb_config = func.load_config('segformer', 'supervised', loading='dict')
    config = Config(main_config, wandb_config)
    tiler = Tiler(config)
    tiles = tiler.build(labeled=False)
    image = Image.open('data/raw/train/unlabeled/0a6odx.jpg')
    image = np.moveaxis(np.asarray(image), -1, 0)
    tiles = tiler.tile(image)

    return


if __name__ == "__main__":
    _debug()
