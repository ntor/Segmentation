#!/usr/bin/env python3

import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
from ShapeGenerator import ShapeGenerator
from ChanVese import ChanVese
from tqdm import tqdm

SAMPLE_FOLDER_PREFIX = "image_"

IMAGE_TYPE_NAMES = {"dirty": "dirty.png", "clean": "clean.png"}

SEGMENTATION_TYPE_NAMES = {
    "clean": "clean_seg.npy",
    "chan-vese": "dirty_cv_seg.npy",
    "deep-segmentation": "dirty_ds_seg.npy",
}


class ImageDataset(Dataset):
    def __init__(self, data_root, image_type="dirty"):
        self.data_root = data_root
        self.image_type = image_type
        if not os.path.isdir(data_root):
            print("ERROR: data_root is not a valid directory")

    def __len__(self):
        root_list = os.listdir(self.data_root)
        image_folders = [s for s in root_list if s.startswith(SAMPLE_FOLDER_PREFIX)]
        return len(image_folders)

    def __getitem__(self, idx):
        im = Image.open(
            os.path.join(
                self.data_root,
                "image_{}".format(idx),
                IMAGE_TYPE_NAMES[self.image_type],
            )
        )
        return im


class SegmentationDataset(Dataset):
    def __init__(self, data_root, seg_type="dirty"):
        self.data_root = data_root
        self.seg_type = seg_type
        if not os.path.isdir("data_root"):
            print("ERROR: data_root is not a valid directory")

    def __len__(self):
        root_list = os.listdir(self.data_root)
        image_folders = [s for s in root_list if s.startswith(SAMPLE_FOLDER_PREFIX)]
        return len(image_folders)

    def __getitem__(self, idx):
        seg = np.load(
            os.path.join(
                self.data_root,
                SAMPLE_FOLDER_PREFIX + "{}".format(idx),
                SEGMENTATION_TYPE_NAMES[self.seg_type],
            )
        )
        return seg


def generate_data(times, root_dir, size=(128, 128)):
    for i in tqdm(range(times)):
        sample_folder = os.path.join(root_dir, SAMPLE_FOLDER_PREFIX + "{}".format(i))
        os.mkdir(sample_folder)
        shapes = ShapeGenerator(128, 128)
        shapes.add_polygon(times=np.random.randint(2, 6))
        shapes.add_ellipse(times=np.random.ranint(2, 6))
        shapes.image.save(
            fp=os.path.join(sample_folder, IMAGE_TYPE_NAMES["clean"]), format="PNG"
        )
        np.save(
            file=os.path.join(sample_folder, SEGMENTATION_TYPE_NAMES["clean"]),
            arr=np.array(shapes.image) / 255,
        )
        shapes.add_holes(np.random.randint(15, 60))
        shapes.add_noise()
        shapes.image.save(
            fp=os.path.join(sample_folder, IMAGE_TYPE_NAMES["dirty"]), format="PNG"
        )

        shapes = ChanVese(shapes.image)
        shapes.run(steps=500, show_iterations=False)
        # save in chan-vese
        np.save(
            file=os.path.join(sample_folder, SEGMENTATION_TYPE_NAMES["chan-vese"]),
            arr=shapes.u,
        )
