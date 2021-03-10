#!/usr/bin/env python3

import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from ClassFiles.ShapeGenerator import ShapeGenerator
from ClassFiles.ChanVese import ChanVese
from tqdm import tqdm

# This file implements two torch Dataset classes, 'ImageDataset' and
# 'SegmentationDataset', and a function 'generate_data'. The datasets are
# initialised with a path to a directory in which we assume a folder structure
# as follows:
#     + image_0
#      - clean.png
#      - dirty.png
#      - clean_seg.npy
#      - dirty_cv_seg.npy
#    + image_1
#      - clean.png
#      - ...
#
# The dataloaders then provide an interface for loading, for example the clean
# dirty images, or their segmentations.
#
# The actual names are not hardcoded but can be adjusted in the constants below.
# The initialised Dataset instances are not to be used directly but rather in
# order to initialise a torch DataLoader object.
#
# The 'generate_data' function can be used in order to populate a folder with
# artifical data as described above.


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
        return transforms.ToTensor()(im)


class SegmentationDataset(Dataset):
    def __init__(self, data_root, seg_type="chan-vese"):
        self.data_root = data_root
        self.seg_type = seg_type
        if not os.path.isdir(data_root):
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
        return torch.Tensor(seg)


def generate_data(times, root_dir, size=(128, 128), append=True):
    if append:
        start_index = sum(
            1 for s in os.listdir(root_dir) if s.startswith(SAMPLE_FOLDER_PREFIX)
        )
    else:
        start_index = 0

    for i in tqdm(range(times)):
        sample_folder = os.path.join(
            root_dir, SAMPLE_FOLDER_PREFIX + "{}".format(i + start_index)
        )
        try:
            os.mkdir(sample_folder)
        except FileExistsError:
            pass

        shapes = ShapeGenerator(128, 128)
        shapes.add_polygon(times=np.random.randint(10, 25))
        shapes.add_ellipse(times=np.random.randint(10, 25))
        shapes.add_holes(numholes=np.random.randint(10, 40))

        shapes.image.save(
            fp=os.path.join(sample_folder, IMAGE_TYPE_NAMES["clean"]), format="PNG"
        )
        np.save(
            file=os.path.join(sample_folder, SEGMENTATION_TYPE_NAMES["clean"]),
            arr=np.array(shapes.image) / 255,
        )
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
