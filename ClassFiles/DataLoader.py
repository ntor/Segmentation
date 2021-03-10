#!/usr/bin/env python3

import torch
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Path constants used to load the generated images. See
# 'get_generated_dataloader' for more information.
GENERATED_DATA_PATH = {
    "train": "./data/train/",
    "eval": "./data/eval/"
}

GENERATED_DATA_FOLDERS = {
    "clean": "clean/",
    "dirty": "dirty/",
    "chan-vese": "chan-vese/",
}

"""
NOTE: ImageFolder datasets are tuples,
[0] is the tensor image we want,
[1] is the Folder index it came from (suppose to be used for labels like 'dogs', 'cats'), for us they are all just 0
"""

def generate_data():


def get_generated_dataloader(train_or_eval, data_type, batch_size=20, shuffle=True):
    """Returns a torch.utils.data.DataLoader object for iterating over the generated
    data. The images are chosen from one of the subfolders in
    GENERATED_DATA_PATH, according to the value corresponding to DATA_TYPE in
    the GENERATED_DATA_FOLDERS dict. The images will be loaded as tensors.

    Parameters:
    
    TRAIN_OR_EVAL (string): one of the key values of GENERATED_DATA_PATH

    DATA_TYPE (string): one of the key values of GENERATED_DATA_FOLDERS. Used to
    choose the folder from which to load the images

    BATCH_SIZE (int): is forwarded to the parameters for the DataLoader object

    SHUFFLE (bool):  is forwarded to the parameters for the DataLoader object
    """

    try:
        path_folder = GENERATED_DATA_FOLDERS[data_type]
        
        image_transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
        
        """
        dataset contains all 3 folders, clean, dirty, chan-vese
        Images are stored as a list of 3000 tuples, index [0] is tensor version of image,
        index [1] is the folder the image belongs to, with value
        0: chan-vese
        1: clean
        2: dirty
        """
        good_dataset = datasets.ImageFolder(
            os.path.join(GENERATED_DATA_PATH[train_or_eval], path_folder),
            transform=image_transform,
        )

        return torch.utils.data.DataLoader(
            good_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory = True,
        drop_last = True
        )

    except KeyError:
        raise RuntimeError(
            "Invalid 'data_type' value. Please use one of {}".format(
                list(GENERATED_DATA_FOLDERS.keys())
            )
        )

