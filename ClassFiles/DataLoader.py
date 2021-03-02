#!/usr/bin/env python3

import torch
import torchvision
import os

# Path constants used to load the generated images. See
# 'get_generated_dataloader' for more information.
GENERATED_DATA_PATH = "./data/generated/"
GENERATED_DATA_FOLDERS = {
    "clean": "clean/",
    "dirty": "dirty/",
    "chan-vese": "chan-vese/",
}


def get_generated_dataloader(data_type, batch_size=4, shuffle=True):
    """Returns a torch.utils.data.DataLoader object for iterating over the generated
    data. The images are chosen from one of the subfolders in
    GENERATED_DATA_PATH, according to the value corresponding to DATA_TYPE in
    the GENERATED_DATA_FOLDERS dict. The images will be loaded as tensors.

    Parameters:

    DATA_TYPE (string): one of the key values of GENERATED_DATA_FOLDERS. Used to
    choose the folder from which to load the images

    BATCH_SIZE (int): is forwarded to the parameters for the DataLoader object

    SHUFFLE (bool):  is forwarded to the parameters for the DataLoader object
    """

    try:
        path_folder = GENERATED_DATA_FOLDERS[data_type]
        image_transform = torchvision.transforms.ToTensor()

        good_dataset = torchvision.datasets.ImageFolder(
            os.path.join(GENERATED_DATA_PATH, path_folder),
            transform=image_transform,
        )

        return torch.utils.data.DataLoader(
            good_dataset, batch_size=batch_size, shuffle=shuffle
        )

    except KeyError:
        raise RuntimeError(
            "Invalid 'data_type' value. Please use one of {}".format(
                list(GENERATED_DATA_FOLDERS.keys())
            )
        )
