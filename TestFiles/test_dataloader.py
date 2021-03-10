#!/usr/bin/env python3

import os
import ClassFiles.GeneratedDataset as dat
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dirty_image_dataset = dat.ImageDataset("./data/train/")
image_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)]
)
dirty_image_dataloader = DataLoader(
    dirty_image_dataset, batch_size=1
)

dirty_image_iter = iter(dirty_image_dataloader)

for im_array_batch in dirty_image_iter:
    plt.imshow(im_array_batch[0][0])
    plt.show()
