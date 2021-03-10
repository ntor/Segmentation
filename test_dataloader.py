#!/usr/bin/env python3

import os
import ClassFiles.GeneratedDataset as dat
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

dirty_image_dataset = dat.ImageDataset("./data/train/")
image_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)]
)
dirty_image_dataloader = DataLoader(dirty_image_dataset, batch_size=1)
dirty_image_iter = iter(dirty_image_dataloader)

dirty_seg_dataset = dat.SegmentationDataset("./data/train/", seg_type="clean")
dirty_seg_dataloader = DataLoader(dirty_seg_dataset)
dirty_seg_iter = iter(dirty_seg_dataloader)

for im_tensor_batch, seg in zip(dirty_image_iter, dirty_seg_iter):
    plt.imshow(im_tensor_batch[0][0], cmap="gray")
    plt.contour(np.clip(seg.squeeze(0).numpy(), 0.5, 1), [0], colors="red")
    plt.show()
