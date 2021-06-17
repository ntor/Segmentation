#!/usr/bin/env python3

# This files shows how to use the "ImageDataset" and "SegmentationDataset"
# classes in order to load images and segmentation from a given folder. As an
# example we browse through the 10 first pairs of images and segmentations and
# show them on the screen.

from torch.utils.data import DataLoader
import ClassFiles.GeneratedDataset as dat
import matplotlib.pyplot as plt
import numpy as np

dirty_image_dataset = dat.ImageDataset("./data/")
dirty_image_dataloader = DataLoader(dirty_image_dataset, batch_size=1)
dirty_image_iter = iter(dirty_image_dataloader)

dirty_seg_dataset = dat.SegmentationDataset("./data/", seg_type="chan-vese")
dirty_seg_dataloader = DataLoader(dirty_seg_dataset)
dirty_seg_iter = iter(dirty_seg_dataloader)

i = 0
for im_tensor_batch, seg in zip(dirty_image_iter, dirty_seg_iter):
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    u = seg.squeeze(0).numpy()

    axs.imshow(im_tensor_batch[0][0], cmap="gray")
    axs.contour(np.clip(u, 0.5, 1), [0], colors="red")
    # axs[1].hist(u.flatten())
    plt.show()
    i += 1
    if i >= 10:
        break
