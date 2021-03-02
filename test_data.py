#!/usr/bin/env python3

import torch
import torchvision
import matplotlib.pyplot as plt
# import numpy as np

# This file is for browsing random samples of the artificially generated data.
# We assume that the images lie in "./data/generated/[clean,dirty,chan-vese]".

# Below we initialise a dataloader for the "clean data", loading the images into
# tensors in minibatches.

image_transform = torchvision.transforms.ToTensor()

good_dataset = torchvision.datasets.ImageFolder("./data/generated/clean/", transform=image_transform)
good_dataloader = torch.utils.data.DataLoader(good_dataset, batch_size=4, shuffle=True)

good_dataiter = iter(good_dataloader)

for i in range(10):
    sample_image = good_dataiter.next()
    plt.imshow(sample_image.numpy())
    plt.show()
