#!/usr/bin/env python3

# import torch
# import torchvision
import matplotlib.pyplot as plt
import ClassFiles.DataLoader as load
# import numpy as np

# This file is for browsing random samples of the artificially generated data.
# We assume that the images lie in "./data/generated/[clean,dirty,chan-vese]".


dataloader = load.get_generated_dataloader('clean')
dataiter = iter(dataloader)

for i in range(10):
    sample_image = dataiter.next()
    plt.imshow(sample_image.numpy())
    plt.show()
