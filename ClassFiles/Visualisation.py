#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


def show_segmentation(image, u, segmentation_threshold=0.5):
    plt.imshow(image, cmap='gray')
    plt.contour(np.clip(u, segmentation_threshold, 1), [0], colors="red")
    plt.show()
