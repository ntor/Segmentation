#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# This file should collect methods for visualisation of segmentations and their
# evolution during the optimisation process.


def show_segmentation(image, u, segmentation_threshold=0.5):
    """Shows the level set contour of 'u' at level 'segmentation_threshold' on top of
    'image'.

    Parameters:

    'image' (Image): Underlay image, given as an instance of the
    'Image' class from the Pillow library.

    'u' (ndarray): Segmentation function with values in [0,1], given as a 2D numpy
    array with the same shape as 'image'.

    """
    plt.imshow(image, cmap='gray')
    plt.contour(np.clip(u, segmentation_threshold, 1), [0], colors="red")
    plt.show()
