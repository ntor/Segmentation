#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm  # implements a "status bar" for iterations

def divergence(f):
    """Computes the divergence of the vector field f.

    Parameters:

    'f' (ndarray): array of shape (L1,...,Ld,d) representing a discretised
    vector field on a d-dimensional lattice

    Returns: ndarray of shape (L1,...,Ld)
    """

    num_dims = len(f.shape) - 1
    return np.ufunc.reduce(
        np.add, [np.gradient(f[..., i], axis=i) for i in range(num_dims)]
    )


def cv_energy(u, c1, c2, lambd, image_arr):
    """Calculates the "convexified Chan-Vese" functional.

    Parameters:

    'u' (ndarray): 2D array of values in [0,1], representing the ("loosened")
    characteristic function of the segmentation domain.

    'c1', 'c2' (uint8): approximate average pixel value of 'image_arr' inside,
    resp. outside, the segmentation domain

    'lambd' (float): positive parameter in front of the "data fitting term"

    'image_arr' (ndarray dtype:uint8): original image, which we want to segment
    """

    TV_energy = np.sum(
        np.apply_along_axis(np.linalg.norm, -1, np.stack(np.gradient(u), axis=-1))
    )
    data_fitting = np.sum((image_arr - c1) ** 2 * u + (image_arr - c2) ** 2 * (1 - u))
    return TV_energy + lambd * data_fitting


def get_segmentation_mean_colours(u, image_arr, threshold=0.5):
    """Returns the "mean colors" of 'image_arr' inside and outside the segmentation
    domain, which is defined by {u > threshold} (cf. equation (11) in
    "Algorithms for finding global minimizers ..." by Chan et al., 2006)

    Parameters:

    'u' (ndarray or Tensor):
    "loosened" characteristic function of the segmentation domain in 'image_arr'

    'image_arr' (ndarray or Tensor):
    image to calculater the segmentation average colours from

    'threshold' (float):
    in [0,1], used the determine the segmentation domain {u > threshold}
    """
    mask = u > threshold
    c1 = (u * image_arr)[mask].mean()
    c2 = (u * image_arr)[~mask].mean()

    return c1, c2


class ChanVese:
    def __init__(self, image_arr, c1=None, c2=None):
        tmp_shape = image_arr.shape()
        self.image_shape, self.channels = tmp_shape[:-1], tmp_shape[-1]
