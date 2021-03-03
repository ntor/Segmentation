#!/usr/bin/env python3

import matplotlib.pyplot as plt

# import torch
# from PIL import Image
import numpy as np
from tqdm import tqdm  # implements a "status bar" for iterations


class ChanVese:
    def __init__(self, image, segmentation_threshold=0.5, c=None):
        self._image_arr = np.array(image, dtype=float) / 255
        self.image_shape = self._image_arr.shape
        self.channels = len(image.getbands())
        if self.channels > 1:
            self.image_shape = self.image_shape[:-1]
        self._dim = len(self.image_shape)
        self.segmentation_threshold = segmentation_threshold
        if c is None:
            self.c = (0, 1)
        else:
            self.c = c

        self.u = np.random.random(self.image_shape)
        self._u_interm = np.array(self.u)
        self._z = (np.random.random(self.image_shape + (self._dim,)) - 0.5) / self._dim

    def single_step(self, lmb=0.5, epsilon=0.1, theta=0.2):
        """Update 'self.u', as well as the helper functions 'self._u_interm' and
        'self._z', according to a 'primal-dual' algorithm (Chambolla-Pock, 2011)
        for minimisation of the Chan-Esedoglu-Nikolova functional.

        Consult the last part of
        https://www.math.u-bordeaux.fr/~npapadak/TP/TP2.pdf for more
        information, especially on the role of z, which plays the role of a test function.

        Parameters:

        'lmb': parameter for the "data-fitting" term in the Chan-Esedoglu-Nikolova functional.

        'epsilon': step size for both the 'u' and 'z' gradient steps

        'theta':

        """

        self._z = clip_vector_field(
            self._z + epsilon * np.stack(np.gradient(self._u_interm), axis=-1)
        )
        tmp = lmb * (
            (self._image_arr - self.c[0]) ** 2 - (self._image_arr - self.c[1]) ** 2
        )
        u_update = np.clip(self.u + epsilon * (divergence(self._z) - tmp), 0, 1)
        self._u_interm = (1 + theta) * u_update - theta * self.u
        self.u = u_update

    def update_c(self):
        """Update the average colours in the segmentation domain and its complement. See
        'get_segmentation_mean_colours' for more information.

        Parameters:

        'segmentation_threshold' (float): in [0,1], used to determine the
        segmentation boundary as the level set of 'self.u' at this level
        """
        self.c = get_segmentation_mean_colours(
            self.u, self._image_arr, self.segmentation_threshold
        )

    # TODO Modify to also work with coloured images.
    def show_segmentation(self):
        """Plots and shows the image with its segmentation contour superimposed."""
        plt.imshow(self._image_arr, cmap="gray", vmin=0, vmax=1)
        plt.contour(np.clip(self.u, self.segmentation_threshold, 1), [0], colors="red")
        plt.show()

    # TODO Implement way to stop according to energy stabilisation.
    def run(
        self,
        steps,
        lmb=0.5,
        epsilon=0.1,
        theta=0.2,
        update_c_interval=20,
        show_iterations=False,
    ):
        step_range = range(steps)
        if show_iterations:
            step_range = tqdm(step_range)

        for i in step_range:
            self.single_step(lmb, epsilon, theta)
            if (i + 1) % update_c_interval == 0:
                self.update_c()
                print(
                    "Energy: {}".format(
                        CEN_energy(self.u, self.c[0], self.c[1], lmb, self._image_arr)
                    )
                )


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


def CEN_energy(u, c1, c2, lambd, image_arr):
    """Calculates the Chan-Esedoglu-Nikolova functional.
    (cf. Theorem 2 in "Algorithms for finding global minimizers ..." by Chan et al., 2006)

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


def clip_vector_field(z, threshold=1):
    """Truncate the vector field 'z' to have at most pointwise norm of value
    'threshold'. If z[i1,..,id] has norm ≤ 1, we do nothing, otherwise we
    replace it by z[i1,...,id]/norm(z[i1,...,id])

    Parameters:

    'z' (ndarray): vector field of shape (N1,...,Nd,k)

    """

    def criterion(v):
        norm = np.linalg.norm(v)
        return ((v / norm) if norm > threshold else v)

    return np.apply_along_axis(criterion, -1, z)
    # return z / ((1 + np.maximum(0, np.apply_along_axis(np.linalg.norm, -1, z) - 1))[...,np.newaxis])
