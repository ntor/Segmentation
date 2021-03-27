#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # implements a progress bar for iterations
from scipy import signal

# import torch
# from PIL import Image


class ChanVese:
    def __init__(self, image, segmentation_threshold=0.8, c=None):
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

        self.u = self._image_arr
        self._u_interm = np.array(self.u)
        self._z = (np.random.random((self._dim,) + self.image_shape) - 0.5) / self._dim

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
            self._z + epsilon * gradient(self._u_interm)
        )
        tmp = lmb * (
            (self._image_arr - self.c[0]) ** 2 - (self._image_arr - self.c[1]) ** 2
        )
        u_update = np.clip(self.u + epsilon * (div(self._z) - tmp), 0, 1)
        self._u_interm = (1 + theta) * u_update - theta * self.u
        self.u = u_update

    def update_c(self):
        """Update the average colours in the segmentation domain and its complement. See
        'get_segmentation_mean_colours' for more information.
        """
        try:
            self.c = get_segmentation_mean_colours(
                self.u, self._image_arr, self.segmentation_threshold
            )
        except RuntimeError:
            self.c = tuple(np.random.rand(2))

    def show_segmentation(self):
        """Plots and shows the image with its segmentation contour superimposed."""
        plt.imshow(self._image_arr, cmap="gray", vmin=0, vmax=1)
        plt.contour(np.clip(self.u, self.segmentation_threshold, 1), [0], colors="red")
        plt.show()

    def run(
        self,
        steps,
        lmb=1.0,
        epsilon=0.05,
        theta=0.2,
        update_c_interval=5,
        show_iterations=False,
        show_energy=False,
    ):
        step_range = range(steps)
        if show_iterations:
            step_range = tqdm(step_range)

        for i in step_range:
            self.single_step(lmb, epsilon, theta)
            if (i + 1) % update_c_interval == 0:
                self.update_c()
                if show_energy:
                    print(
                        "Energy: {}".format(
                            CEN_energy(
                                self.u, self.c[0], self.c[1], lmb, self._image_arr
                            )
                        )
                    )

    def run_until_stable(
        self,
        lmb=0.5,
        epsilon=0.1,
        theta=0.2,
        update_c_interval=1,
        energy_sample_interval=10,
        energy_sample_length=5,
        stability_tolerance=1e-6,
        print_fluctuation=False,
        print_total_steps=False,
    ):
        has_stabilised = False
        energy_sample_list = []
        i = 0
        while not has_stabilised:
            self.single_step(lmb, epsilon, theta)
            if (i + 1) % update_c_interval == 0:
                self.update_c()
            if (i + 1) % energy_sample_interval == 0:
                energy_sample_list.append(
                    CEN_energy(self.u, self.c[0], self.c[1], lmb, self._image_arr)
                )
                if len(energy_sample_list) % energy_sample_length == 0:
                    fluctuations = [
                        x / y - 1
                        for x in energy_sample_list
                        for y in energy_sample_list
                    ]

                    mean_fluctuation = sum(fluctuations) / len(fluctuations)

                    if print_fluctuation:
                        print("Fluctuation: {}".format(mean_fluctuation))
                    if abs(mean_fluctuation) < stability_tolerance:
                        has_stabilised = True
                    else:
                        energy_sample_list = []

            i = i + 1

        if print_total_steps:
            print("Total steps until stabilisation: {}".format(i))


# Obsolete divergence function. Dropped in favor of the simpler
# finite-difference version "div"

# def divergence(f):
#     """Computes the divergence of the vector field f.

#     Parameters:

#     'f' (ndarray): array of shape (L1,...,Ld,d) representing a discretised
#     vector field on a d-dimensional lattice

#     Returns: ndarray of shape (L1,...,Ld)
#     """

#     num_dims = len(f.shape) - 1
#     return np.ufunc.reduce(
#         np.add, [np.gradient(f[..., i], axis=i) for i in range(num_dims)]
#     )


def div(grad):
    '''
    Compute the divergence of a gradient
    Courtesy : E. Gouillart - https://github.com/emmanuelle/tomo-tv/
    '''
    res = np.zeros(grad.shape[1:])
    for d in range(grad.shape[0]):
        this_grad = np.rollaxis(grad[d], d)
        this_res = np.rollaxis(res, d)
        this_res[:-1] += this_grad[:-1]
        this_res[1:-1] -= this_grad[:-2]
        this_res[-1] -= this_grad[-2]
    return res


def gradient(img):
    '''
    Compute the gradient of an image as a numpy array
    Courtesy : E. Gouillart - https://github.com/emmanuelle/tomo-tv/
    '''
    shape = [img.ndim, ] + list(img.shape)
    gradient = np.zeros(shape, dtype=img.dtype)
    slice_all = [0, slice(None, -1),]
    for d in range(img.ndim):
        gradient[slice_all] = np.diff(img, axis=d)
        slice_all[0] = d + 1
        slice_all.insert(1, slice(None))
    return gradient


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
    data_fitting = CEN_data_fitting_energy(u, c1, c2, image_arr)
    return TV_energy + lambd * data_fitting


def CEN_data_fitting_energy(u, c1, c2, image_arr):
    """Returns the data-fitting term, measuring how well 'u' segments 'image_arr'
    into regions of colour 'c1' and 'c2'. For more information on the parameters
    take a look at 'CEN_energy'.

    This function is compatible both with numpy's ndarrays and
    torch's Tensor classes.

    """
    return ((image_arr - c1) ** 2 * u + (image_arr - c2) ** 2 * (1 - u)).sum()


def CEN_data_fitting_L2gradient(u, c1, c2, image_arr):
    """Returns the L2 norm of the gradient of the data-fitting term with respect to u,
    used for initialising the reconstruction lambda
    
    This function is compatibile both with numpy's ndarrays and torch's Tensor classes
    
    """
    return (((image_arr - c1) ** 2 - (image_arr - c2) ** 2) ** 2).sum() ** 0.5


def initialise_reconstruction_lambda(u_groundtruth, image_arr, NN_L2gradient=1, lambda_CEN=0.5, threshold=0.5):
    """Return reconstruction_lambda for which u_groundtruth is a critical point of
    DeepSegmentation applied to image_arr.
    
    If the gradient loss of the NN is g, the NN_L2gradient is 1 + g ** 0.5.
    
    lambda_CEN is the lambda used to weight the datafitting term in ChanVese.single_step.
    
    threshold is the threshold used to calculate c1, c2 in ChanVese.update_c.
    
    To properly initialise lambda, this output must be averaged over a suitably large
    batch of images/
    
    """
    c1, c2 = get_segmentation_mean_colours(u_groundtruth, image_arr, threshold = threshold)
    return lambda_CEN * CEN_data_fitting_L2gradient(u_groundtruth, c1, c2, image_arr) / NN_L2gradient


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
    above = (u * image_arr)[mask]
    below = (u * image_arr)[~mask]

    if above.size == 0 or below.size == 0:
        raise RuntimeError("Empty segmentation domain. Cannot calculate mean.")
    else:
        c1 = above.mean()
        c2 = below.mean()

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
        return (v / norm) if norm > threshold else v

    return np.apply_along_axis(criterion, -1, z)
    # return z / ((1 + np.maximum(0, np.apply_along_axis(np.linalg.norm, -1, z) - 1))[...,np.newaxis])
