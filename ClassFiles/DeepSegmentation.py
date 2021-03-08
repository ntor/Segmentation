#!/usr/bin/env python3

# We want to implement
# - the data_fitting term without penalty
# - gradient descent routine with clipping and c1/c2-updating

import numpy as np
import matplotlib.pyplot as plt
import ClassFiles.ChanVese as cv
import torch
from tqdm import tqdm


class DeepSegmentation:
    def __init__(
        self, image, regulariser, u_init=None, segmentation_threshold=0.5, c=None
    ):
        self._image_arr = torch.Tensor(np.array(image, dtype=float) / 255)
        self.image_shape = self._image_arr.shape
        self.channels = len(image.getbands())
        if self.channels > 1:
            self.image_shape = self.image_shape[:-1]
        self._dim = len(self.image_shape)
        self.segmentation_threshold = segmentation_threshold
        self.c = (0, 1) if c is None else c

        self.regulariser = regulariser

        if u_init is None:
            self.u = torch.rand(size=self.image_shape)
        else:
            self.u = torch.Tensor(u_init)
            self.c = cv.get_segmentation_mean_colours(
                self.u, self._image_arr, self.segmentation_threshold
            )

    def show_segmentation(self):
        """Plots and shows the image with its segmentation contour superimposed."""
        plt.imshow(self._image_arr.numpy(), cmap="gray", vmin=0, vmax=1)
        plt.contour(
            np.clip(self.u.numpy(), self.segmentation_threshold, 1), [0], colors="red"
        )
        plt.show()

    def update_c(self):
        """Update the average colours in the segmentation domain and its complement. See
        'get_segmentation_mean_colours' for more information.
        """
        self.c = cv.get_segmentation_mean_colours(
            self.u, self._image_arr, self.segmentation_threshold
        )

    def single_step(self, lmb_reg=1, epsilon=0.1):
        """Performs a single gradient descent step for 'self.u' along the CEN
        data-fitting term plus the regulariser term. After the gradient step, u
        is clipped in order to lie in [0,1].

        Parameters:

        'lmb_reg' (float): Weight for the regulariser.

        'epsilon' (float): Step size for gradient descent.
        """

        self.u.requires_grad = True
        data_fitting = cv.CEN_data_fitting_energy(
            self.u, self.c[0], self.c[1], self._image_arr
        )
        error = data_fitting + lmb_reg * self.regulariser(self.u.unsqueeze(0).unsqueeze(0))
        gradients = torch.autograd.grad(error, self.u)[0]
        self.u = (self.u - epsilon * gradients).detach()
        self.u = torch.clamp(self.u, min=0.0, max=1.0)

    def run(self, steps, lmb_reg=1, epsilon=0.1, show_iterations=False):
        """Runs 'steps' iteration of 'single_step' with same parameters 'lmb_reg' and
        'epsilon', displaying the iterations with a loading bar if
        'show_iterations' is True.

        """
        step_range = range(steps)
        if show_iterations:
            step_range = tqdm(step_range)
            for i in step_range:
                self.single_step(lmb_reg, epsilon)
                self.update_c()
