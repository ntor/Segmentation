#!/usr/bin/env python3

import numpy as np
import skfmm
from scipy.ndimage import gaussian_filter, morphology


def geodesic_distance(z, marker_points, xi=0.1):
    """Return the geodesic distance from the marker points, with coordinates given
    by 'marker_points[i,0]' and 'marker_points[i,1]', in a mollified version of 'z'
    """

    beta = 1000  # weight for gradient in geodesic distance
    z_sm = gaussian_filter(z, sigma=1)  # smoothen the image
    gx, gy = np.gradient(z_sm)
    # calculate the gradients
    nab_z = np.sqrt(gx ** 2 + gy ** 2)
    # get the gradients norm

    # putting 1's into an array at the marker points' positions
    R = np.zeros(z.shape)
    for m in marker_points:
        R[m[0], m[1]] = 1

    # Euclidean distance transform of 1-R
    # --> Fill array with distance to marker points
    BW = morphology.distance_transform_edt(1 - R)

    # Normalise this to [0,1]
    D_E = BW / np.max(BW.flatten())

    # define the "barrier function"
    f = (1.0e-3) * np.ones(np.shape(D_E)) + beta * nab_z ** 2 + xi * D_E
    # normalize to [0,1]
    f = (f - np.min(f.flatten())) / (np.max(f.flatten()) - np.min(f.flatten()))
    f = f + 0.01
    f_inverse = (1.0 / f) + 0.01  # invert f to bring everything into eikonal form

    # use the 'fast marching' method (FMM) to solve the eikonal equation
    T = skfmm.travel_time(
        R - 0.5 * np.ones(np.shape(R)),
        speed=f_inverse,
        dx=1.0 / np.shape(R)[0],
        order=1,
    )

    return T
