"""
This file provides a class interface to generate 2D images of random
geometric shapes and to corrupt them with different types of noise.
"""

import math
import random
import numpy as np
from skimage.util import random_noise
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

class ShapeGenerator:
    "A wrapper class to generate random geometric shapes "

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.image = Image.new("L", (width, height), np.random.randint(150, 200))
        self._canvas = ImageDraw.Draw(self.image)

    def add_polygon(self, size=0, times=1,colour=0):
        r = int(random.random()*(255-colour)/10)
        if size == 0:
            # If the user does not provide an explicit size for the polygon,
            # then take an eigth of the minimal dimensions of the image (as an
            # arbitrary scale).
            size = min(self.width, self.height) / 8

        for i in range(times):
            x_center = random.uniform(0, self.width)
            y_center = random.uniform(0, self.height)
            verts = generate_polygon_coords(
                x_center,
                y_center,
                aveRadius=size,
                irregularity=0.35,
                spikeyness=0.2,
                numVerts=10,
            )
            self._canvas.polygon(verts, outline=colour+r, fill=colour+r)

    def add_ellipse(self, size=0, times=1):
        r = int(25*random.random())
        if size == 0:
            # If the user does not provide an explicit size for the polygon,
            # then take an eigth of the minimal dimensions of the image (as an
            # arbitrary scale).
            size = min(self.width, self.height) / 8

        for i in range(times):
            x_center = random.uniform(self.width/4, 3*self.width/4)
            y_center = random.uniform(self.height/4, 3*self.height/4)

            lowerx = x_center - np.random.exponential(size)/2-size/2
            lowery = y_center - np.random.exponential(size)/2-size/2
            upperx = x_center + np.random.exponential(size)/2+size/2
            uppery = y_center + np.random.exponential(size)/2+size/2
            # Imagedarw then draws the ellipse
            shape = [(lowerx, lowery), (upperx, uppery)]
            self._canvas.ellipse(shape, outline=0+r, fill=0+r)
    
    def add_holes(self,numholes=40, width=2):
            for i in range(numholes):
                tint = np.random.randint(150,255)
                self.add_polygon(size=width, times=1, colour=tint)
    
    def add_noise(self,sig=2):
        im_arr = np.asarray(self.image)
        #adding a gaussian filter will blur the image
        #chunk_img = chunks(im_arr,n,m,lilrad=5,numchunk=50,colour=(255,255,255))
        blur_img  = gaussian_filter(im_arr, sigma=sig)
        # random_noise() method will convert image in [0, 255] to [0, 1.0],
        # inherently it use np.random.normal() to create normal distribution
        # and adds the generated noised back to image
        noise_img = random_noise(blur_img, mode="gaussian", var=0.05)
        noise_img = (255 * noise_img).astype(np.uint8)
        self.image = Image.fromarray(noise_img)
        
     
    def add_blur(self,sig=2):
        im_arr = np.asarray(self.image)
        #adding a gaussian filter will blur the image
        #chunk_img = chunks(im_arr,n,m,lilrad=5,numchunk=50,colour=(255,255,255))
        blur_img  = gaussian_filter(im_arr, sigma=sig)
        # random_noise() method will convert image in [0, 255] to [0, 1.0],
        # inherently it use np.random.normal() to create normal distribution
        # and adds the generated noised back to image
        #noise_img = random_noise(blur_img, mode="gaussian", var=0.05)
        #noise_img = (255 * noise_img).astype(np.uint8)
        self.image = Image.fromarray(blur_img)




def generate_polygon_coords(ctrX, ctrY, aveRadius, irregularity, spikeyness, numVerts):
    """Start with the centre of the polygon at ctrX, ctrY,
    then creates the polygon by sampling points on a circle around the centre.
    Randon noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    ctrX, ctrY - coordinates of the "centre" of the polygon

    aveRadi - in px, the average radius of this polygon, this roughly controls how large the
    polygon is, really only useful for order of magnitude.

    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices.
    [0,1] will map to [0, 2pi/numberOfVerts]

    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius.
    [0,1] will map to [0, aveRadius]

    numVert - self-explanatory

    Returns a list of vertices, in CCW order.

    """

    def clip(x, min, max):
        if min > max:
            return x
        elif x < min:
            return min
        elif x > max:
            return max
        else:
            return x

    irregularity = clip(irregularity, 0, 1) * 2 * math.pi / numVerts
    spikeyness = clip(spikeyness, 0, 1) * aveRadius

    # generate n angle steps
    angleSteps = []
    lower = (2 * math.pi / numVerts) - irregularity
    upper = (2 * math.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts):
        tmp = random.uniform(lower, upper)
        angleSteps.append(tmp)
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2 * math.pi)
    for i in range(numVerts):
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(numVerts):
        r_i = clip(random.gauss(aveRadius, spikeyness), 0, 2 * aveRadius)
        x = ctrX + r_i * math.cos(angle)
        y = ctrY + r_i * math.sin(angle)
        points.append((int(x), int(y)))

        angle = angle + angleSteps[i]

    return points



