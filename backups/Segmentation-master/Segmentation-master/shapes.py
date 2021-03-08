#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 20:53:23 2021

@author: jamesmason
"""


"""
for image size nxm, cleanim(n,m) puts a random black shape on a white background.

roughup(im,n,m) will take out some chunks and add noise.


to show example images run: 
    
n = 500
m = 500
im = cleanim(n,m)
im.show()
rim = roughup(im,n,m)
rim.show()



parameters for testing: 
rad = 100
lilrad =5
nverts  =16
numchunk = 50
black = (0,0,0)
white = =(255,255,255)
"""


import math, random
import numpy as np
from skimage.util import random_noise
from PIL import Image, ImageDraw


def clip(x, min, max) :
    if( min > max ) :  return x    
    elif( x < min ) :  return min
    elif( x > max ) :  return max
    else :             return x
    
    
    
def generatePolygon( ctrX, ctrY, aveRadius, irregularity, spikeyness, numVerts) :
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

    irregularity = clip( irregularity, 0,1 ) * 2*math.pi /numVerts
    spikeyness = clip( spikeyness, 0,1 ) * aveRadius

    # generate n angle steps
    angleSteps = []
    lower = (2*math.pi / numVerts) - irregularity
    upper = (2*math.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts) :
        tmp = random.uniform(lower, upper)
        angleSteps.append( tmp )
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2*math.pi)
    for i in range(numVerts) :
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2*math.pi)
    for i in range(numVerts) :
        r_i = clip( random.gauss(aveRadius, spikeyness), 0, 2*aveRadius )
        x = ctrX + r_i*math.cos(angle)
        y = ctrY + r_i*math.sin(angle)
        points.append( (int(x),int(y)) )

        angle = angle + angleSteps[i]

    return points



def blank(n,m):
    #this creates a blank canvas
    white=(255,255,255)
    im = Image.new('RGB', (n, m), white)
    return im


def polygonz(im, n, m, rad , nverts,colour) :
    #generate polgon generates the verticies of a ploygon then the ImageDraw package fills this in on the Image
    verts = generatePolygon( ctrX = n/2, ctrY= m/2, aveRadius=rad, irregularity=0.35, spikeyness=0.2, numVerts=nverts)
    draw = ImageDraw.Draw(im)
    draw.polygon(verts,outline=colour, fill=colour)
    return im 


def ellipze(im, n, m, rad) :
    ctrX = n/2
    ctrY= m/2
    # from the centre point, crtX ctrY, this adds/subtracks exponential parameters to form a box the ellipse lies inside
    lowerx = ctrX - np.random.exponential(rad)
    lowery = ctrY - np.random.exponential(rad)
    upperx = ctrX + np.random.exponential(rad)
    uppery = ctrY + np.random.exponential(rad)
    #Imagedarw then draws the ellipse
    shape = [(lowerx, lowery), (upperx , uppery)]
    draw = ImageDraw.Draw(im)
    black = (0,0,0)
    draw.ellipse(shape, outline=black, fill=black)
    return im 



# convert PIL Image to ndarray
def noize(im):
    im_arr = np.asarray(im)
    #random_noise() method will convert image in [0, 255] to [0, 1.0],
    #inherently it use np.random.normal() to create normal distribution
    #and adds the generated noised back to image
    noise_img = random_noise(im_arr[:,:,1], mode='gaussian', var=0.05)
    noise_img = np.repeat(noise_img[:, :, np.newaxis], 3, axis=2)
    noise_img = (255*noise_img).astype(np.uint8)
    img = Image.fromarray(noise_img)
    return img


def chunks(im,n,m,lilrad,numchunk,colour):
    #this fills in numchunk small polygons 
    for i in range(numchunk):
        X = random.uniform(0, n)
        Y = random.uniform(0, m)
        im = polygonz(im, 2*X, 2*Y, lilrad , 10,colour)
    return im
    
    
def cleanim(n,m):
    im = blank(n,m)
    k = random.randint(1,3) #number os shapes drawn
    print(k)
    for i in range(k):
        j = random.randint(1,2) #ellipse or polygon? 
        if j == 1:
            im = polygonz(im, n, m, 100 , 16, (0,0,0))
        else:
            im =  ellipze(im, n, m, rad=100)
    return im

def roughup(im,n,m):
    rim = chunks(im,n,m,lilrad=5,numchunk=50,colour=(255,255,255))
    rim = noize(im)
    return rim
     

     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
