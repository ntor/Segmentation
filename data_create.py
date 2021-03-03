#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 10:33:00 2021

@author: jamesmason
"""
from ClassFiles.ShapeGenerator import ShapeGenerator
from ClassFiles.ChanVese import ChanVese


from PIL import Image
import datetime

def create(times=1,size=(128, 128), cleansaave ="images/clean/clean_", 
                            dirtysave  ="images/dirty/dirty_",
                            chansave   ="images/chan-vese/chanvese_"):
    
    for i in range(times):
        e = datetime.datetime.now().strftime("%m_%d_%H_%M_%S_%f")
        #create a clean image
        shapes = ShapeGenerator(128, 128)
        shapes.add_polygon(times=3)
        shapes.add_ellipse(times=3)
        #save in clean
        shapes.image.save(fp = cleansaave+e+".png", format = 'PNG')
        #add noise
        shapes.add_holes(40)
        shapes.add_noise()
        #save in dirty 
        shapes.image.save(fp = dirtysave+e+".png", format = 'PNG')
        #apply chan-vese
        shapes = ChanVese(shapes.image)
        shapes.run(steps = 400,show_iterations=False)
        #save in chan-vese
        im = Image.fromarray(shapes._image_arr).convert("L")
        im.save(fp =  chansave+e+".png", format = 'PNG')
    
    





