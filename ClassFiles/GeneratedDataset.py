#!/usr/bin/env python3

import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from ClassFiles.ShapeGenerator import ShapeGenerator
from ClassFiles.ChanVese import ChanVese
from tqdm import tqdm
import ClassFiles.EvaluationMetrics as EM
import copy
import math

# This file implements two torch Dataset classes, 'ImageDataset' and
# 'SegmentationDataset', and a function 'generate_data'. The datasets are
# initialised with a path to a directory in which we assume a folder structure
# as follows:
#     + image_0
#      - clean.png
#      - dirty.png
#      - clean_seg.npy
#      - dirty_cv_seg.npy
#    + image_1
#      - clean.png
#      - ...
#
# The dataloaders then provide an interface for loading, for example the clean
# dirty images, or their segmentations.
#
# The actual names are not hardcoded but can be adjusted in the constants below.
# The initialised Dataset instances are not to be used directly but rather in
# order to initialise a torch DataLoader object.
#
# The 'generate_data' function can be used in order to populate a folder with
# artifical data as described above.


SAMPLE_FOLDER_PREFIX = "image_"

IMAGE_TYPE_NAMES = {
    "dirty": "dirty.png",
    "clean": "clean.png",
    "chan-vese": "dirty_cv_seg.png",
}

SEGMENTATION_TYPE_NAMES = {
    "clean": "clean_seg.npy",
    "dirty": "dirty_seg.npy",
    "chan-vese": "dirty_cv_seg.npy",
    "deep-segmentation": "dirty_ds_seg.npy",
}


class ImageDataset(Dataset):
    def __init__(self, data_root, image_type="dirty"):
        self.data_root = data_root
        self.image_type = image_type
        if not os.path.isdir(data_root):
            print("ERROR: data_root is not a valid directory")

    def __len__(self):
        root_list = os.listdir(self.data_root)
        image_folders = [s for s in root_list if s.startswith(SAMPLE_FOLDER_PREFIX)]
        return len(image_folders)

    def __getitem__(self, idx):
        im = Image.open(
            os.path.join(
                self.data_root,
                "image_{}".format(idx),
                IMAGE_TYPE_NAMES[self.image_type],
            )
        )
        return transforms.ToTensor()(im)


class SegmentationDataset(Dataset):
    def __init__(self, data_root, seg_type="chan-vese"):
        self.data_root = data_root
        self.seg_type = seg_type
        if not os.path.isdir(data_root):
            print("ERROR: data_root is not a valid directory")

    def __len__(self):
        root_list = os.listdir(self.data_root)
        image_folders = [s for s in root_list if s.startswith(SAMPLE_FOLDER_PREFIX)]
        return len(image_folders)

    def __getitem__(self, idx):
        seg = np.load(
            os.path.join(
                self.data_root,
                SAMPLE_FOLDER_PREFIX + "{}".format(idx),
                SEGMENTATION_TYPE_NAMES[self.seg_type],
            )
        )
        return torch.Tensor(seg)


def generate_data(times, root_dir, size=(128, 128), append=True):
    if append:
        start_index = sum(
            1 for s in os.listdir(root_dir) if s.startswith(SAMPLE_FOLDER_PREFIX)
        )
    else:
        start_index = 0

    for i in tqdm(range(times)):
        sample_folder = os.path.join(
            root_dir, SAMPLE_FOLDER_PREFIX + "{}".format(i + start_index)
        )
        try:
            os.mkdir(sample_folder)
        except FileExistsError:
            pass

        shapes = ShapeGenerator(128, 128)
        shapes.add_polygon(times=np.random.randint(10, 35))
        shapes.add_ellipse(times=np.random.randint(10, 35))
        shapes.add_holes(
            numholes=np.random.randint(5, 20), width=np.random.randint(5, 20)
        )

        shapes.image.save(
            fp=os.path.join(sample_folder, IMAGE_TYPE_NAMES["clean"]), format="PNG"
        )
        np.save(
            file=os.path.join(sample_folder, SEGMENTATION_TYPE_NAMES["clean"]),
            arr=np.array(shapes.image) / 255,
        )
        shapes.add_noise()
        shapes.image.save(
            fp=os.path.join(sample_folder, IMAGE_TYPE_NAMES["dirty"]), format="PNG"
        )

        shapes = ChanVese(shapes.image)
        shapes.run(steps=500, show_iterations=False)
        # save in chan-vese
        np.save(
            file=os.path.join(sample_folder, SEGMENTATION_TYPE_NAMES["chan-vese"]),
            arr=shapes.u,
        )


def generate_data_lunglike(times, root_dir, size=(128, 128), append=True):
    if append:
        start_index = sum(
            1 for s in os.listdir(root_dir) if s.startswith(SAMPLE_FOLDER_PREFIX)
        )
    else:
        start_index = 0

    for i in tqdm(range(times)):
        sample_folder = os.path.join(
            root_dir, SAMPLE_FOLDER_PREFIX + "{}".format(i + start_index)
        )
        try:
            os.mkdir(sample_folder)
        except FileExistsError:
            pass

        # if the cv is good enough then it will save if not it goes again
        evaluation = 0
        while evaluation < 0.8:

            shapes = ShapeGenerator(128, 128)
            shapes.add_ellipse(times=np.random.randint(1, 3), size=0.2 * 128)

            cleanimage = shapes.image.copy()
            
            #need to trun the grey btis white for the clean segmentation 
            datas = cleanimage.getdata()
            new_image_data = []
            for item in datas:
                # change all grey pixels to white
                if item in list(range(50,255)):
                    new_image_data.append(255)
                else:
                    new_image_data.append(item)

            # update image data
            cleanimage.putdata(new_image_data)
            clean_seg = np.array(cleanimage)
            # now all white :) 
            
            #add some grey/white holes 
            shapes.add_holes(
                numholes=np.random.randint(40, 50),
                width=np.random.randint(3, 4),
            )
            #add blur
            shapes.add_blur(sig=1.5)
            
            cvshapes = ChanVese(shapes.image)
            cvshapes.run(steps=500, show_iterations=False)

            # they are meant to be reshaped inside Jaccard but python is
            # ignoring that for some reason so Im doing it here :))))
            u1 = np.reshape(cvshapes.u, np.size(cvshapes.u))
            u2 = np.reshape(clean_seg, np.size(clean_seg))
            evaluation = EM.Jaccard(u1, u2)

        # save all the images
        cleanimage.save(
            fp=os.path.join(sample_folder, IMAGE_TYPE_NAMES["clean"]), format="PNG"
        )
        np.save(
            file=os.path.join(sample_folder, SEGMENTATION_TYPE_NAMES["clean"]),
            arr=np.array(cleanimage) / 255,
        )

        shapes.image.save(
            fp=os.path.join(sample_folder, IMAGE_TYPE_NAMES["dirty"]), format="PNG"
        )

        np.save(
            file=os.path.join(sample_folder, SEGMENTATION_TYPE_NAMES["chan-vese"]),
            arr=cvshapes.u,
        )

        cvim = Image.fromarray(255 * cvshapes.u).convert("L")
        cvim.save(
            fp=os.path.join(sample_folder, IMAGE_TYPE_NAMES["chan-vese"]), format="PNG"
        )
        
        
def generate_data_lunglike_tagged(times, root_dir, size=(128, 128), append=True):
    if append:
        start_index = sum(
            1 for s in os.listdir(root_dir) if s.startswith(SAMPLE_FOLDER_PREFIX)
        )
    else:
        start_index = 0

    for i in tqdm(range(times)):
        sample_folder = os.path.join(
            root_dir, SAMPLE_FOLDER_PREFIX + "{}".format(i + start_index)
        )
        try:
            os.mkdir(sample_folder)
        except FileExistsError:
            pass

        # if the cv is good enough then it will save if not it goes again
        evaluation = 0
        while evaluation < 0.8:
            
            r = np.random.choice([0,1])
            shapes = ShapeGenerator(128, 128)
            if r ==0:
                #shapes.add_smallcorner_ellipse()
                centre = shapes.add_side_ellipse()
                cleanimage = shapes.image.copy()
                
                #add some extra ellispses but not ontop of the tagged one
                for i in range(np.random.choice([1,2])):
                    theta = np.random.choice([90,180])
                    shapes.rotation(angle=theta)
                    shapes.add_smallcorner_ellipse()
                    shapes.rotation(angle=-theta)
            else:
                centre = shapes.add_smallcorner_ellipse()
                #shapes.add_side_ellipse()
                cleanimage = shapes.image.copy()
                #add some extra ellispses but not ontop of the tagged one
                for i in range(np.random.choice([1,2,3])):
                    theta = np.random.choice([90,180,270])
                    shapes.rotation(angle=theta)
                    shapes.add_smallcorner_ellipse()
                    shapes.rotation(angle=-theta)
                    
            #need to trun the grey btis white for the clean segmentation 
            datas = cleanimage.getdata()
            new_image_data = []
            for item in datas:
                # change all grey pixels to white
                if item in list(range(50,255)):
                    new_image_data.append(255)
                else:
                    new_image_data.append(item)
            # update image data
            cleanimage.putdata(new_image_data)
            # now all white :) 
            
            
            #need to randomise the corner the ellipse was placed in 
            theta = np.random.choice([0,90,180,270])
            shapes.rotation(angle=theta)
            cleanimage = cleanimage.rotate(theta)
            centre = rotate_around_point_highperf(centre, theta, 
                                                  origin=(shapes.height/2, shapes.width/2))

            #add some grey/white holes 
            shapes.add_holes2(
                numholes=np.random.randint(40, 50),
                width=np.random.randint(3, 4),
            )
            #add blur
            shapes.add_blur(sig=1.5)
            
            #cvshapes = ChanVese(shapes.image)
            #cvshapes.run(steps=500, show_iterations=False)

            # they are meant to be reshaped inside Jaccard but python is
            # ignoring that for some reason so Im doing it here :))))
            #u1 = np.reshape(cvshapes.u, np.size(cvshapes.u))
            #u2 = np.reshape(clean_seg, np.size(clean_seg))
            #evaluation = EM.Jaccard(u1, u2)
            evaluation =1
            
            
        # save all the images
        cleanimage.save(
            fp=os.path.join(sample_folder, IMAGE_TYPE_NAMES["clean"]), format="PNG"
        )
        np.save(
            file=os.path.join(sample_folder, SEGMENTATION_TYPE_NAMES["clean"]),
            arr=np.array(cleanimage) / 255,
        )

        shapes.image.save(
            fp=os.path.join(sample_folder, IMAGE_TYPE_NAMES["dirty"]), format="PNG"
        )

        #np.save(
        #    file=os.path.join(sample_folder, SEGMENTATION_TYPE_NAMES["chan-vese"]),
        #    arr=cvshapes.u,
        #)

        #cvim = Image.fromarray(255 * cvshapes.u).convert("L")
        #cvim.save(
                #    fp=os.path.join(sample_folder, IMAGE_TYPE_NAMES["chan-vese"]), format="PNG"
        #)
        np.save(
            file=os.path.join(sample_folder, "tag.npy"),
            arr = np.array(centre).astype(int),
        )
        
        
def gen_data_polygons(FILEPATH,datasize,append=True,size=(128, 128)):
    
    #Polygons

    SHAPENAME = ["null", "null", "triangle","square"]
    times = datasize
    for j in range(2,4):
        
        root_dir = os.path.join(FILEPATH, SHAPENAME[j])
        
        if append:
            start_index = sum(
                1 for s in os.listdir(root_dir) if s.startswith(SAMPLE_FOLDER_PREFIX)
            )
        else:
            start_index = 0

        for i in tqdm(range(times)):
            sample_folder = os.path.join(
                root_dir, SAMPLE_FOLDER_PREFIX + "{}".format(i + start_index)
            )
            try:
                os.mkdir(sample_folder)
            except FileExistsError:
                pass

            # create random polygon
            shapes = ShapeGenerator(128, 128)
            shapes.add_polygon(size=0.4 * 128,nverts=j+1)

            cleanimage = shapes.image.copy()

            #need to trun the grey btis white for the clean segmentation 
            datas = cleanimage.getdata()
            new_image_data = []
            for item in datas:
                    # change all grey pixels to white
                    if item in list(range(50,255)):
                        new_image_data.append(255)
                    else:
                        new_image_data.append(item)

            #update image data
            cleanimage.putdata(new_image_data)
            clean_seg = np.array(cleanimage)
            # now all white :)

            #add some grey/white holes 
            shapes.add_holes(
                    numholes=np.random.randint(40, 50),
                    width=np.random.randint(3, 4),
                )


            #add blur
            shapes.add_blur(sig=1.5)
            #add noise
            shapes.add_noise(sig=0.1)

            #save 

            cleanimage.save(
                fp=os.path.join(sample_folder, IMAGE_TYPE_NAMES["clean"]), format="PNG"
            )
            np.save(
                file=os.path.join(sample_folder, SEGMENTATION_TYPE_NAMES["clean"]),
                arr=np.array(cleanimage) / 255,
            )

            shapes.image.save(
                fp=os.path.join(sample_folder, IMAGE_TYPE_NAMES["dirty"]), format="PNG"
            )
            np.save(
                file=os.path.join(sample_folder, SEGMENTATION_TYPE_NAMES["dirty"]),
                arr=np.array(shapes.image) / 255,
            )

def gen_data_round(FILEPATH,datasize,append=True,size=(128, 128)):
    
    #Polygons

    SHAPENAME = ["ellipse", "annulus"]
    times = datasize
    for j in range(2):
        
        root_dir = os.path.join(FILEPATH, SHAPENAME[j])
        
        if append:
            start_index = sum(
                1 for s in os.listdir(root_dir) if s.startswith(SAMPLE_FOLDER_PREFIX)
            )
        else:
            start_index = 0

        for i in tqdm(range(times)):
            sample_folder = os.path.join(
                root_dir, SAMPLE_FOLDER_PREFIX + "{}".format(i + start_index)
            )
            try:
                os.mkdir(sample_folder)
            except FileExistsError:
                pass

            # create random polygon
            shapes = ShapeGenerator(128, 128)
            if j==0:
                shapes.add_ellipse(size=0.4 * 128)
            else:
                shapes.add_anulus(size=0.4 * 128)

            cleanimage = shapes.image.copy()

            #need to trun the grey btis white for the clean segmentation 
            datas = cleanimage.getdata()
            new_image_data = []
            for item in datas:
                    # change all grey pixels to white
                    if item in list(range(50,255)):
                        new_image_data.append(255)
                    else:
                        new_image_data.append(item)

            #update image data
            cleanimage.putdata(new_image_data)
            clean_seg = np.array(cleanimage)
            # now all white :)

            #add some grey/white holes 
            shapes.add_holes(
                    numholes=np.random.randint(40, 50),
                    width=np.random.randint(3, 4),
                )


            #add blur
            shapes.add_blur(sig=1.5)
            #add noise
            shapes.add_noise(sig=0.01)

            #save 

            cleanimage.save(
                fp=os.path.join(sample_folder, IMAGE_TYPE_NAMES["clean"]), format="PNG"
            )
            np.save(
                file=os.path.join(sample_folder, SEGMENTATION_TYPE_NAMES["clean"]),
                arr=np.array(cleanimage) / 255,
            )

            shapes.image.save(
                fp=os.path.join(sample_folder, IMAGE_TYPE_NAMES["dirty"]), format="PNG"
            )
            np.save(
                file=os.path.join(sample_folder, SEGMENTATION_TYPE_NAMES["dirty"]),
                arr=np.array(shapes.image) / 255,
            )


 




        

def rotate_around_point_highperf(xy, theta, origin=(0, 0)):

    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(np.pi*theta/180)
    sin_rad = math.sin(np.pi*theta/180)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy
        
        
