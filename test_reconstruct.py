#!/usr/bin/env python3

# This file loads a networks and uses it for constructing a segmentation of some
# image. (Don't reuse, the __getitem__ really shouldn't be used this way.)

import torch
import ClassFiles.Networks as net
import ClassFiles.GeneratedDataset as dat
import ClassFiles.DeepSegmentation as rec
from PIL import Image

clean_image_dataset = dat.ImageDataset("./Neural_Networks_lunglike/eval", "clean")
dirty_image_dataset = dat.ImageDataset("./Neural_Networks_lunglike/eval", "dirty")
cv_seg_dataset = dat.SegmentationDataset("./Neural_Networks_lunglike/eval", "chan-vese")

for i in range(2):
    print('Image Number: {:.2f}'.format(i))
    image_number = i

    NN = net.ConvNet8(1, 128, 128)
    NN.load_state_dict(torch.load("./Neural_Networks_lunglike/ConvNet8_trained", map_location = torch.device("cpu")))

    dirty_im = Image.fromarray(
        255 * dirty_image_dataset.__getitem__(image_number)[0].numpy()
    ).convert("L")
    cv_seg = cv_seg_dataset.__getitem__(image_number)

    reconstruction = rec.DeepSegmentation(dirty_im, NN, cv_seg)
    reconstruction.show_segmentation()
    reconstruction.run(500, lmb_reg=14, epsilon=0.1, show_iterations=True)
    reconstruction.show_segmentation()
