#!/usr/bin/env python3

import torch
import ClassFiles.Networks as net
import ClassFiles.GeneratedDataset as dat
import ClassFiles.DeepSegmentation as rec
from torch.utils.data import DataLoader
from PIL import Image

clean_image_dataset = dat.ImageDataset("./Neural_Networks_lunglike/eval", "clean")
dirty_image_dataset = dat.ImageDataset("./Neural_Networks_lunglike/eval", "dirty")
cv_seg_dataset = dat.SegmentationDataset("./Neural_Networks_lunglike/eval", "chan-vese")

clean_image_dataloader = DataLoader(clean_image_dataset, batch_size=1)
dirty_image_dataloader = DataLoader(dirty_image_dataset, batch_size=1)
cv_seg_dataloader = DataLoader(cv_seg_dataset, batch_size=1)

NN = net.ConvNet1(1, 128, 128)
NN.load_state_dict(torch.load("./Neural_Networks_lunglike/ConvNet1_trained"))

clean_image_iter = iter(clean_image_dataloader)
dirty_image_iter = iter(dirty_image_dataloader)
cv_seg_iter = iter(cv_seg_dataloader)


dirty_im = Image.fromarray(
    255 * dirty_image_iter.next()[0, 0].numpy()
).convert("L")
cv_seg = cv_seg_iter.next().squeeze()

reconstruction = rec.DeepSegmentation(dirty_im, NN, cv_seg)
reconstruction.show_segmentation()
reconstruction.run(1000, lmb_reg=10, epsilon=0.001, show_iterations=True)
reconstruction.show_segmentation()
