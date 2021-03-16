#!/usr/bin/env python3

import torch
import numpy as np
import ClassFiles.Networks as nets
import ClassFiles.GeneratedDataset as dat
from torch.utils.data import DataLoader
import ClassFiles.Training as train

# Initialise the neural network. Optionally load a previously trained one.

NN = nets.ConvNet4(1, 128, 128)
NN.load_state_dict(
    torch.load(
        "./Neural_Networks/ConvNet4_trained_v2", map_location=torch.device("cpu")
    )
)


# Set up DataLoader instances for iterating over the data during training

cv_seg_dataset = dat.SegmentationDataset("./data/", seg_type="chan-vese")
cv_seg_dataloader = DataLoader(cv_seg_dataset, batch_size=50, shuffle=True)
# cv_seg_iter = iter(cv_seg_dataloader)

clean_seg_dataset = dat.SegmentationDataset("./data/", seg_type="clean")
clean_seg_dataloader = DataLoader(clean_seg_dataset, batch_size=50, shuffle=True)
# clean_seg_iter = iter(clean_seg_dataloader)

NN = train.train_regulariser(NN, clean_seg_dataloader, cv_seg_dataloader, epochs=1)
