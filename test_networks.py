#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
from Neural_Networks.networks import *

def test_net(net):
    for i in range(10):
        x = np.load(f"./data/image_{i}/dirty_cv_seg.npy")
        y = np.load(f"./data/image_{i}/clean_seg.npy")
        x_tensor = torch.tensor(x).unsqueeze(0).unsqueeze(0).float()
        y_tensor = torch.tensor(y).unsqueeze(0).unsqueeze(0).float()
        print("Dirty: {:.2f}\t Clean: {:.2f}\t Diff:{:.2f}".format(
            float(net(x_tensor)),
            float(net(y_tensor)),
            float(net(x_tensor) - net(y_tensor))))



NN = ConvNet1(1, 128, 128)
NN.load_state_dict(
    torch.load("./Neural_Networks/ConvNet1_trained", map_location=torch.device("cpu"))
)
print("Network: ConvNet1")
test_net(NN)

NN = ConvNet2(1, 128, 128)
NN.load_state_dict(
    torch.load("./Neural_Networks/ConvNet2_trained", map_location=torch.device("cpu"))
)
print("Network: ConvNet2")
test_net(NN)

NN = ConvNet2(1, 128, 128)
NN.load_state_dict(
    torch.load("./Neural_Networks/ConvNet2_trained_v2", map_location=torch.device("cpu"))
)
print("Network: ConvNet2_v2")
test_net(NN)

NN = ConvNet3(1, 128, 128)
NN.load_state_dict(
    torch.load("./Neural_Networks/ConvNet3_trained", map_location=torch.device("cpu"))
)
print("Network: ConvNet3")
test_net(NN)

NN = ConvNet4(1, 128, 128)
NN.load_state_dict(
    torch.load("./Neural_Networks/ConvNet4_trained", map_location=torch.device("cpu"))
)
print("Network: ConvNet4")
test_net(NN)

NN = ConvNet4(1, 128, 128)
NN.load_state_dict(
    torch.load("./Neural_Networks/ConvNet4_trained_v2", map_location=torch.device("cpu"))
)
print("Network: ConvNet4_v2")
test_net(NN)
