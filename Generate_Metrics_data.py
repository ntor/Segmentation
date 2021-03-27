#!/usr/bin/env python3

import Neural_Networks_lunglike.ClassFiles.networks as net
import torch

FILEPATH = "./data"


NN = net.ConvNet1(1, 128, 128)
NN.load_state_dict(
    torch.load("./Neural_Networks_lunglike/ConvNet1_trained", map_location=torch.device("cpu") ))
NetName = "ConnNet1"


from ClassFiles.GeneratedDatasetNN import generate_data_NN as gen

gen(1000,FILEPATH,NN,NetName)