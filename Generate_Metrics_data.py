#!/usr/bin/env python3

import ClassFiles.Networks as net
import torch

FILEPATH = "./data"


NN = net.ConvNet8(1, 128, 128)
NN.load_state_dict(
    torch.load("./Neural_Networks_lunglike/ConvNet8_trained", map_location=torch.device("cpu") ))
NetName = "ConvNet8"


from ClassFiles.GeneratedDatasetNN import generate_data_NN as gen

gen(1,FILEPATH,NN,NetName)