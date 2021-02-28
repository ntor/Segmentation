import torch
import torch.nn as nn
import torch.nn.functional as F


"""
currently using identical architecture to the image denoising ConvNet in Sebastian's paper
"""
class SebastianConvNet(nn.Module):
    def __init__(self, in_channels, in_height, in_width): #for greyscale set channels = 1
        if in_height % 16 or in_width % 32:
            print('Error: height and width of image incompatible with architecture, must be divisible by 16')
            return
        super().__init__()
        
        self.in_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width
        
        #input is size [batchsize, in_channels, in_height, in_width]
        self.conv1 = nn.Conv2d(in_channels, 16, 5, padding = 2) #[batchsize, 16, heigh, width]
        self.conv2 = nn.Conv2d(16, 32, 5, padding = 2) #[batchsize, 32, in_height, in_width]
        self.conv3 = nn.Conv2d(32, 32, 5, stride = 2, padding = 2) #batchsize, 32, in_height / 2, in_width / 2]
        self.conv4 = nn.Conv2d(32, 64, 5, stride = 2, padding = 2) #[batchsize, 64, in_height / 4, in_width / 4]
        self.conv5 = nn.Conv2d(64, 64, 5, stride = 2, padding = 2) #batchsize, 64, in_height / 8, in_width / 8]
        self.conv6 = nn.Conv2d(64, 128, 5, stride = 2, padding = 2) #[batchsize, 128, in_height / 16, in_width / 16]
        
        remaining_pixels = in_height * in_width // (16 * 16)
        self.remaining_dimensions = remaining_pixels * 128
        
        #after reshaping, the input to the feedforward layers will be [batchsize, remaining_dimension]
        self.linear1 = nn.Linear(self.remaining_dimensions, 256) #[batchsize, 256]
        self.linear2 = nn.Linear(256, 1) #[batchsize, 1]
    
    def forward(self, batch):
        #batch must be a torch.tensor on device = device, of dtype = torch.float, and of size [any, self.in_channels, self.in_height, self.in_width]
        if list(batch.size())[1:] != [self.in_channels, self.in_height, self.in_width]:
            print('Error: channels, height, and width different to initialisation')
            return
        
        layer1 = F.leaky_relu(self.conv1(batch), negative_slope = 0.1)
        layer2 = F.leaky_relu(self.conv2(layer1), negative_slope = 0.1)
        layer3 = F.leaky_relu(self.conv3(layer2), negative_slope = 0.1)
        layer4 = F.leaky_relu(self.conv4(layer3), negative_slope = 0.1)
        layer5 = F.leaky_relu(self.conv5(layer4), negative_slope = 0.1)
        layer6 = F.leaky_relu(self.conv6(layer5), negative_slope = 0.1)
        layer6_reshape = layer6.view(-1, self.remaining_dimensions)
        layer7 = F.leaky_relu(self.linear1(layer6_reshape), negative_slope = 0.1)
        output = self.linear2(layer7)
        return output #[batchsize, 1]