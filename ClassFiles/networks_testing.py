import torch
import torch.nn as nn
import torch.nn.functional as F


"""
currently using identical architecture to the image denoising ConvNet in Sebastian's paper
"""

"""
large output = chanvese, small output = groundtruth
"""
class ConvNet(nn.Module):
    def __init__(
        self, in_channels, in_height, in_width
    ):  # for greyscale set channels = 1
        if in_height % 16 or in_width % 16:
            print(
                "Error: height and width of image incompatible with architecture, must be divisible by 16"
            )
            return
        super().__init__()

        self.in_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width
        
        
        """
        defined incase you want to use them
        """
        self.poolavg = nn.AvgPool2d(2)
        self.poolmax = nn.MaxPool2d(2)
        
        
        """
        these are the actual NN layers
        """
        
        """
        inital edge detection (3x3 convolution) and processing of edge information (1x1 convolution)
        ideally would have many more channels in these two layers, but my GPU can't handle it
        """
        # input is size [batchsize, in_channels, in_height, in_width]
        self.conv1 = nn.Conv2d(
            in_channels, 32, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 32, heigh, width]
        
        self.conv2 = nn.Conv2d(
            32, 32, 1, padding=0, padding_mode='reflect'
        )  # [batchsize, 16, in_height, in_width]
        
        """
        aggregate information over a large region (for some reason strided 5x5 convolutions worked better than pooling)
        channels doubles each time image size halfs (at least every time my GPU can handle to increase),
        so we don't really lose any information (at least not if you count the number of dimensions of each layer)
        """
        self.conv3 = nn.Conv2d(
            32, 64, 5, stride=2, padding=2, padding_mode='reflect'
        )  # [batchsize, 16, in_height / 2, in_width / 2]
        
        self.conv4 = nn.Conv2d(
            64, 128, 5, stride=2, padding=2, padding_mode='reflect'
        )  # [batchsize, 32, in_height / 4, in_width / 4]
        
        self.conv5 = nn.Conv2d(
            128, 256, 5, stride=2, padding=2, padding_mode='reflect'
        )  # [batchsize, 64, in_height / 8, in_width / 8]
        
        self.conv6 = nn.Conv2d(
            256, 512, 5, stride=2, padding=2, padding_mode='reflect'
        )  # [batchsize, 128, in_height / 16, in_width / 16]
        
        """
        final processing of aggregated information (1x1 convolution) to output a single channel - how much the NN thinks this region looks like groundtruth or chanvese
        """
        self.conv7 = nn.Conv2d(
            512, 1, 1
        )  # [batchsize, 1, in_height / 16, in_width / 16]
        

    def forward(self, batch):
        # batch must be a torch.tensor on device = device, of dtype = torch.float, and of size [any, self.in_channels, self.in_height, self.in_width]
        if list(batch.size())[1:] != [self.in_channels, self.in_height, self.in_width]:
            print("Error: channels, height, and width different to initialisation")
            return

        layer1 = F.relu(self.conv1(batch))
        layer2 = F.relu(self.conv2(layer1))
        layer3 = F.relu(self.conv3(layer2))
        layer4 = F.relu(self.conv4(layer3))
        layer5 = F.relu(self.conv5(layer4))
        layer6 = F.relu(self.conv6(layer5))
        layer7 = self.conv7(layer6)
        assert list(layer7.size()) == [layer7.size(0), 1, self.in_height // 16, self.in_width // 16]
        
        """
        sum the outputs of the final layer
        - each output should represent how much each region looks like groundtruth or chanvese,
        and (at least in terms of the lipschitz condition I'm using for training)
        these 'distances' between groundtruth and chanvese should sum over disjoint regions
        (the regions are overlapping, but hey ho, what can you do)
        """
        output = layer7.flatten(1).sum(-1) * (16 ** 2) #scale so each pixel has O(1) effect on output...
        return output  # [batchsize, 1]
