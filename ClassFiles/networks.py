import torch
import torch.nn as nn
import torch.nn.functional as F


"""
large output = chanvese, small output = groundtruth
"""
class SebastianConvNet(nn.Module):
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

        # input is size [batchsize, in_channels, in_height, in_width]
        self.conv1 = nn.Conv2d(
            in_channels, 16, 5, padding=2
        )  # [batchsize, 16, heigh, width]
        self.conv2 = nn.Conv2d(
            16, 32, 5, padding=2
        )  # [batchsize, 32, in_height, in_width]
        self.conv3 = nn.Conv2d(
            32, 32, 5, stride=2, padding=2
        )  # batchsize, 32, in_height / 2, in_width / 2]
        self.conv4 = nn.Conv2d(
            32, 64, 5, stride=2, padding=2
        )  # [batchsize, 64, in_height / 4, in_width / 4]
        self.conv5 = nn.Conv2d(
            64, 64, 5, stride=2, padding=2
        )  # batchsize, 64, in_height / 8, in_width / 8]
        self.conv6 = nn.Conv2d(
            64, 128, 5, stride=2, padding=2
        )  # [batchsize, 128, in_height / 16, in_width / 16]

        remaining_pixels = in_height * in_width // (16 * 16)
        self.remaining_dimensions = remaining_pixels * 128

        # after reshaping, the input to the feedforward layers will be [batchsize, remaining_dimension]
        self.linear1 = nn.Linear(self.remaining_dimensions, 256)  # [batchsize, 256]
        self.linear2 = nn.Linear(256, 1)  # [batchsize, 1]

    def forward(self, batch):
        # batch must be a torch.tensor on device = device, of dtype = torch.float, and of size [any, self.in_channels, self.in_height, self.in_width]
        if list(batch.size())[1:] != [self.in_channels, self.in_height, self.in_width]:
            print("Error: channels, height, and width different to initialisation")
            return

        layer1 = F.leaky_relu(self.conv1(batch), negative_slope=0.1)
        layer2 = F.leaky_relu(self.conv2(layer1), negative_slope=0.1)
        layer3 = F.leaky_relu(self.conv3(layer2), negative_slope=0.1)
        layer4 = F.leaky_relu(self.conv4(layer3), negative_slope=0.1)
        layer5 = F.leaky_relu(self.conv5(layer4), negative_slope=0.1)
        layer6 = F.leaky_relu(self.conv6(layer5), negative_slope=0.1)
        layer6_reshape = layer6.view(-1, self.remaining_dimensions)
        layer7 = F.leaky_relu(self.linear1(layer6_reshape), negative_slope=0.1)
        output = self.linear2(layer7)
        return output  # [batchsize, 1]



class ConvNet1(nn.Module):
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
        self.poolavg = nn.AvgPool2d(16)
        self.poolmax = nn.MaxPool2d(4)
        
        
        """
        these are the actual NN layers
        """
        
        """
        inital edge detection (3x3 convolution) and processing of edge information (1x1 convolution)
        ideally would have many more channels in these two layers, but my GPU can't handle it
        """
        # input is size [batchsize, in_channels, in_height, in_width]
        self.conv1 = nn.Conv2d(
            in_channels, 16, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 32, heigh, width]
        
        """
        we don't actually use this layer, needs to be here to be able to reload the NN
        """
        self.conv2 = nn.Conv2d(
            32, 16, 1, padding=0, padding_mode='reflect'
        )  # [batchsize, 16, in_height, in_width]
        
        """
        aggregate information over a large region (for some reason strided 5x5 convolutions works better than pooling)
        channels doubles each time image size halfs, so we don't really lose any information
        (at least not if you count the number of dimensions of each layer)
        """
        self.conv3 = nn.Conv2d(
            16, 32, 5, stride=2, padding=2, padding_mode='reflect'
        )  # [batchsize, 16, in_height / 2, in_width / 2]
        
        self.conv4 = nn.Conv2d(
            32, 64, 5, stride=2, padding=2, padding_mode='reflect'
        )  # [batchsize, 32, in_height / 4, in_width / 4]
        
        self.conv5 = nn.Conv2d(
            64, 128, 5, stride=2, padding=2, padding_mode='reflect'
        )  # [batchsize, 64, in_height / 8, in_width / 8]
        
        """
        final processing of aggregated information (1x1 convolution) to output only a few channels - then combine with one last 16*16 convolution (linear1), before processing and outputing a single value (F.relu followed by linear2)
        """
        self.conv10 = nn.Conv2d(
            128, 8, 1
        )  # [batchsize, 1, in_height / 128, in_width / 128]
        
        self.linear1 = nn.Linear(8 * 16 * 16, 8 * 16 * 16)
        self.linear2 = nn.Linear(8 *16 * 16, 1)
        
        
    def forward(self, batch):
        # batch must be a torch.tensor on device = device, of dtype = torch.float, and of size [any, self.in_channels, self.in_height, self.in_width]
        if list(batch.size())[1:] != [self.in_channels, self.in_height, self.in_width]:
            print("Error: channels, height, and width different to initialisation")
            return

        layer1 = F.relu(self.conv1(batch))
        #layer2 = F.relu(self.conv2(layer1))
        layer3 = F.relu(self.conv3(layer1))
        layer4 = F.relu(self.conv4(layer3))
        layer5 = F.relu(self.conv5(layer4))
        layer10 = F.relu(self.conv10(layer5))
        output = self.linear2(F.relu(self.linear1(layer10.flatten(1))))
        assert list(output.size()) == [output.size(0), 1]
            
        return output  # [batchsize, 1]



class ConvNet2(nn.Module):
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
        these are the actual NN layers
        """
        
        """
        inital edge detection (3x3 convolution) followed by large 'gaussian' - like convolution
        (3 lots of 3x3 convolutions with no relu inbetween)
        """
        # input is size [batchsize, in_channels, in_height, in_width]
        self.conv1 = nn.Conv2d(
            in_channels, 16, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 32, heigh, width]
        
        self.conv2_1 = nn.Conv2d(
            16, 32, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 16, in_height, in_width]
        
        self.conv2_2 = nn.Conv2d(
            32, 32, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 16, in_height, in_width]
        
        self.conv2_3 = nn.Conv2d(
            32, 32, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 16, in_height, in_width]
        
        """
        aggregate information over a large region (for some reason strided 5x5 convolutions works better than pooling)
        channels doubles each time image size halfs, so we don't really lose any information
        (at least not if you count the number of dimensions of each layer)
        """
        
        self.conv3 = nn.Conv2d(
            32, 64, 5, stride=2, padding=2, padding_mode='reflect'
        )  # [batchsize, 16, in_height / 4, in_width / 4]
        
        self.conv4 = nn.Conv2d(
            64, 128, 5, stride=2, padding=2, padding_mode='reflect'
        )  # [batchsize, 32, in_height / 8, in_width / 8]
        
        self.conv5 = nn.Conv2d(
            128, 256, 5, stride=2, padding=2, padding_mode='reflect'
        )  # [batchsize, 64, in_height / 16, in_width / 16]
        
        """
        final processing of aggregated information (1x1 convolution) to output only a few channels - then combine with one last 16*16 convolution with no padding (linear1), before processing and outputing a single value (F.relu followed by linear2)
        """
        self.conv10 = nn.Conv2d(
            256, 8, 1
        )  # [batchsize, 1, in_height / 128, in_width / 128]
        
        self.linear1 = nn.Linear(8 * 16 ** 2, 8 * 16 ** 2)
        self.linear2 = nn.Linear(8 * 16 ** 2, 1)
        
        
    def forward(self, batch):
        # batch must be a torch.tensor on device = device, of dtype = torch.float, and of size [any, self.in_channels, self.in_height, self.in_width]
        if list(batch.size())[1:] != [self.in_channels, self.in_height, self.in_width]:
            print("Error: channels, height, and width different to initialisation")
            return

        layer1 = F.relu(self.conv1(batch))
        layer2 = F.relu(self.conv2_3(self.conv2_2(self.conv2_1(layer1))))
        layer3 = F.relu(self.conv3(layer2))
        layer4 = F.relu(self.conv4(layer3))
        layer5 = F.relu(self.conv5(layer4))
        layer10 = F.relu(self.conv10(layer5))
        output = self.linear2(F.relu(self.linear1(layer10.flatten(1))))
        assert list(output.size()) == [output.size(0), 1]
            
        return output  # [batchsize, 1]



class ConvNet3(nn.Module):
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
        these are the actual NN layers
        """
        
        """
        inital edge detection (3x3 convolution)
        """
        # input is size [batchsize, in_channels, in_height, in_width]
        self.conv1 = nn.Conv2d(
            in_channels, 16, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 32, heigh, width]
        
        """
        quick&dirty downscaling strategy
        (4x4 strided convolutions with no relu inbetween,
        we use 4x4 not 5x5 to avoid some pixels being unreasonably weighted)
        """
        self.conv3 = nn.Conv2d(
            16, 32, 4, stride=2, padding=1, padding_mode='reflect'
        )  # [batchsize, 16, in_height / 2, in_width / 2]
        
        self.conv4 = nn.Conv2d(
            32, 32, 4, stride=2, padding=1, padding_mode='reflect'
        )  # [batchsize, 32, in_height / 4, in_width / 4]
        
        """
        local aggregation
        """
        self.conv5 = nn.Conv2d(
            32, 128, 7, stride=2, padding=3, padding_mode='reflect'
        )  # [batchsize, 32, in_height / 8, in_width / 8]
        
        self.conv10 = nn.Conv2d(
            128, 8, 1
        )  # [batchsize, 1, in_height / 128, in_width / 128]
        
        """
        global aggregation
        """
        self.linear1 = nn.Linear(8 * 16 * 16, 8 * 16 * 16)
        self.linear2 = nn.Linear(8 * 16 * 16, 1)
        
        
    def forward(self, batch):
        # batch must be a torch.tensor on device = device, of dtype = torch.float, and of size [any, self.in_channels, self.in_height, self.in_width]
        if list(batch.size())[1:] != [self.in_channels, self.in_height, self.in_width]:
            print("Error: channels, height, and width different to initialisation")
            return

        layer1 = F.relu(self.conv1(batch))
        layer4 = F.relu(self.conv4(self.conv3(layer1)))
        layer10 = F.relu(self.conv10(F.relu(self.conv5(layer4))))
        output = self.linear2(F.relu(self.linear1(layer10.flatten(1))))
        assert list(output.size()) == [output.size(0), 1]
            
        return output  # [batchsize, 1]



class ConvNet4(nn.Module):
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
        these are the actual NN layers
        """
        
        """
        inital edge detection (3x3 convolution) followed by large 'gaussian' - like convolution
        (3 lots of 3x3 convolutions with no relu inbetween)
        """
        # input is size [batchsize, in_channels, in_height, in_width]
        self.conv1 = nn.Conv2d(
            in_channels, 16, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 32, heigh, width]
        
        self.conv2_1 = nn.Conv2d(
            16, 32, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 16, in_height, in_width]
        
        self.conv2_2 = nn.Conv2d(
            32, 32, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 16, in_height, in_width]
        
        self.conv2_3 = nn.Conv2d(
            32, 32, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 16, in_height, in_width]
        
        
        """
        quick&dirty downscaling strategy
        (4x4 strided convolutions with no relu inbetween,
        we use 4x4 not 5x5 to avoid some pixels being unreasonably weighted)
        """
        self.conv3 = nn.Conv2d(
            32, 32, 4, stride=2, padding=1, padding_mode='reflect'
        )  # [batchsize, 16, in_height / 2, in_width / 2]
        
        self.conv4 = nn.Conv2d(
            32, 32, 4, stride=2, padding=1, padding_mode='reflect'
        )  # [batchsize, 32, in_height / 4, in_width / 4]
        
        
        """
        local aggregation
        """
        self.conv5 = nn.Conv2d(
            32, 128, 7, stride=2, padding=3, padding_mode='reflect'
        )  # [batchsize, 32, in_height / 8, in_width / 8]
        
        self.conv10 = nn.Conv2d(
            128, 8, 1
        )  # [batchsize, 1, in_height / 128, in_width / 128]
        
        
        """
        global aggregation
        """
        self.linear1 = nn.Linear(8 * 16 * 16, 8 * 16 * 16)
        self.linear2 = nn.Linear(8 * 16 * 16, 1)
        
        
    def forward(self, batch):
        # batch must be a torch.tensor on device = device, of dtype = torch.float, and of size [any, self.in_channels, self.in_height, self.in_width]
        if list(batch.size())[1:] != [self.in_channels, self.in_height, self.in_width]:
            print("Error: channels, height, and width different to initialisation")
            return

        layer1 = F.relu(self.conv1(batch))
        layer2 = F.relu(self.conv2_3(self.conv2_2(self.conv2_1(layer1))))
        layer4 = F.relu(self.conv4(self.conv3(layer2)))
        layer10 = F.relu(self.conv10(F.relu(self.conv5(layer4))))
        output = self.linear2(F.relu(self.linear1(layer10.flatten(1))))
        assert list(output.size()) == [output.size(0), 1]
            
        return output  # [batchsize, 1]



class ConvNet5(nn.Module):
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
        inital edge detection (3x3 convolution) and processing of edge information (1x1 convolution)
        ideally would have many more channels in these two layers, but my GPU can't handle it
        """
        # input is size [batchsize, in_channels, in_height, in_width]
        self.conv1 = nn.Conv2d(
            in_channels, 16, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 32, heigh, width]
        
        """
        aggregate information over a large region (for some reason strided convolutions works better than pooling)
        channels doubles each time image size halfs, so we don't really lose any information
        (at least not if you count the number of dimensions of each layer)
        """
        self.conv3 = nn.Conv2d(
            16, 32, 5, stride=2, padding=2, padding_mode='reflect'
        )  # [batchsize, 16, in_height / 2, in_width / 2]
        
        self.conv4 = nn.Conv2d(
            32, 64, 5, stride=2, padding=2, padding_mode='reflect'
        )  # [batchsize, 32, in_height / 4, in_width / 4]
        
        self.conv5 = nn.Conv2d(
            64, 128, 5, stride=2, padding=2, padding_mode='reflect'
        )  # [batchsize, 64, in_height / 8, in_width / 8]
        
        """
        final processing of aggregated information (1x1 convolution) to output only a few channels - then combine with one last 16*16 convolution (linear1), before processing and outputing a single value (F.relu followed by linear2)
        """
        self.conv10 = nn.Conv2d(
            128, 8, 1
        )  # [batchsize, 1, in_height / 16, in_width / 16]
        
        self.linear1 = nn.Linear(8 * 16 * 16, 8 * 16 * 16)
        self.linear2 = nn.Linear(8 * 16 * 16, 1)
        
        
    def forward(self, batch):
        # batch must be a torch.tensor on device = device, of dtype = torch.float, and of size [any, self.in_channels, self.in_height, self.in_width]
        if list(batch.size())[1:] != [self.in_channels, self.in_height, self.in_width]:
            print("Error: channels, height, and width different to initialisation")
            return

        layer1 = F.relu(self.conv1(batch))
        layer3 = F.relu(self.conv3(layer1))
        layer4 = F.relu(self.conv4(layer3))
        layer5 = F.relu(self.conv5(layer4))
        layer10 = F.relu(self.conv10(layer5))
        output = (self.linear2(F.relu(self.linear1(layer10.flatten(1)))))
        assert list(output.size()) == [output.size(0), 1]
            
        return output  # [batchsize, 1]



class ConvNet6(nn.Module):
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
        these are the actual NN layers
        """
        
        """
        inital edge detection (3x3 convolution) followed by large 'gaussian' - like convolution
        (3 lots of 3x3 convolutions with no relu inbetween)
        """
        # input is size [batchsize, in_channels, in_height, in_width]
        self.conv1 = nn.Conv2d(
            in_channels, 16, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 32, heigh, width]
        
        self.conv2_1 = nn.Conv2d(
            16, 32, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 16, in_height, in_width]
        
        self.conv2_2 = nn.Conv2d(
            32, 32, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 16, in_height, in_width]
        
        self.conv2_3 = nn.Conv2d(
            32, 32, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 16, in_height, in_width]
        
        self.conv2_4 = nn.Conv2d(
            32, 32, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 16, in_height, in_width]
        
        self.conv2_5 = nn.Conv2d(
            32, 32, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 16, in_height, in_width]
        
        
        """
        quick&dirty downscaling strategy
        (4x4 strided convolutions with no relu inbetween,
        we use 4x4 not 5x5 to avoid some pixels being unreasonably weighted)
        """
        self.conv3 = nn.Conv2d(
            32, 64, 6, stride=2, padding=2, padding_mode='reflect'
        )  # [batchsize, 16, in_height / 2, in_width / 2]
        
        self.conv4 = nn.Conv2d(
            64, 64, 6, stride=2, padding=2, padding_mode='reflect'
        )  # [batchsize, 32, in_height / 4, in_width / 4]
        
        
        """
        local aggregation
        """
        self.conv5 = nn.Conv2d(
            64, 256, 8, stride=2, padding=3, padding_mode='reflect'
        )  # [batchsize, 32, in_height / 8, in_width / 8]
        
        self.conv10 = nn.Conv2d(
            256, 8, 1
        )  # [batchsize, 1, in_height / 8, in_width / 8]
        
        
        """
        global aggregation
        """
        self.linear1 = nn.Linear(8 * 16 * 16, 8 * 16 * 16)
        self.linear2 = nn.Linear(8 * 16 * 16, 1)
        
        
    def forward(self, batch):
        # batch must be a torch.tensor on device = device, of dtype = torch.float, and of size [any, self.in_channels, self.in_height, self.in_width]
        if list(batch.size())[1:] != [self.in_channels, self.in_height, self.in_width]:
            print("Error: channels, height, and width different to initialisation")
            return

        layer1 = F.relu(self.conv1(batch))
        layer2 = F.relu(self.conv2_5(self.conv2_4(self.conv2_3(self.conv2_2(self.conv2_1(layer1))))))
        layer4 = F.relu(self.conv4(self.conv3(layer2)))
        layer10 = F.relu(self.conv10(F.relu(self.conv5(layer4))))
        output = self.linear2(F.relu(self.linear1(layer10.flatten(1))))
        assert list(output.size()) == [output.size(0), 1]
            
        return output  # [batchsize, 1]
    


class ConvNet7(nn.Module):
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
        these are the actual NN layers
        """
        
        """
        inital edge detection (3x3 convolution) followed by large 'gaussian' - like convolution
        (3 lots of 3x3 convolutions with no relu inbetween)
        """
        # input is size [batchsize, in_channels, in_height, in_width]
        self.conv1 = nn.Conv2d(
            in_channels, 16, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 32, heigh, width]
        
        self.conv2_1 = nn.Conv2d(
            16, 32, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 16, in_height, in_width]
        
        self.conv2_2 = nn.Conv2d(
            32, 64, 3, stride = 2, padding=1, padding_mode='reflect'
        )  # [batchsize, 16, in_height, in_width]
        
        self.conv2_3 = nn.Conv2d(
            64, 64, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 16, in_height, in_width]
        
        self.conv2_4 = nn.Conv2d(
            64, 128, 3, stride = 2, padding=1, padding_mode='reflect'
        )  # [batchsize, 16, in_height, in_width]
        
        self.conv2_5 = nn.Conv2d(
            128, 128, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 16, in_height, in_width]
        
        
        
        self.conv10 = nn.Conv2d(
            128, 8, 1
        )  # [batchsize, 1, in_height / 4, in_width / 4]
        
        
        """
        global aggregation
        """
        self.linear1 = nn.Linear(8 * 32 * 32, 8 * 32 * 32)
        self.linear2 = nn.Linear(8 * 32 * 32, 1)
        
        
    def forward(self, batch):
        # batch must be a torch.tensor on device = device, of dtype = torch.float, and of size [any, self.in_channels, self.in_height, self.in_width]
        if list(batch.size())[1:] != [self.in_channels, self.in_height, self.in_width]:
            print("Error: channels, height, and width different to initialisation")
            return

        layer1 = F.relu(self.conv1(batch))
        layer2 = F.relu(self.conv2_5(self.conv2_4(self.conv2_3(self.conv2_2(self.conv2_1(layer1))))))
        layer10 = F.relu(self.conv10(layer2))
        output = self.linear2(F.relu(self.linear1(layer10.flatten(1))))
        assert list(output.size()) == [output.size(0), 1]
            
        return output  # [batchsize, 1]



class ConvNet8(nn.Module):
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
        inital edge detection (3x3 convolution) and processing of edge information (1x1 convolution)
        """
        # input is size [batchsize, in_channels, in_height, in_width]
        self.conv1 = nn.Conv2d(
            in_channels, 16, 3, padding=1, padding_mode='reflect'
        )  # [batchsize, 32, height, width]
        
        
        """
        'autoencoder' aggregation followed by 4x4 strided convolution
        found that strided convolutions worked better than pooling
        use 4x4 to avoided certain pixels being weighted more than others
        channels double for each strided convolution, to reduce loss of information
        repeat four times for suitable downscaling
        """
        self.conv2_0 = nn.Conv2d(
            16, 32, 5, padding=2, padding_mode='reflect'
        )  # [batchsize, 32, height, width]
        
        self.conv2_1 = nn.Conv2d(
            32, 16, 1, padding=0, padding_mode='reflect'
        )  # [batchsize, 32, height, width]
        
        self.conv2_2 = nn.Conv2d(
            16, 32, 4, padding=1, stride = 2, padding_mode='reflect'
        )  # [batchsize, 32, height / 2, width / 2]
        
        
        self.conv3_0 = nn.Conv2d(
            32, 64, 5, padding=2, padding_mode='reflect'
        )  # [batchsize, 32, height / 2, width / 2]
        
        self.conv3_1 = nn.Conv2d(
            64, 32, 1, padding=0, padding_mode='reflect'
        )  # [batchsize, 32, height / 2, width / 2]
        
        self.conv3_2 = nn.Conv2d(
            32, 64, 4, padding=1, stride = 2, padding_mode='reflect'
        )  # [batchsize, 32, height / 4, width / 4]
        
        
        self.conv4_0 = nn.Conv2d(
            64, 128, 5, padding=2, padding_mode='reflect'
        )  # [batchsize, 32, height / 4, width / 4]
        
        self.conv4_1 = nn.Conv2d(
            128, 64, 1, padding=0, padding_mode='reflect'
        )  # [batchsize, 32, height / 4, width / 4]
        
        self.conv4_2 = nn.Conv2d(
            64, 128, 4, padding=1, stride = 2, padding_mode='reflect'
        )  # [batchsize, 32, height / 8, width / 8]
        
        
        self.conv5_0 = nn.Conv2d(
            128, 256, 5, padding=2, padding_mode='reflect'
        )  # [batchsize, 32, height / 8, width / 8]
        
        self.conv5_1 = nn.Conv2d(
            256, 128, 1, padding=0, padding_mode='reflect'
        )  # [batchsize, 32, height / 8, width / 8]
        
        self.conv5_2 = nn.Conv2d(
            128, 256, 4, padding=1, stride = 2, padding_mode='reflect'
        )  # [batchsize, 32, height / 16, width / 16]
        
        """
        1x1 convolution with only a few channels
        this is so the final feedforward network is not too large
        use feedforward network with one hidden layer to combine all convolutions
        """
        self.conv10 = nn.Conv2d(
            256, 32, 1
        )  # [batchsize, 1, in_height / 16, in_width / 16]
        
        self.linear1 = nn.Linear(32 * 8 * 8, 32 * 8 * 8)
        self.linear2 = nn.Linear(32 * 8 * 8, 1)
        
        
    def forward(self, batch):
        # batch must be a torch.tensor on device = device, of dtype = torch.float, and of size [any, self.in_channels, self.in_height, self.in_width]
        if list(batch.size())[1:] != [self.in_channels, self.in_height, self.in_width]:
            print("Error: channels, height, and width different to initialisation")
            return

        layer1 = F.relu(self.conv1(batch))
        layer2 = F.relu(self.conv2_2(F.relu(self.conv2_1(F.relu(self.conv2_0(layer1))))))
        layer3 = F.relu(self.conv3_2(F.relu(self.conv3_1(F.relu(self.conv3_0(layer2))))))
        layer4 = F.relu(self.conv4_2(F.relu(self.conv4_1(F.relu(self.conv4_0(layer3))))))
        layer5 = F.relu(self.conv5_2(F.relu(self.conv5_1(F.relu(self.conv5_0(layer4))))))
        layer10 = F.relu(self.conv10(layer5))
        output = (self.linear2(F.relu(self.linear1(layer10.flatten(1)))))
        assert list(output.size()) == [output.size(0), 1]
            
        return output  # [batchsize, 1]
