import torch
import torch.nn as nn
from collections import OrderedDict


class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        """
        This is a set of Conv-BatchNormalization-Relu module.
        Args:
            in_channels: input channels of the module
            out_channels: output channels of the module
            **kwargs: other parameters
        """
        super(ConvBNRelu, self).__init__()
        self.Conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.BN = nn.BatchNorm2d(out_channels)
        self.Relu = nn.ReLU()

    def forward(self, x):
        x = self.Conv(x)
        x = self.BN(x)
        x = self.Relu(x)

        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()

        self.branch1 = nn.Sequential(OrderedDict([
            ('1*1 Conv', ConvBNRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1))  # 64*26*26
        ]))
        self.branch2 = nn.Sequential(OrderedDict([
            ('3*3 Conv', ConvBNRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)),
            ('1*1 Conv', ConvBNRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=1))  # 64*26*26
        ]))
        self.branch3 = nn.Sequential(OrderedDict([
            ('1*1 Conv', ConvBNRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=2)),
            ('5*5 Conv', ConvBNRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=5))  # 64*26*26
        ]))
        self.branch4 = nn.Sequential(OrderedDict([
            ('3*3 Max Pool', nn.MaxPool2d(kernel_size=3, stride=1, padding=1)),
            ('1*1 Conv', ConvBNRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1))  # 64*26*26
        ]))

    def forward(self, x):
        x_branch1 = self.branch1(x)
        x_branch2 = self.branch2(x)
        x_branch3 = self.branch3(x)
        x_branch4 = self.branch4(x)
        # dim=1 means concat in the channel dimension, batch_size*channel*width*height
        x = torch.cat([x_branch1, x_branch2, x_branch3, x_branch4], dim=1)

        return x


class Inception10(nn.Module):
    def __init__(self):
        super(Inception10, self).__init__()

        self.c1 = ConvBNRelu(in_channels=1, out_channels=16, kernel_size=3)  # 16*26*26
        self.inception1 = InceptionModule(in_channels=16, out_channels=64)   # 256*26*26
        self.avg_pool1 = nn.AdaptiveAvgPool2d((16, 16))  # 256*16*16
        self.inception2 = InceptionModule(in_channels=256, out_channels=512)   # 2048*14*14
        self.avg_pool2 = nn.AdaptiveAvgPool2d((10, 10))  # 2048*10*10
        self.c3 = nn.Sequential(OrderedDict([
            ('FullCon3', nn.Linear(in_features=2048*10*10, out_features=1024)),
            ('Relu3', nn.ReLU()),
            ('Drop3', nn.Dropout(p=0.5))
        ]))
        self.c4 = nn.Sequential(OrderedDict([
            ('FullCon4', nn.Linear(in_features=1024, out_features=128)),
            ('Relu4', nn.ReLU()),
            ('Drop4', nn.Dropout(p=0.5))
        ]))
        self.c5 = nn.Sequential(OrderedDict([
            ('FullCon5', nn.Linear(in_features=128, out_features=10)),
            ('Sig5', nn.LogSoftmax(dim=-1)),
        ]))

    def forward(self, img):
        output = self.c1(img)
        output = self.inception1(output)
        output = self.avg_pool1(output)
        output = self.inception2(output)
        output = self.avg_pool2(output)

        output = torch.flatten(output, 1)
        output = self.c3(output)
        output = self.c4(output)
        output = self.c5(output)

        return output
