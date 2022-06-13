import torch
import torch.nn as nn
from collections import OrderedDict


class VGGNet16(nn.Module):
    def __init__(self):
        super(VGGNet16, self).__init__()
        self.c1 = nn.Sequential(OrderedDict([
            ('Conv1', nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding=1)),  # 64*28*28
            ('Relu1', nn.ReLU()),
        ]))
        self.c2 = nn.Sequential(OrderedDict([
            ('Conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)),  # 64*28*28
            ('Relu2', nn.ReLU()),
            ('Pool2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),  # 64*14*14
        ]))
        self.c3 = nn.Sequential(OrderedDict([
            ('Conv3', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)),  # 128*14*14
            ('Relu3', nn.ReLU()),
        ]))
        self.c4 = nn.Sequential(OrderedDict([
            ('Conv4', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1)),  # 128*14*14
            ('Relu4', nn.ReLU()),
            ('Pool4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),  # 128*7*7
        ]))
        self.c5 = nn.Sequential(OrderedDict([
            ('Conv5', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)),  # 256*7*7
            ('Relu5', nn.ReLU()),
        ]))
        self.c6 = nn.Sequential(OrderedDict([
            ('Conv6', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)),  # 256*7*7
            ('Relu6', nn.ReLU()),
        ]))
        self.c7 = nn.Sequential(OrderedDict([
            ('Conv7', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)),  # 256*7*7
            ('Relu7', nn.ReLU()),
            ('Pool7', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),  # 256*3*3
        ]))
        # self.c8 = nn.Sequential(OrderedDict([
        #     ('Conv8', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1)),  # 512*3*3
        #     ('Relu8', nn.ReLU()),
        # ]))
        # self.c9 = nn.Sequential(OrderedDict([
        #     ('Conv9', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)),  # 512*3*3
        #     ('Relu9', nn.ReLU()),
        # ]))
        # self.c10 = nn.Sequential(OrderedDict([
        #     ('Conv10', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)),  # 512*3*3
        #     ('Relu10', nn.ReLU()),
        #     ('Pool10', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),  # 512*1*1
        # ]))
        # self.c11 = nn.Sequential(OrderedDict([
        #     ('Conv11', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)),  # 512*2*2
        #     ('Relu11', nn.ReLU()),
        # ]))
        # self.c12 = nn.Sequential(OrderedDict([
        #     ('Conv12', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)),  # 512*2*2
        #     ('Relu12', nn.ReLU()),
        # ]))
        # self.c13 = nn.Sequential(OrderedDict([
        #     ('Conv13', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)),  # 512*2*2
        #     ('Relu13', nn.ReLU()),
        #     ('Pool13', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),  # 512*2*2
        # ]))
        self.c14 = nn.Sequential(OrderedDict([
            ('FullCon14', nn.Linear(in_features=256*3*3, out_features=512)),
            ('Relu14', nn.ReLU()),
            ('Drop14', nn.Dropout(p=0.5)),
        ]))
        self.c15 = nn.Sequential(OrderedDict([
            ('FullCon15', nn.Linear(in_features=512, out_features=128)),
            ('Relu15', nn.ReLU()),
            ('Drop15', nn.Dropout(p=0.5)),
        ]))
        self.c16 = nn.Sequential(OrderedDict([
            ('FullCon16', nn.Linear(in_features=128, out_features=10)),
            ('Sig16', nn.LogSoftmax(dim=-1)),
        ]))

    def forward(self, img):
        output = self.c1(img)
        output = self.c2(output)
        output = self.c3(output)
        output = self.c4(output)
        output = self.c5(output)
        output = self.c6(output)
        output = self.c7(output)
        # output = self.c8(output)
        # output = self.c9(output)
        # output = self.c10(output)
        # output = self.c11(output)
        # output = self.c12(output)
        # output = self.c13(output)

        output = torch.flatten(output, 1)
        output = self.c14(output)
        output = self.c15(output)
        output = self.c16(output)

        return output
