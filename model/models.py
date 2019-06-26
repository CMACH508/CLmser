import torch
import torch.nn as nn
import torch.nn.init
from base import BaseModel
from modules.lmser_parts import ConvBlock, DeconvBlock, LinearLayer, BottomBlock


class CLmser_w(BaseModel):
    def __init__(self, depth, kernels, channels, strides, h_down, h_top):

        super(CLmser_w, self).__init__()
        self.ekernels = nn.ParameterList()
        self.dkernels = nn.ParameterList()

        self.weights_up = nn.Parameter(torch.FloatTensor(h_top, h_down))
        self.weights_down = nn.Parameter(torch.FloatTensor(h_down, h_top))
        self.bias_up = nn.Parameter(torch.FloatTensor(h_down))
        self.bias_down = nn.Parameter(torch.FloatTensor(h_top))
        nn.init.xavier_uniform(self.weights_up)
        nn.init.xavier_uniform(self.weights_down)
        nn.init.constant_(self.bias_up, 0.0)
        nn.init.constant_(self.bias_down, 0.0)

        for i, kernel in zip(range(depth), kernels):
            self.ekernels.append(nn.Parameter(torch.FloatTensor(channels[i + 1], channels[i], kernel, kernel)))
            self.dkernels.append(nn.Parameter(torch.FloatTensor(channels[i + 1], channels[i], kernel, kernel)))
            torch.nn.init.xavier_uniform(self.ekernels[i])
            torch.nn.init.xavier_uniform(self.dkernels[i])

        for i in range(0, depth):
            downlayer = ConvBlock(self.ekernels[i], strides[i], is_norm=True, activation='relu')
            self.add_module('down%d' % (i + 1), downlayer)
            uplayer = DeconvBlock(self.dkernels[i], strides[i], is_norm=True, activation='relu')
            self.add_module('up%d' % (i + 1), uplayer)

        self.linear_up = LinearLayer(self.weights_up, self.bias_up,
                                     is_norm=True, activation='relu')
        self.linear_down = LinearLayer(self.weights_down, self.bias_down,
                                       is_norm=False, activation=None)

        self.out = DeconvBlock(self.dkernels[0], strides[0],
                               is_norm=False, activation='tanh')

    def forward(self, inputs, label=None):

        x1 = self.down1(inputs, 0, 0)
        x2 = self.down2(x1, 0, 0)
        x3 = self.down3(x2, 0, 0)
        x4 = self.down4(x3, 0, 0)
        x5 = self.down5(x4, 0, 0)
        x6 = self.down6(x5, 0, 0)

        # x = x6.reshape(x6.shape[0], -1)
        # logits = self.linear_down(x)
        # x = self.linear_up(logits, label)
        # x = x.reshape_as(x6)

        x21 = self.up6(x6, x6, x5.size(), 1)
        x22 = self.up5(x21, x5, x4.size(), 1)
        x23 = self.up4(x22, x4, x3.size(), 1)
        x24 = self.up3(x23, x3, x2.size(), 1)
        x25 = self.up2(x24, x2, x1.size(), 1)
        x26 = self.out(x25, x1, inputs.size(), 1)
        return 0, x26


class CLmser(BaseModel):
    def __init__(self, depth, kernels, channels, strides, h_down, h_top):

        super(CLmser, self).__init__()

        self.weights = nn.Parameter(torch.FloatTensor(h_down, h_top))
        self.bias_up = nn.Parameter(torch.FloatTensor(h_down))
        self.bias_down = nn.Parameter(torch.FloatTensor(h_top))
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.bias_up, 0.0)
        nn.init.constant_(self.bias_down, 0.0)

        self.kernels = nn.ParameterList()

        for i, kernel in zip(range(depth), kernels):
            self.kernels.append(nn.Parameter(torch.FloatTensor(channels[i + 1], channels[i], kernel, kernel)))
            torch.nn.init.xavier_uniform_(self.kernels[i])

        for i, kernel, stride in zip(range(depth), self.kernels, strides):
            downlayer = ConvBlock(kernel, stride, is_norm=True, activation='relu')
            self.add_module('down%d' % (i + 1), downlayer)

        for i, kernel, stride in zip(range(1, depth), self.kernels[1:], strides[1:]):
            uplayer = DeconvBlock(kernel, stride, is_norm=True, activation='relu')
            self.add_module('up%d' % (i + 1), uplayer)

        self.linear_up = LinearLayer(nn.Parameter(self.weights.permute([1, 0])), self.bias_up,
                                     is_norm=True, activation='relu')
        self.linear_down = LinearLayer(self.weights, self.bias_down,
                                       is_norm=False, activation=None)
        self.out = BottomBlock(self.kernels[0], strides[0], is_norm=False, activation='tanh')

    def forward(self, inputs, label=None):

        x1 = self.down1(inputs, 0, 0)
        x2 = self.down2(x1, 0, 0)
        x3 = self.down3(x2, 0, 0)
        x4 = self.down4(x3, 0, 0)
        x5 = self.down5(x4, 0, 0)
        x6 = self.down6(x5, 0, 0)

        # if not full-cnn
        # x = x6.reshape(x6.shape[0], -1)
        # logits = self.linear_down(x)
        # x = self.linear_up(logits, label)
        # x = x.reshape_as(x6)

        x21 = self.up6(x6, x6, x5.size(), 1)
        x22 = self.up5(x21, x5, x4.size(), 1)
        x23 = self.up4(x22, x4, x3.size(), 1)
        x24 = self.up3(x23, x3, x2.size(), 1)
        x25 = self.up2(x24, x2, x1.size(), 1)
        x26 = self.out(x25, x1, inputs.size(), 1)
        return 0, x26
