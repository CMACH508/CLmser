import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.module import SingleDeconvLayer
from utils import select_act_func


class LinearLayer(nn.Module):
    def __init__(self, weights, bias, activation=None, is_norm=False):
        super(LinearLayer, self).__init__()
        self.weights = weights
        self.bias = bias
        self.activation = select_act_func(activation)
        if is_norm:
            self.bn = nn.BatchNorm1d(weights.shape[1])

    def forward(self, inputs, label=None):
        if label is None:
            x = torch.matmul(inputs, self.weights) + self.bias
        else:
            x = torch.matmul(label, self.weights) + self.bias
        if hasattr(self, 'bn'):
            x = self.bn(x)
        x = self.activation(x)
        return x


class SingleConvLayer(nn.Module):
    def __init__(self, kernel, stride=1, padding=1, groups=1, dilation=1, is_bias=True):
        super(SingleConvLayer, self).__init__()
        self.kernel = kernel
        self.out_channels = np.shape(kernel)[0]
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        if is_bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.out_channels))
            nn.init.constant_(self.bias, 0.0)
        else:
            self.bias = None

    def forward(self, inputs):
        x = nn.functional.conv2d(inputs, self.kernel, bias=self.bias,
                                 stride=self.stride, padding=self.padding, dilation=self.dilation)
        return x


class ConvBlock(nn.Module):
    def __init__(self, kernel, stride, padding=1, dilation=1, groups=1,
                 is_bias=True, activation=None, is_norm=False):
        super(ConvBlock, self).__init__()
        self.activation = select_act_func(activation)
        self.conv = SingleConvLayer(kernel, stride, padding, dilation, groups, is_bias)
        if is_norm:
            self.bn = nn.InstanceNorm2d(kernel.shape[0])

    def forward(self, x_top, x_old, is_skip):
        if is_skip:
            x = self.conv(x_top + x_old)
        else:
            x = self.conv(x_top)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        x = self.activation(x)
        return x


class DeconvBlock(nn.Module):
    def __init__(self, kernel, stride, padding=1, output_padding=0, dilation=1, groups=1,
                 is_bias=True, activation=None, is_norm=False):
        super(DeconvBlock, self).__init__()
        self.out_channel = np.shape(kernel)[1]
        if is_norm:
            self.bn = nn.InstanceNorm2d(self.out_channel)
        self.activation = select_act_func(activation)
        self.deconv = SingleDeconvLayer(kernel, stride, padding, output_padding, groups, is_bias, dilation)

    def forward(self, x_top, x_old, shape, is_skip):

        if is_skip:
            x_top = x_top + x_old
        x = self.deconv(x_top, shape)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        x = self.activation(x)
        return x


class BottomBlock(nn.Module):
    def __init__(self, kernel, stride, padding=1, output_padding=0, dilation=1, groups=1,
                 is_bias=True, activation=None, is_norm=False):
        super(BottomBlock, self).__init__()
        self.out_channel = np.shape(kernel)[1]
        if is_norm:
            self.bn = nn.InstanceNorm2d(self.out_channel)
        self.activation = select_act_func(activation)
        self.deconv = SingleDeconvLayer(kernel, stride, padding, output_padding, groups, is_bias, dilation)

    def forward(self, x_top, x_old, shape, is_skip):
        if is_skip:
            x_top = x_top + x_old
        x = self.deconv(x_top, shape)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        x = self.activation(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        self.channels = channels
        self.weights = nn.Parameter(torch.FloatTensor(channels, channels, 1, 1))
        # self.weights = self.weights.expand(1, 1, channels, channels)
        torch.nn.init.xavier_uniform(self.weights)

    def forward(self, inputs):
        x = nn.functional.conv2d(inputs, self.weights)
        return x


class MapAttention(nn.Module):
    def __init__(self, channels):
        super(MapAttention, self).__init__()
        self.channels = channels
        self.weights = nn.Parameter(torch.FloatTensor(1, 1, 3, 3))
        # self.weights = self.weights.expand(1, 1, channels, channels)
        torch.nn.init.xavier_uniform(self.weights)

    def forward(self, inputs):
        batch_size, channel, h, w = inputs.shape
        inputs = torch.reshape(inputs, [-1, h, w])
        inputs = torch.unsqueeze(inputs, 1)
        x = nn.functional.conv2d(inputs, self.weights, padding=1)
        x = torch.reshape(x, [batch_size, channel, h, w])
        return x
