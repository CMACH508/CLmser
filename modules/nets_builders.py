import torch.nn as nn
import torch.nn.functional as F
import torch
from modules.jit import List
from modules.util import _pair


def get_activattion(activation):
    if activation == 'ReLU':
        return nn.ReLU(inplace=True)
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.2, inplace=True)
    elif activation == 'Tanh':
        return nn.Tanh()
    else:
        raise ValueError


class ConvBlock(nn.Module):
    def __init__(self, weight, bias=None, stride=1, padding=0,
                 activation=None, norm=False):
        super(ConvBlock, self).__init__()
        self.conv2d = Conv2D(weight, bias, stride=stride, padding=padding)
        if norm:
            self.bn = nn.BatchNorm2d(weight.shape[0])
        if activation is not None:
            self.activation = get_activattion(activation)

    def forward(self, inputs):
        x = self.conv2d(inputs)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        if hasattr(self, 'activation'):
            x = self.activation(x)
        return x


class DeConvBlock(nn.Module):
    def __init__(self, weight, bias=None, stride=1, padding=0,
                 activation=None, norm=False):
        super(DeConvBlock, self).__init__()
        self.dconv2d = DeConv2D(weight, bias, stride=stride, padding=padding)
        if norm:
            self.bn = nn.BatchNorm2d(weight.shape[1])
        if activation is not None:
            self.activation = get_activattion(activation)

    def forward(self, inputs, output_size):
        x = self.dconv2d(inputs, output_size)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        if hasattr(self, 'activation'):
            x = self.activation(x)
        return x


class Conv2D(nn.Module):
    def __init__(self, weight, bias=None, stride=1, padding=0):
        super(Conv2D, self).__init__()
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding

    def forward(self, inputs):
        x = F.conv2d(inputs, self.weight, bias=self.bias,
                     stride=self.stride, padding=self.padding)
        return x


class DeConv2D(nn.Module):
    def __init__(self, weight, bias=None, stride=1, padding=0):
        super(DeConv2D, self).__init__()
        self.weight = weight
        self.bias = bias
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.kernel_size = _pair(weight.shape[2])

    def forward(self, inputs, output_size):
        output_padding = self._output_padding(inputs, output_size,
                                              self.stride, self.padding,
                                              self.kernel_size)
        x = F.conv_transpose2d(
            inputs, self.weight, self.bias, self.stride, self.padding,
            output_padding)
        return x

    def _output_padding(self, input, output_size, stride, padding, kernel_size):
        output_size = torch.jit._unwrap_optional(output_size)
        k = input.dim() - 2
        if len(output_size) == k + 2:
            output_size = output_size[2:]
        if len(output_size) != k:
            raise ValueError(
                "output_size must have {} or {} elements (got {})"
                    .format(k, k + 2, len(output_size)))

        min_sizes = torch.jit.annotate(List[int], [])
        max_sizes = torch.jit.annotate(List[int], [])
        for d in range(k):
            dim_size = ((input.size(d + 2) - 1) * stride[d] -
                        2 * padding[d] + kernel_size[d])
            min_sizes.append(dim_size)
            max_sizes.append(min_sizes[d] + stride[d] - 1)

        for i in range(len(output_size)):
            size = output_size[i]
            min_size = min_sizes[i]
            max_size = max_sizes[i]
            if size < min_size or size > max_size:
                raise ValueError((
                    "requested an output size of {}, but valid sizes range "
                    "from {} to {} (for an input of {})").format(
                    output_size, min_sizes, max_sizes, input.size()[2:]))

        res = torch.jit.annotate(List[int], [])
        for d in range(k):
            res.append(output_size[d] - min_sizes[d])

        ret = res
        return ret


