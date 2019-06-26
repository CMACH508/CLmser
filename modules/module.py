# coding=utf-8
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair
from torch._jit_internal import weak_module, weak_script_method, List


class _ConvNd(Module):

    def __init__(self, kernel, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.weight = kernel
        if bias:
            self.bias = Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


@weak_module
class _ConvTransposeMixin(object):
    __constants__ = ['stride', 'padding', 'kernel_size', 'dim_size',
                     'output_padding', 'groups', 'dilation', 'transposed', 'bias']

    @weak_script_method
    def forward(self, input, output_size=None):
        # type(Tensor, Optional[List[int]]) -> Tensor
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        func = self._backend.ConvNd(
            self.stride, self.padding, self.dilation, self.transposed,
            output_padding, self.groups)
        if self.bias is None:
            return func(input, self.weight)
        else:
            return func(input, self.weight, self.bias)

    @weak_script_method
    def _output_padding(self, input, output_size, stride, padding, kernel_size):
        # type: (Tensor, Optional[List[int]], List[int], List[int], List[int]) -> List[int]
        if output_size is None:
            ret = _single(self.output_padding)  # converting to list if was not already
        else:
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


@weak_module
class SingleDeconvLayer(_ConvTransposeMixin, _ConvNd):

    def __init__(self, kernel, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        in_channels, out_channels, kernel_size = kernel.shape[1], kernel.shape[0], kernel.shape[2]
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(SingleDeconvLayer, self).__init__(
            kernel, in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias)

    @weak_script_method
    def forward(self, input, output_size=None):
        # type: (Tensor, Optional[List[int]]) -> Tensor
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        return F.conv_transpose2d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
