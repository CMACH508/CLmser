import torch
import torch.nn.functional as F
from torch import nn


class PartialConv2d(nn.Module):
    def __init__(self, kernel, bias=None, stride=1, padding=0):

        # whether the mask is multi-channel or not

        super(PartialConv2d, self).__init__()

        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.out_channels, self.in_channels, self.kernel_size, _ = kernel.shape

        self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size,
                                                 self.kernel_size)

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, inputs, mask=None):

        if mask is not None or self.last_size != (inputs.data.shape[2], inputs.data.shape[3]):
            self.last_size = (inputs.data.shape[2], inputs.data.shape[3])

            with torch.no_grad():
                if self.weight_maskUpdater.type() != inputs.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(inputs)

                if mask is None:
                    # if mask is not provided, create a mask

                    mask = torch.ones(inputs.data.shape[0], inputs.data.shape[1], inputs.data.shape[2],
                                          inputs.data.shape[3]).to(inputs)

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding)

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        if self.update_mask.type() != inputs.type() or self.mask_ratio.type() != inputs.type():
            self.update_mask.to(inputs)
            self.mask_ratio.to(inputs)

        raw_out = nn.functional.conv2d(torch.mul(inputs, mask) if mask is not None else inputs, self.kernel,
                                       bias=self.bias,
                                       stride=self.stride,
                                       padding=self.padding)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)
        return output, self.update_mask


