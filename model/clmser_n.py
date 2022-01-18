import torch.nn as nn
import torch
from modules.nets_builders import ConvBlock, DeConvBlock

cfg = {
    'STRIDE': [1, 2, 2, 2, 2, 2, 2],
    'CHANNEL': [16, 32, 64, 128, 256, 256, 256],
    'DECODER': [2, 2, 2, 2, 2, 2],
}


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.kernels = self.create_weights(cfg['CHANNEL'], 3)
        self.DownBlocks = self._make_layers(ConvBlock, self.kernels, cfg['STRIDE'], padding=1,
                                            activation='ReLU', norm=True)
        self.UpBlocks = self._make_layers(DeConvBlock, self.kernels[1:], cfg['STRIDE'][1:], padding=1,
                                          activation='ReLU', norm=True)
        self.OutBlock = DeConvBlock(self.kernels[0], stride=1, padding=1,
                                    activation='Tanh', norm=False)

    def forward(self, inputs):
        depth = len(self.DownBlocks)
        features = []
        x = inputs
        for i in range(depth):
            x = self.DownBlocks.__getitem__(i)(x)
            features.append(x)

        x = features[-1]

        for i in range(depth - 2, -1, -1):
            output_size = [x.shape[2] * 2, x.shape[3] * 2]
            x = self.UpBlocks.__getitem__(i)(x, output_size)

        output = self.OutBlock(x, inputs.shape[2:])

        return output

    def create_weights(self, cfg, in_channels):
        kernels = []
        for x in cfg:
            weights = nn.Parameter(torch.FloatTensor(x, in_channels, 3, 3))
            nn.init.xavier_uniform_(weights)
            kernels += [weights]
            in_channels = x
        return kernels

    def _make_layers(self, block, weights, stride, padding, activation, norm):
        layers = []
        for (x, stride) in zip(weights, stride):
            layers += [block(x, stride=stride, padding=padding,
                             activation=activation, norm=norm)]
        return nn.Sequential(*layers)
