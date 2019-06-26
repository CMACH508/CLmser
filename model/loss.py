import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self, image, fake):
        loss = self.mse_loss(image, fake)
        return loss
