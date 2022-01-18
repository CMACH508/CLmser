import torch.nn.functional as F
import torch.nn as nn
import torch


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse_loss = nn.MSELoss()


