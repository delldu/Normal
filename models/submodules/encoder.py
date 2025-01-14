import torch
import torch.nn as nn
import torch.nn.functional as F
from .gen_efficientnet import GenEfficientNet
import pdb


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.original_model = GenEfficientNet() # base model -- tf_efficientnet_b5_ap

    def forward(self, x):
        return self.original_model(x)
