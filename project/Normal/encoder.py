import torch
import torch.nn as nn
from .gen_efficientnet import GenEfficientNet
from typing import List

import pdb

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.original_model = GenEfficientNet() # base model -- tf_efficientnet_b5_ap

    def forward(self, x) -> List[torch.Tensor]:
        return self.original_model(x)
