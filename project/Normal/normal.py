import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder
import pdb

class NNET(nn.Module):
    def __init__(self, arch):
        super(NNET, self).__init__()
        self.MAX_H = 1024
        self.MAX_W = 1024
        self.MAX_TIMES = 8

        self.encoder = Encoder()
        self.decoder = Decoder(arch)

        self.load_weights()

    def forward(self, x):
        B, C, H, W = x.size()

        # Need Pad ?
        if H % self.MAX_TIMES != 0 or W % self.MAX_TIMES != 0:
            r_pad = self.MAX_TIMES - (W % self.MAX_TIMES)
            b_pad = self.MAX_TIMES - (H % self.MAX_TIMES)
            img = F.pad(x, (0, r_pad, 0, b_pad), mode="replicate")
        else:
            img = x

        # tensor [img] size: [1, 3, 480, 640] , min: -2.1179 , max: 2.6400, mean: -0.2764998

        e = self.encoder(img)
        # len(e) -- 16
        # e[0].size() -- [1, 3, 480, 640]
        # e[1].size() -- [1, 48, 240, 320]
        # e[2].size() -- [1, 48, 240, 320]
        # e[15].size() -- [1, 2048, 15, 20] 

        ret = self.decoder(e)
        # tensor [ret] size: [1, 4, 480, 640] , min: -0.9968838095664978 , max: 29.235084533691406 mean: 3.119987726211548
        ret = ret[:, 0:3, 0:H, 0:W] # remove pads and normal alpha

        return (ret + 1.0)/2.0 # convert data from [-1.0, 1.0] to [0.0, 1.0]


    def load_weights(self, model_path="models/Normal.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path

        if os.path.exists(checkpoint):
            print(f"Loading weight from {checkpoint} ...")
            self.load_state_dict(torch.load(checkpoint)['model'])
        else:
            print("-" * 32, "Warnning", "-" * 32)
            print(f"Weight file '{checkpoint}' not exist !!!")
