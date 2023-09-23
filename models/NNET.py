import torch
import torch.nn as nn
import torch.nn.functional as F

from models.submodules.encoder import Encoder
from models.submodules.decoder import Decoder
import todos
import pdb

class NNET(nn.Module):
    def __init__(self, args):
        super(NNET, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(args)

        # args = Namespace(architecture='GN', pretrained='nyu', 
        #     sampling_ratio=0.4, importance_ratio=0.7, input_height=480, 
        #     input_width=640, imgs_dir='./examples')


    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        return self.decoder.parameters()

    def forward(self, img):
        # kwargs -- {}
        # tensor [img] size: [1, 3, 480, 640] , min: -2.1179 , max: 2.6400, mean: -0.2764998

        e = self.encoder(img)
        # len(e) -- 16
        # (Pdb) e[0].size() -- [1, 3, 480, 640]
        # (Pdb) e[1].size() -- [1, 48, 240, 320]
        # (Pdb) e[2].size() -- [1, 48, 240, 320]
        # (Pdb) e[15].size() -- [1, 2048, 15, 20] 

        ret = self.decoder(e)
        # tensor [ret] size: [1, 4, 480, 640] , min: -0.9968838095664978 , max: 29.235084533691406 mean: 3.119987726211548

        return ret