""" Generic Efficient Networks
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch.nn as nn
import torch.nn.functional as F

from .conv2d_layers import select_conv2d
from .efficientnet_builder import (
    SqueezeExcite,
    DepthwiseSeparableConv,
    EfficientNetBuilder,
    BN_EPS_TF_DEFAULT,
    decode_arch_def,
    round_channels,
    resolve_act_layer,
    resolve_bn_args,
)
import todos
import pdb


class GenEfficientNet(nn.Module):
    """ Generic EfficientNets
    """

    def __init__(self, num_classes=1000, in_chans=3, 
                 num_features=2048, stem_size=32,
                 channel_multiplier=1.6, depth_multiplier=2.2,
                 channel_divisor=8, channel_min=None,
                 pad_type='same', act_layer=nn.SiLU, drop_connect_rate=0.,
                 se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs={'eps': 0.001},
                ):
        super(GenEfficientNet, self).__init__()
        kwargs = {}
        kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
        kwargs['pad_type'] = pad_type
        arch_def = [
            ['ds_r1_k3_s1_e1_c16_se0.25'],
            ['ir_r2_k3_s2_e6_c24_se0.25'],
            ['ir_r2_k5_s2_e6_c40_se0.25'],
            ['ir_r3_k3_s2_e6_c80_se0.25'],
            ['ir_r3_k5_s1_e6_c112_se0.25'],
            ['ir_r4_k5_s2_e6_c192_se0.25'],
            ['ir_r1_k3_s1_e6_c320_se0.25'],
        ]
        block_args=decode_arch_def(arch_def, depth_multiplier)
        num_features=round_channels(1280, channel_multiplier, 8, None)
        act_layer=resolve_act_layer(kwargs, 'swish')
        norm_kwargs=resolve_bn_args(kwargs)

        stem_size = round_channels(stem_size, channel_multiplier, channel_divisor, channel_min)
        self.conv_stem = select_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        in_chs = stem_size

        builder = EfficientNetBuilder(
            channel_multiplier, channel_divisor, channel_min,
            pad_type, act_layer, se_kwargs, norm_layer, norm_kwargs, drop_connect_rate)

        self.blocks = nn.Sequential(*builder(in_chs, block_args))
        in_chs = builder.in_chs

        self.conv_head = select_conv2d(in_chs, num_features, 1, padding=pad_type)
        self.bn2 = norm_layer(num_features, **norm_kwargs)
        self.act2 = act_layer(inplace=True)

    def forward(self, x):
        outs = [x]

        x = self.conv_stem(x)
        outs.append(x)
        x = self.bn1(x)
        outs.append(x)
        x = self.act1(x)
        outs.append(x)

        # x = self.blocks(x)
        for layer in self.blocks:
            x = layer(x)
            outs.append(x)

        x = self.conv_head(x)
        outs.append(x)
        x = self.bn2(x)
        outs.append(x)
        x = self.act2(x)
        outs.append(x)

        return outs
