import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from torchvision import models, utils

from arcface.iresnet import *


class fs_encoder_v2(nn.Module):

    def __init__(self,
                 opts=None,
                 residual=False,
                 use_coeff=False,
                 resnet_layer=None,
                 video_input=False,
                 f_maps=512,
                 stride=(1, 1)):
        super(fs_encoder_v2, self).__init__()

        resnet50 = iresnet50()
        resnet50.load_state_dict(torch.load(opts.arcface_model_path))

        # input conv layer
        if video_input:
            self.conv = nn.Sequential(
                nn.Conv2d(6, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                *list(resnet50.children())[1:3])
        else:
            self.conv = nn.Sequential(*list(resnet50.children())[:3])

        # define layers
        self.block_1 = list(resnet50.children())[3]  # 15-18
        self.block_2 = list(resnet50.children())[4]  # 10-14
        self.block_3 = list(resnet50.children())[5]  # 5-9
        self.block_4 = list(resnet50.children())[6]  # 1-4

        self.avg_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.styles = nn.ModuleList()
        self.styles = nn.Linear(960 * 9, 512)

    def forward(self, x):
        latents = []
        features = []
        x = self.conv(x)
        x = self.block_1(x)
        features.append(self.avg_pool(x))
        x = self.block_2(x)
        features.append(self.avg_pool(x))
        x = self.block_3(x)
        features.append(self.avg_pool(x))
        x = self.block_4(x)
        features.append(self.avg_pool(x))
        x = torch.cat(features, dim=1)
        x = x.view(x.size(0), -1)
        out = self.styles(x)
        return out
