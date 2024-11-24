import os
import lmdb
import yaml
import math
import socket
import argparse
import numpy as np
from io import BytesIO
from PIL import Image
from glob import glob
from tqdm import tqdm
from random import randrange

import torch
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 10 * (1024**2)


def safe_randrange(low, high):
    if low == high:
        return low
    else:
        return randrange(low, high)


class DictTensor(dict):

    def to(self, device):
        new_self = DictTensor()
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                new_self[k] = v.to(device)
            else:
                new_self[k] = v
        return new_self

    def cpu(self):
        new_self = DictTensor()
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                new_self[k] = v.cpu()
            else:
                new_self[k] = v
        return new_self

    def detach(self):
        new_self = DictTensor()
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                new_self[k] = v.detach()
            else:
                new_self[k] = v
        return new_self

    def get_device(self):
        return list(self.values())[0].device

    def __setattr__(self, attr, value):
        if attr == "requires_grad":
            for v in self.values():
                # Note: Tensor with non-float type cannot requires grad
                if isinstance(v, torch.Tensor) and v.dtype not in {torch.int32, torch.int64}:
                    v.requires_grad = value
            #for v in self.attrs.values():
            #    v.requires_grad = value
        else:
            super().__setattr__(attr, value)
