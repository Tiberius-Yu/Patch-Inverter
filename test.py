# breakpoint()
print("SET breakpoint freely")

import argparse

import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import yaml

from PIL import Image
from tqdm import tqdm
from easydict import EasyDict
from torchvision import transforms, utils

from utils.datasets import *
from utils.functions import *
from trainer import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None
device = torch.device('cuda')

attr_list = [
    'aurora', 'beach', 'bridge', 'canyon', 'cliff', 'forest', 'fountain', 'glacier', 'hayfield',
    'lake', 'lighthouse', 'maple', 'meteor_shower', 'mountain', 'ocean', 'sakura', 'snowfield',
    'storm', 'sunrise', 'sunset', 'valley', 'waterfall', 'wave', 'wisteria'
]


def choose_dataset(config, train):
    if "Land" in config:
        return MultiResolutionDataset
    else:
        return MyDataSet if train else TestDataSet


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='001', help='Path to the config file.')
parser.add_argument('--model_config', type=str, default='001', help='Path to the config file.')
parser.add_argument(
    '--pretrained_model_path',
    type=str,
    default='./pretrained_models/143_enc.pth',
    help='pretrained stylegan2 model')
parser.add_argument(
    '--stylegan_model_path',
    type=str,
    default='./pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt',
    help='pretrained stylegan2 model')
parser.add_argument(
    '--arcface_model_path',
    type=str,
    default='./pretrained_models/backbone.pth',
    help='pretrained arcface model')
parser.add_argument(
    '--parsing_model_path',
    type=str,
    default='./pretrained_models/79999_iter.pth',
    help='pretrained parsing model')
parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file path')
parser.add_argument('--checkpoint_noiser', type=str, default='', help='checkpoint file path')
parser.add_argument('--multigpu', type=bool, default=False, help='use multiple gpus')
parser.add_argument('--input_path', type=str, default='./test/', help='evaluation data file path')
parser.add_argument('--scale', type=float, default=15.0, help='editing step')
parser.add_argument('--localscale', type=float, default=15.0, help='editing step')
parser.add_argument('--attribute', type=str, default="aurora", help='editing step')
parser.add_argument(
    '--save_path', type=str, default='./output/image/', help='output data save path')

parser.add_argument(
    '--moco_model_path',
    type=str,
    default=
    'pretrained_models/moco_v2_800ep_pretrain.pt',
    help='pretrained arcface model')
opts = parser.parse_args()

log_dir = os.path.join(opts.log_path, opts.config) + '/'
with open('./configs/' + opts.config + '.yaml', 'r', encoding="UTF-8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
with open('./configs/' + opts.model_config + '.yaml', 'r', encoding="UTF-8") as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)
    model_config = EasyDict(model_config)
    model_config.var = EasyDict()
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        n_gpu = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        model_config.var.dataparallel = n_gpu > 1
        model_config.var.n_gpu = n_gpu
        if n_gpu > 1:
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.benchmark = True
    else:
        raise ValueError(" [!] Please specify CUDA_VISIBLE_DEVICES!")
config['model'] = model_config

batch_size = config['batch_size']
epochs = config['epochs']
iter_per_epoch = config['iter_per_epoch']
img_size = (config['resolution'], config['resolution'])
VIDEO_DATA_INPUT = False

img_to_tensor = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
img_to_tensor_car = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Initialize trainer
trainer = Trainer(config, opts, opts.save_path)
trainer.initialize(opts.stylegan_model_path, opts.moco_model_path, opts.parsing_model_path)
trainer.to(device)

state_dict = torch.load(opts.checkpoint)
trainer.globEnc.load_state_dict(state_dict['global_enc_state_dict'])
trainer.locEnc.load_state_dict(state_dict['local_enc_state_dict'])
trainer.randomized_noises = state_dict['random_noise']

img_to_tensor = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

batch_size = config['batch_size']
epochs = config['epochs']
iter_per_epoch = config['iter_per_epoch']
img_size = (config['resolution'], config['resolution'])
VIDEO_DATA_INPUT = False

# simple inference
image_dir = opts.input_path
save_dir = opts.save_path
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, "recon"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "edit"), exist_ok=True)
print(f"[*] savedir is {save_dir}")

bound = None
if opts.attribute != "None":
    bound = torch.Tensor(
        np.load(f"boundaries/{opts.attribute}_boundary.npy")
    ).cuda() * opts.scale
# local_bound = torch.Tensor(
#     np.load(f"boundaries-test/{opts.attribute}_boundary.npy")
# ).cuda()
# for attr in attr_list:
#     globals()[f"{attr}_bound"] = torch.Tensor(
#         np.load(f"boundaries/{attr}_boundary.npy")).cuda()

which_dataset = choose_dataset(opts.config, False)
dataset_test = which_dataset(
    path=image_dir, resolution=img_size, test_num=3000, train=False, split=0.9)
loader_test = data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
iter_test = iter(loader_test)

with torch.no_grad():
    print(f"[INFO]: Testing | Using {dataset_test.length} images")
    trainer.globEnc.eval()
    trainer.locEnc.eval()
    cnt = 0
    flag = True
    while flag:
        print(f"[INFO]: | Evaluating {cnt}th batch |", end="\r")
        try:
            img_test = next(iter_test)
        except StopIteration:
            break
        # img_test = img_test.to(device)
        # if cnt < 1000:
        #     cnt += 1
        #     continue
        trainer.test(img=img_test, iter=cnt, logdir=save_dir, bound=bound, name=str(opts.attribute))
        # trainer.log_test_loss()
        cnt += 1
        # if cnt > 4999: break
    trainer.compute_test_loss()
