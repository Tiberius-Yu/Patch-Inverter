"""
Filename: train.py
Author: leafy
"""

# breakpoint()

import argparse
import os
import shutil
import yaml
import time
import glob

import torch
import torch.utils.data as data

from utils.datasets import MyDataSet, TestDataSet, MultiResolutionDataset
from utils.functions import clip_img
from easydict import EasyDict
from trainer import *

from PIL import Image
from tqdm import tqdm
from torchvision import transforms, utils
from tensorboard_logger import Logger

current_path = os.getcwd()
print("当前路径：" + current_path)


def choose_dataset(config, train):
    if "Land" in config:
        return MultiResolutionDataset
    else:
        return MyDataSet if train else TestDataSet


torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None
device = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='001', help='Path to the config file.')
parser.add_argument(
    '--model_config', type=str, default='InfinityGAN', help='Path to the config file.')

parser.add_argument(
    '--train_dataset_path',
    type=str,
    default='DATA/stanfordcar/cars_train/',
    help='dataset path')
parser.add_argument(
    '--test_dataset_path',
    type=str,
    default='DATA/stanfordcar/cars_train/',
    help='dataset path')
parser.add_argument(
    '--dataset_path',
    type=str,
    default='DATA/car-stylegan2-generate-images/ims/',
    help='dataset path')
parser.add_argument(
    '--label_path',
    type=str,
    default='DATA/car-stylegan2-generate-images/seeds_pytorch_1.8.1.npy',
    help='laebl path')
parser.add_argument(
    '--stylegan_model_path',
    type=str,
    default=
    'experiment/2023/pixel2style2pixel-master/pretrained_models/stylegan2-car-config-f.pt',
    help='pretrained stylegan2 model')
parser.add_argument(
    '--arcface_model_path',
    type=str,
    default='./pretrained_models/backbone.pth',
    help='pretrained arcface model')
parser.add_argument(
    '--moco_model_path',
    type=str,
    default=
    'experiment/2023/pixel2style2pixel-master/pretrained_models/moco_v2_800ep_pretrain.pt',
    help='pretrained arcface model')
parser.add_argument(
    '--parsing_model_path',
    type=str,
    default='./pretrained_models/79999_iter.pth',
    help='pretrained parsing model')
parser.add_argument(
    '--log_path', type=str, default='PatchInv', help='log file path')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file path')
opts = parser.parse_args()

VERB = True

print(opts)
log_dir = os.path.join(opts.log_path,
                       opts.config) + f"_resume_{(opts.resume)}_{'_'.join(time.asctime().split())}/"
os.makedirs(log_dir, exist_ok=True)
shutil.copytree(current_path, os.path.join(log_dir, "code"))
logger = Logger(log_dir)

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
trainer = Trainer(config, opts, log_dir)
trainer.initialize(opts.stylegan_model_path, opts.moco_model_path, opts.parsing_model_path)
trainer.to(device)

noise_exemple = None
train_data_split = 0.9 if 'train_split' not in config else config['train_split']

# Load synthetic dataset
# dataset_A = MyDataSet(
#     image_dir=opts.dataset_path,
#     label_dir=None,
#     output_size=img_size,
#     noise_in=noise_exemple,
#     training_set=True,
#     train_split=1)
# loader_A = data.DataLoader(
#     dataset_A, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
# Load real dataset

which_dataset = choose_dataset(opts.config, train=True)
dataset_B = which_dataset(path=opts.train_dataset_path, resolution=img_size, train=True, split=0.9)
loader_B = data.DataLoader(
    dataset_B, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

which_dataset = choose_dataset(opts.config, train=False)
dataset_test = which_dataset(
    path=opts.test_dataset_path, resolution=img_size, test_num=20, train=False, split=0.9)
loader_test = data.DataLoader(
    dataset_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
# breakpoint()

EPOCH_0 = 0
# check if checkpoint exist
if 'checkpoint.pth' in os.listdir(log_dir):
    EPOCH_0 = trainer.load_checkpoint(os.path.join(log_dir, 'checkpoint.pth'))

if opts.resume:
    EPOCH_0 = trainer.load_checkpoint(opts.checkpoint)
elif opts.checkpoint != '':
    trainer.load_noise(opts.checkpoint)

torch.manual_seed(0)
os.makedirs(log_dir + 'validation/', exist_ok=True)
os.makedirs(log_dir + 'train/', exist_ok=True)
os.makedirs(log_dir + 'checkpoint/', exist_ok=True)

n_iter = 0
noise = None
print("Start!")

for n_epoch in tqdm(range(EPOCH_0, epochs)):

    iter_test = iter(loader_test)
    # iter_A = iter(loader_A)
    iter_B = iter(loader_B)
    iter_0 = n_epoch * iter_per_epoch

    print(f"[INFO]: {n_epoch:2} / {epochs:2} | Evaluating | Using {dataset_test.length} images")
    with torch.no_grad():
        trainer.globEnc.eval()
        trainer.locEnc.eval()
        cnt = 0
        flag = True
        while flag:
            print(f"[INFO]: | Evaluating {cnt}th batch |", end=" ")
            try:
                img_test = next(iter_test)
            except StopIteration:
                del iter_test
                break
            # img_test = img_test.to(device)
            trainer.compute_loss(img=img_test, train=False)
            trainer.log_test_loss()
            trainer.save_image(log_dir, n_epoch, cnt, prefix='/test/', img=img_test, train=False)
            cnt += 1
            if n_epoch == 0 and cnt > 1: break
        if n_epoch > 0:
            trainer.log_loss(logger, n_iter - 1 + iter_per_epoch, prefix='test')

    trainer.enc_opt.zero_grad()

    trainer.globEnc.train()
    trainer.locEnc.train()

    prev_time = time.perf_counter()

    for n_iter in range(iter_0, iter_0 + iter_per_epoch):
        cur_time = time.perf_counter()
        print(
            f"[INFO]: {n_epoch:2} / {epochs:2} | {n_iter - iter_0:5} / {iter_per_epoch:5} | {cur_time - prev_time :.5f} time spent",
            end="\r")
        prev_time = cur_time

        w = None
        noise = None

        img_B = None

        try:
            img_B = next(iter_B)
            if img_B.size(0) != batch_size:
                iter_B = iter(loader_B)
                img_B = next(iter_B)
        except StopIteration:
            iter_B = iter(loader_B)
            img_B = next(iter_B)
        # try:
        #     img_A = next(iter_A)
        #     if img_A.size(0) != batch_size:
        #         iter_A = iter(loader_A)
        #         img_A = next(iter_A)
        # except StopIteration:
        #     iter_A = iter(loader_A)
        #     img_A = next(iter_A)
        # img_B = img_B.to(device)
        # img_B = torch.cat([img_A, img_B], dim=0)
        if VERB:
            print(f"[*] Concat real and fake {img_B.shape=}")
            VERB = False

        trainer.update(img=img_B, n_iter=n_iter)
        # breakpoint()
        if (n_iter + 1) % config['log_iter'] == 0:
            trainer.log_loss(logger, n_iter, prefix='train')
        if (n_iter + 1) % config['image_save_iter'] == 0:
            trainer.save_image(log_dir, n_epoch, n_iter, prefix='/train/', img=img_B, train=True)
    trainer.enc_scheduler.step()
    trainer.save_checkpoint(n_epoch, log_dir + 'checkpoint/')

trainer.save_model(log_dir)
