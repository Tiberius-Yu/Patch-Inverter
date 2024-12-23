import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from PIL import Image
from torch.autograd import grad


def clip_img(x):
    """Clip stylegan generated image to range(0,1)"""
    if x.shape[0] != 1:
        img_tmp = torch.cat(list(x), dim=1)
    else:
        img_tmp = x.clone()[0]
    img_tmp = (img_tmp + 1) / 2
    img_tmp = torch.clamp(img_tmp, 0, 1)
    return [img_tmp.detach().cpu()]


def tensor_byte(x):
    return x.element_size() * x.nelement()


def count_parameters(net):
    s = sum([np.prod(list(mm.size())) for mm in net.parameters()])
    print(s)


def gan_to_classifier(x, out_size=(224, 224)):
    """Clip image to range(0,1)"""
    img_tmp = x.clone()
    img_tmp = torch.clamp((0.5 * img_tmp + 0.5), 0, 1)
    img_tmp = F.interpolate(img_tmp, size=out_size, mode='bilinear')
    img_tmp[:, 0] = (img_tmp[:, 0] - 0.485) / 0.229
    img_tmp[:, 1] = (img_tmp[:, 1] - 0.456) / 0.224
    img_tmp[:, 2] = (img_tmp[:, 2] - 0.406) / 0.225
    #img_tmp = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tmp)
    return img_tmp


def downscale(x, scale_times=1, mode='bilinear'):
    for i in range(scale_times):
        x = F.interpolate(x, scale_factor=0.5, mode=mode)
    return x


def upscale(x, scale_times=1, mode='bilinear'):
    for i in range(scale_times):
        x = F.interpolate(x, scale_factor=2, mode=mode)
    return x


def hist_transform(source_tensor, target_tensor):
    """Histogram transformation"""
    c, h, w = source_tensor.size()
    s_t = source_tensor.view(c, -1)
    t_t = target_tensor.view(c, -1)
    s_t_sorted, s_t_indices = torch.sort(s_t)
    t_t_sorted, t_t_indices = torch.sort(t_t)
    for i in range(c):
        s_t[i, s_t_indices[i]] = t_t_sorted[i]
    return s_t.view(c, h, w)


def init_weights(m):
    """Initialize layers with Xavier uniform distribution"""
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.Linear:
        nn.init.uniform_(m.weight, 0.0, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)


def total_variation(x, delta=1):
    """Total variation, x: tensor of size (B, C, H, W)"""
    out = torch.mean(torch.abs(x[:, :, :, :-delta] - x[:, :, :, delta:]))\
        + torch.mean(torch.abs(x[:, :, :-delta, :] - x[:, :, delta:, :]))
    return out


def vgg_transform(x):
    """Adapt image for vgg network, x: image of range(0,1) subtracting ImageNet mean"""
    r, g, b = torch.split(x, 1, 1)
    out = torch.cat((b, g, r), dim=1)
    out = F.interpolate(out, size=(224, 224), mode='bilinear')
    out = out * 255.
    return out


# warp image with flow
def normalize_axis(x, L):
    return (x - 1 - (L - 1) / 2) * 2 / (L - 1)


def unnormalize_axis(x, L):
    return x * (L - 1) / 2 + 1 + (L - 1) / 2


def torch_flow_to_th_sampling_grid(flow, h_src, w_src, use_cuda=False):
    b, c, h_tgt, w_tgt = flow.size()
    grid_y, grid_x = torch.meshgrid(
        torch.tensor(range(1, w_tgt + 1)), torch.tensor(range(1, h_tgt + 1)))
    disp_x = flow[:, 0, :, :]
    disp_y = flow[:, 1, :, :]
    source_x = grid_x.unsqueeze(0).repeat(b, 1, 1).type_as(flow) + disp_x
    source_y = grid_y.unsqueeze(0).repeat(b, 1, 1).type_as(flow) + disp_y
    source_x_norm = normalize_axis(source_x, w_src)
    source_y_norm = normalize_axis(source_y, h_src)
    sampling_grid = torch.cat((source_x_norm.unsqueeze(3), source_y_norm.unsqueeze(3)), dim=3)
    if use_cuda:
        sampling_grid = sampling_grid.cuda()
    return sampling_grid


def warp_image_torch(image, flow):
    """
    Warp image (tensor, shape=[b, 3, h_src, w_src]) with flow (tensor, shape=[b, h_tgt, w_tgt, 2])
    """
    b, c, h_src, w_src = image.size()
    sampling_grid_torch = torch_flow_to_th_sampling_grid(flow, h_src, w_src)
    warped_image_torch = F.grid_sample(image, sampling_grid_torch)
    return warped_image_torch


def safe_load_state_dict(model, state_dict, strict=True):
    # FrEAI saves some unused runtime variables
    state_dict = {k: v for k, v in state_dict.items() if 'tmp_var' not in k}
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict, strict=strict)
    elif isinstance(model, nn.Module):
        model.load_state_dict(state_dict, strict=strict)
    else:  # Optimizer
        model.load_state_dict(state_dict)


class MMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''

    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul**(kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source,
                target,
                kernel_mul=self.kernel_mul,
                kernel_num=self.kernel_num,
                fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss
