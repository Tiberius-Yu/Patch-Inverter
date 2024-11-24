import sys
import time

import os
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from PIL import Image
from torch.autograd import grad
from torchvision import transforms, utils
from IPython import embed
from generator.infinity.infinitygan_generator import InfinityGanGenerator
from test_managers.infinite_generation import InfiniteGenerationManager
from nets.local_latent_encoder import localLatentEncoder
from random import randint

from itertools import product as iter_product
from test_managers.global_config import *

# import face_alignment
import lpips

sys.path.append('pixel2style2pixel/')

from nets.feature_style_encoder import *
from utils.functions import *
from arcface.iresnet import *
from face_parsing.model import BiSeNet
from ranger import Ranger


class Trainer(nn.Module):

    def __init__(self, config, opts, log_dir):
        super(Trainer, self).__init__()
        # Load Hyperparameters
        self.config = config
        self.logdir = log_dir
        self.device = torch.device(self.config['device'])
        self.scale = int(np.log2(config['resolution'] / config['enc_resolution']))
        self.scale_mode = 'bilinear'
        self.opts = opts
        self.n_styles = 2 * int(np.log2(config['resolution'])) - 2
        self.idx_k = 5
        self.cnt = 0
        self.cntcnt = 0
        self.landmark_lossAll = 0
        self.l1_lossAll = 0
        self.id_lossAll = 0
        self.lpips_lossAll = 0
        self.l2_lossAll = 0
        self.lossAll = 0
        self.parallel_stack = []
        if 'idx_k' in self.config:
            self.idx_k = self.config['idx_k']
        # Networks
        in_channels = 256
        if 'in_c' in self.config:
            in_channels = config['in_c']
        enc_residual = False
        if 'enc_residual' in self.config:
            enc_residual = self.config['enc_residual']
        enc_residual_coeff = False
        if 'enc_residual_coeff' in self.config:
            enc_residual_coeff = self.config['enc_residual_coeff']
        resnet_layers = [4, 5, 6]
        if 'enc_start_layer' in self.config:
            st_l = self.config['enc_start_layer']
            resnet_layers = [st_l, st_l + 1, st_l + 2]
        if 'scale_mode' in self.config:
            self.scale_mode = self.config['scale_mode']
        # Load encoder
        self.stride = (self.config['fs_stride'], self.config['fs_stride'])
        self.globEnc = fs_encoder_v2(
            opts=opts,
            residual=enc_residual,
            use_coeff=enc_residual_coeff,
            resnet_layer=resnet_layers,
            stride=self.stride)
        self.locEnc = localLatentEncoder(config=config['model'])

        ##########################
        # Other nets
        Generator, Discriminator = self.init_gan(config)
        # self.Arcface = iresnet50()
        self.disc = Discriminator

        self.patch_generator = InfiniteGenerationManager(
            Generator, config=config, logdir=self.logdir)
        self.patch_generator.task_specific_init()

        self.mmd = MMDLoss()
        self.isTest = False
        # h, w = self.patch_generator.full_local_latent_shape
        # localshape = (self.patch_generator.config.train_params.batch_size, 1,
        #               h + 2 * self.patch_generator.ss_unfold_size,
        #               w + 2 * self.patch_generator.ss_unfold_size)
        # print(f"[*] Using localshape {localshape} for coord")
        # locallatent = torch.empty(localshape)
        # self.meta_coords = self.patch_generator.coord_handler.sample_coord_grid(
        #     locallatent, is_training=False)

        import torchvision.models as models
        self.MOCO = models.__dict__["resnet50"]()
        # freeze all layers but the last fc
        for name, param in self.MOCO.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False

        self.parsing_net = BiSeNet(n_classes=19)
        self.lambda_ = 0.65
        # Optimizers
        # Latent encoder
        self.enc_params = list(self.globEnc.parameters())
        self.enc_params += list(self.locEnc.parameters())
        if 'freeze_iresnet' in self.config and self.config['freeze_iresnet']:
            self.enc_params = list(self.enc.styles.parameters())
        if 'optimizer' in self.config and self.config['optimizer'] == 'ranger':
            self.enc_opt = Ranger(
                self.enc_params,
                lr=config['lr'],
                betas=(config['beta_1'], config['beta_2']),
                weight_decay=config['weight_decay'])
        else:
            self.enc_opt = torch.optim.Adam(
                self.enc_params,
                lr=config['lr'],
                betas=(config['beta_1'], config['beta_2']),
                weight_decay=config['weight_decay'])
        self.enc_scheduler = torch.optim.lr_scheduler.StepLR(
            self.enc_opt, step_size=config['step_size'], gamma=config['gamma'])

        self.fea_avg = None
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        encode_h = encode_patch_shape + (self.patch_generator.num_steps_h -
                                         1) * self.patch_generator.pixelspace_step_size
        encode_w = encode_patch_shape + (self.patch_generator.num_steps_w -
                                         1) * self.patch_generator.pixelspace_step_size
        self.resize = transforms.Resize([encode_h, encode_w])
        print(f"[*] Building Encode shape {[encode_h, encode_w]}")

        self.randomized_noises = [
            torch.randn(self.patch_generator.config.train_params.batch_size, 1, int(h), int(w))
            for (h, w) in zip(self.patch_generator.noise_heights, self.patch_generator.noise_widths)
        ]
        self.time = 0
        self.cnt = 0

    def initialize(self, gan_model_path, moco_model_path, parsing_model_path):
        # load Generator model
        ckpt = torch.load(gan_model_path, map_location='cpu')
        safe_load_state_dict(self.patch_generator.g_ema_module, ckpt['g_ema'])
        print(" [*] Loaded ckpt at {} iter with FID {:.4f}".format(ckpt["iter"], ckpt["best_fid"]))
        self.patch_generator.g_ema_module.to(self.device)
        # self.Arcface.load_state_dict(torch.load(self.opts.arcface_model_path))
        # self.Arcface.eval()

        checkpoint = torch.load(self.opts.moco_model_path, map_location="cpu")
        state_dict = checkpoint['state_dict']
        # rename moco pre-trained keys
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = self.MOCO.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        # remove output layer
        self.MOCO = nn.Sequential(*list(self.MOCO.children())[:-1]).cuda()

        # load face parsing net weight
        self.parsing_net.load_state_dict(torch.load(self.opts.parsing_model_path))
        self.parsing_net.eval()
        # load lpips net weight
        self.loss_fn = lpips.LPIPS(net='alex', spatial=False)
        self.loss_fn.to(self.device)

    def init_gan(self, config):
        Generator = InfinityGanGenerator(config['model']).cuda().eval()
        return Generator, Discriminator

    def mapping(self, z):
        return self.Generator.get_latent(z).detach()

    def L1loss(self, input, target):
        return nn.L1Loss()(input, target)

    def L2loss(self, input, target):
        return nn.MSELoss()(input, target)

    def CEloss(self, x, target_age):
        return nn.CrossEntropyLoss()(x, target_age)

    def LPIPS(self, input_, target, multi_scale=False):
        if multi_scale:
            out = 0
            try:
                for k in range(3):
                    out += self.loss_fn.forward(
                        downscale(input_, k, self.scale_mode),
                        downscale(target, k, self.scale_mode)).mean()
            except RuntimeError:
                input_ = self.face_pool(input_)
                target = self.face_pool(target)
                for k in range(3):
                    out += self.loss_fn.forward(
                        downscale(input_, k, self.scale_mode),
                        downscale(target, k, self.scale_mode)).mean()
        else:
            out = self.loss_fn.forward(
                downscale(input_, self.scale, self.scale_mode),
                downscale(target, self.scale, self.scale_mode)).mean()
        return out

    def extract_feats(self, x):
        x = F.interpolate(x, size=224)
        x_feats = self.MOCO(x)
        x_feats = nn.functional.normalize(x_feats, dim=1)
        x_feats = x_feats.squeeze()
        return x_feats

    def IDloss(self, input, target):
        n_samples = input.shape[0]
        x_feats = self.extract_feats(input)
        y_feats = self.extract_feats(target)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        sim_logs = []
        count = 0
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        # from IPython import embed
        # embed()
        for i in range(n_samples):
            loss += 1 - cos(x_feats[i], y_feats[i])
            count += 1

        return loss / count

    def IDsim(self, input, target):
        n_samples = input.shape[0]
        x_feats = self.extract_feats(input)
        y_feats = self.extract_feats(target)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        sim_logs = []
        count = 0
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        # from IPython import embed
        # embed()
        for i in range(n_samples):
            loss += cos(x_feats[i], y_feats[i])
            count += 1

        return loss / count

    # def IDloss(self, input, target):
    #     x_1 = F.interpolate(input, (112,112))
    #     x_2 = F.interpolate(target, (112,112))
    #     cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    #     if 'multi_layer_idloss' in self.config and self.config['multi_layer_idloss']:
    #         id_1 = self.Arcface(x_1, return_features=True)
    #         id_2 = self.Arcface(x_2, return_features=True)
    #         return sum([1 - cos(id_1[i].flatten(start_dim=1), id_2[i].flatten(start_dim=1)) for i in range(len(id_1))])
    #     else:
    #         id_1 = self.Arcface(x_1)
    #         id_2 = self.Arcface(x_2)
    #         return 1 - cos(id_1, id_2)

    def landmarkloss(self, input_, target):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        x_1 = gan_to_classifier(input_, out_size=(512, 512))
        x_2 = gan_to_classifier(target, out_size=(512, 512))
        out_1 = self.parsing_net(x_1)
        out_2 = self.parsing_net(x_2)
        parsing_loss = sum([
            1 - cos(out_1[i].flatten(start_dim=1), out_2[i].flatten(start_dim=1))
            for i in range(len(out_1))
        ])
        return parsing_loss.mean()

    def feature_match(self, enc_feat, dec_feat, layer_idx=None):
        loss = []
        if layer_idx is None:
            layer_idx = [i for i in range(len(enc_feat))]
        for i in layer_idx:
            loss.append(self.L1loss(enc_feat[i], dec_feat[i]))
        return loss

    def encode(self, img):
        w_recon, fea = self.enc(downscale(img, self.scale, self.scale_mode))

        return w_recon, fea

    def get_image(self,
                  img=None,
                  train=False,
                  show_ori=False,
                  infer=False,
                  bound=None,
                  infer_reverse=False,
                  oriimg=None):
        patch_conf = self.patch_generator
        bs, _, H, W = img.shape
        # breakpoint()
        assert (not train) or (bs == self.patch_generator.config.train_params.batch_size)
        # input_img = torch.zeros(bs, 3, int(self.patch_generator.meta_height),
        # int(self.patch_generator.meta_width))
        patch_shape = self.patch_generator.outfeat_sizes_list[-1]
        meta_h, meta_w = int(patch_conf.meta_height) + ec_unfold_size, int(
            patch_conf.meta_width) + ec_unfold_size
        # FIXME: Inifinity
        pad_h, pad_w = (int(self.patch_generator.small_meta_height) - img.shape[2]) // 2 + ec_unfold_size, \
                    (int(self.patch_generator.small_meta_width) - img.shape[3]) // 2 + ec_unfold_size
        # input_img[:, :, pad_h:pad_h + H, pad_w:pad_w + W] = img
        # extra_H = H // self.patch_generator.pixelspace_step_size
        # extra_W = W // self.patch_generator.pixelspace_step_size
        # extra_padding_step = max(extra_H, extra_W)
        # print(f"[*] padding {pad_w}, {pad_h}", end = "\r")
        input_img = F.pad(
            img, [
                pad_w,
                int(meta_w) - pad_w - W,
                pad_h,
                int(meta_h) - pad_h - H,
            ], mode="reflect")
        # breakpoint()
        # global_latent = self.globEnc(self.face_pool(img.to(self.device)))
        # rand_glob = torch.rand_like(global_latent).cuda()
        global_latent = torch.randn((bs, 512)).cuda()
        if infer:
            self.lambda_ += 0.05
            lambda_ = self.lambda_
            # lambda_ = 0.5
            print(f"{lambda_=}")
            meta_feat = torch.ones(bs, 256, 41, 41) * -10
            os.makedirs(self.logdir + "/features", exist_ok=True)
            os.makedirs(self.logdir + "/images", exist_ok=True)
            prevfea = None
            prevslImage = None
            meta_img_mix = torch.empty(bs, 3, meta_h, meta_w).float()
            # reverse_global = torch.flip(global_latent, [0])
            reverse_global = lambda_ * torch.randn(
                (bs, 512)).to(self.device) + (1 - lambda_) * global_latent
        # upscale_img = self.resize(input_img)
        # Testing and Get Whole image
        if not train:
            # in_patchs = []
            out_patchs = []
            real_patchs = []
            idx_tuples = list(
                iter_product(
                    range(self.patch_generator.start_pts_mesh_outfeats[-1].shape[0]),
                    range(self.patch_generator.start_pts_mesh_outfeats[-1].shape[1])))
            # pbar = tqdm.tqdm(idx_tuples)
            meta_img = torch.empty(bs, 3, meta_h, meta_w).float()
            if bound is not None:
                meta_img_edit = torch.empty(bs, 3, meta_h, meta_w).float()
                edit_glob = global_latent + bound
            for _, (idx_x, idx_y) in enumerate(idx_tuples):
                # img_x_st, img_y_st = idx_x * self.patch_generator.pixelspace_step_size, idx_y * self.patch_generator.pixelspace_step_size
                # img_x_ed, img_y_ed = img_x_st + encode_patch_shape, img_y_st + encode_patch_shape
                img_x_st, img_y_st = idx_x * self.patch_generator.pixelspace_step_size, idx_y * self.patch_generator.pixelspace_step_size
                img_x_ed, img_y_ed = img_x_st + patch_shape, img_y_st + patch_shape
                zx_st, zy_st = img_x_st, img_y_st
                zx_ed, zy_ed = img_x_ed, img_y_ed

                # Handle the randomized noise input of the texture_synthesizer...
                outfeat_x_st = [
                    start_pts_mesh[idx_x, idx_y, 0]
                    for start_pts_mesh in self.patch_generator.start_pts_mesh_outfeats
                ]
                outfeat_y_st = [
                    start_pts_mesh[idx_x, idx_y, 1]
                    for start_pts_mesh in self.patch_generator.start_pts_mesh_outfeats
                ]
                outfeat_x_ed = [
                    x_st + out_size
                    for (x_st,
                         out_size) in zip(outfeat_x_st, self.patch_generator.outfeat_sizes_list)
                ]
                outfeat_y_ed = [
                    y_st + out_size
                    for (y_st,
                         out_size) in zip(outfeat_y_st, self.patch_generator.outfeat_sizes_list)
                ]
                noises = []
                for i, (fx_st, fy_st, fx_ed, fy_ed) in enumerate(
                        zip(outfeat_x_st, outfeat_y_st, outfeat_x_ed, outfeat_y_ed)):
                    noises.append(self.randomized_noises[i][:bs, :, fx_st:fx_ed,
                                                            fy_st:fy_ed].to(self.device))

                # Deal with ec unfolding here
                img_x_st += ec_unfold_size
                img_x_ed += ec_unfold_size
                img_y_st += ec_unfold_size
                img_y_ed += ec_unfold_size
                zx_ed += ec_unfold_size
                zy_ed += ec_unfold_size

                # cur_coords = self.meta_coords[:bs, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)
                print(
                    f"{(zx_st,zx_ed,zy_st,zy_ed)}|{(img_x_st,img_x_ed,img_y_st,img_y_ed)}",
                    end="\r")
                inp_image = input_img[:bs, :, zx_st:zx_ed, zy_st:zy_ed]
                # pad_image = F.pad(sl_image, [61, 61, 61, 61], mode="replicate")
                # structure_latent = self.locEnc(inp_image.to(self.device))
                structure_latent = self.locEnc(inp_image.to(self.device))
                # local_latent_display = torch.zeros(
                #     self.patch_generator.config.train_params.batch_size, 256, 41, 41).cuda()
                # breakpoint()
                # pre_latent = structure_latent

                index_tuple = (img_x_st, img_x_ed, img_y_st, img_y_ed)
                if self.patch_generator.enable_parallel_batching:
                    self.parallel_stack.append({"count": index_tuple})

                # structure_latent = self.patch_generator.g_ema_module.structure_generate(
                #     global_latent=global_latent,
                #     local_latent=local_latent,
                #     override_coords=cur_coords)
                patch = self.patch_generator.g_ema_module.texture_generate(
                    global_latent=global_latent, structure_latent=structure_latent,
                    noises=noises)["gen"]
                if not infer and bound is not None:
                    patch_new = self.patch_generator.g_ema_module.texture_generate(
                        global_latent=edit_glob, structure_latent=structure_latent,
                        noises=noises)["gen"]
                    meta_img_edit[:, :, img_x_st:img_x_ed,
                                  img_y_st:img_y_ed] = patch_new.detach().cpu()
                if self.patch_generator.enable_parallel_batching:
                    for i, _ in enumerate(self.parallel_stack):
                        meta_img[:, :, img_x_st:img_x_ed,
                                 img_y_st:img_y_ed] = patch[i:i + 1].detach().cpu()
                else:
                    meta_img[:, :, img_x_st:img_x_ed, img_y_st:img_y_ed] = patch.detach().cpu()
                if infer:
                    structure_latent_new = self.locEnc(inp_image.to(self.device), reverse_global)
                    patch_mix = self.patch_generator.g_ema_module.texture_generate(
                        global_latent=reverse_global,
                        structure_latent=structure_latent_new,
                        noises=noises)["gen"]
                    meta_img_mix[:, :, img_x_st:img_x_ed,
                                 img_y_st:img_y_ed] = patch_mix.detach().cpu()
                    # utils.save_image(
                    #     clip_img(meta_img), f"{self.logdir}/images/{idx_x}_{idx_y}_full.png")
                    # # for i, fea in enumerate(feats):
                    # utils.save_image(
                    #     clip_img(patch), f"{self.logdir}/images/{idx_x}_{idx_y}_patch.png")

                    # feats = None
                    # meta_feat[:, :, 6 * idx_x:6 * idx_x + 11,
                    #           6 * idx_y:6 * idx_y + 11] = structure_latent.detach().cpu()
                    # utils.save_image(
                    #     clip_img(meta_feat[:, :3]),
                    #     f"{self.logdir}/features/{idx_x}_{idx_y}_feat.png")
                    # for i, fea in enumerate(feats):
                    #     utils.save_image(
                    #         clip_img(fea[:, :3]),
                    #         f"{self.logdir}/features/{idx_x}_{idx_y}_feat_{i}.png")
                    # prevfea = feats
            if infer:
                return img.to(self.device), meta_img[:, :, pad_h:pad_h + H, pad_w:pad_w + W].to(
                    self.device), meta_img_mix[:, :, pad_h:pad_h + H,
                                               pad_w:pad_w + W].to(self.device)
            if bound is not None:
                return img.to(self.device), meta_img[:, :, pad_h:pad_h + H, pad_w:pad_w + W].to(
                    self.device), meta_img_edit[:, :, pad_h:pad_h + H,
                                                pad_w:pad_w + W].to(self.device)
            if show_ori:
                return img.to(self.device), meta_img[:, :, pad_h:pad_h + H, pad_w:pad_w + W].to(self.device), \
                                    input_img.detach().cpu(), \
                                    meta_img.detach().cpu() #, out_patch.detach().cpu(), real_patch.detach().cpu()
            meta_img = meta_img[:, :, pad_h:pad_h + H, pad_w:pad_w + W]
            return img.to(self.device), meta_img.to(self.device)
        else:
            idx_x = randint(0, self.patch_generator.start_pts_mesh_outfeats[-1].shape[0] - 1)
            idx_y = randint(0, self.patch_generator.start_pts_mesh_outfeats[-1].shape[1] - 1)
            #1).pixelspace_step_size

            # rand_glob = torch.randn((bs, 512)).cuda()
            rand_local = torch.randn((bs, 256, 35, 35)).cuda()

            img_x_st, img_y_st = idx_x * self.patch_generator.pixelspace_step_size, idx_y * self.patch_generator.pixelspace_step_size
            img_x_ed, img_y_ed = img_x_st + patch_shape, img_y_st + patch_shape
            zx_st, zy_st = img_x_st, img_y_st
            zx_ed, zy_ed = img_x_ed, img_y_ed
            # Handle the randomized noise input of the texture_synthesizer...
            outfeat_x_st = [
                start_pts_mesh[idx_x, idx_y, 0]
                for start_pts_mesh in self.patch_generator.start_pts_mesh_outfeats
            ]
            outfeat_y_st = [
                start_pts_mesh[idx_x, idx_y, 1]
                for start_pts_mesh in self.patch_generator.start_pts_mesh_outfeats
            ]
            outfeat_x_ed = [
                x_st + out_size
                for (x_st, out_size) in zip(outfeat_x_st, self.patch_generator.outfeat_sizes_list)
            ]
            outfeat_y_ed = [
                y_st + out_size
                for (y_st, out_size) in zip(outfeat_y_st, self.patch_generator.outfeat_sizes_list)
            ]
            noises = []
            for i, (fx_st, fy_st, fx_ed, fy_ed) in enumerate(
                    zip(outfeat_x_st, outfeat_y_st, outfeat_x_ed, outfeat_y_ed)):
                noises.append(self.randomized_noises[i][:, :, fx_st:fx_ed,
                                                        fy_st:fy_ed].to(self.device))

            # select_img = input_img[:, :, img_x_st:img_x_ed, img_y_st:img_y_ed].to(self.device)
            # breakpoint()
            # Deal with SS unfolding here
            # zx_st -= self.patch_generator.ss_unfold_size
            # zy_st -= self.patch_generator.ss_unfold_size
            # zx_ed += self.patch_generator.ss_unfold_size
            # zy_ed += self.patch_generator.ss_unfold_size
            img_x_st += ec_unfold_size
            img_x_ed += ec_unfold_size
            img_y_st += ec_unfold_size
            img_y_ed += ec_unfold_size
            zx_ed += ec_unfold_size
            zy_ed += ec_unfold_size

            # cur_coords = self.meta_coords[:, :, zx_st:zx_ed, zy_st:zy_ed].to(self.device)

            # sl_image = input_img[:bs, :, img_x_st:img_x_ed, img_y_st:img_y_ed].to(self.device)
            # inp_image = input_img[:bs, :, zx_st:zx_ed, zy_st:zy_ed]
            # real part
            sl_image = input_img[:, :, img_x_st:img_x_ed, img_y_st:img_y_ed].to(self.device)
            inp_image = input_img[:, :, zx_st:zx_ed, zy_st:zy_ed]
            structure_latent = self.locEnc(inp_image.to(self.device))

            patch = self.patch_generator.g_ema_module.texture_generate(
                global_latent=global_latent, structure_latent=structure_latent,
                noises=noises)["gen"]
            # pad_image = F.pad(sl_image, [61, 61, 61, 61], mode="replicate")

            # fake part
            out = self.patch_generator.g_ema(
                global_latent=global_latent,
                local_latent=rand_local,
                noises=noises,
                disable_dual_latents=True)
            rand_struct = out['structure_latent']
            sl_image_fake = out['gen']
            inp_image = F.pad(sl_image_fake, (60, 0, 60, 0), mode="constant")
            structure_latent = self.locEnc(inp_image.to(self.device))

            # structure_latent = self.patch_generator.g_ema_module.structure_generate(
            #     global_latent=global_latent, local_latent=local_latent, override_coords=cur_coords)

            # patch_fake = self.patch_generator.g_ema_module.texture_generate(
            #     global_latent=global_latent, structure_latent=structure_latent,
            #     noises=noises)["gen"]

            # patch_rand = self.patch_generator.g_ema_module.texture_generate(
            #     global_latent=rand_glob, structure_latent=structure_latent, noises=noises)["gen"]
            # img_in = torch.cat([sl_image, sl_image_fake], dim=0)
            # img_out = torch.cat([patch, patch_fake])
            img_in = sl_image
            img_out = patch
            return rand_glob, rand_struct, global_latent, structure_latent, img_in, img_out

    def compute_loss(self, img=None, train=False):
        return self.compute_loss_gan(img=img, train=train)

    def compute_loss_gan(self, img=None, train=False):
        if not train:
            self.rand_scale = randint(5, 20)
            bound = torch.Tensor(
                np.load("~/PatchInv/DATA/scenen-interface/boundaries/snowfield_boundary.npy")
            ).cuda() * self.rand_scale
            with torch.no_grad():
                out = self.get_image(img=img, train=train, show_ori=True, bound=bound)
        else:
            out = self.get_image(img=img, train=train, show_ori=True)
        if not train:
            x_1, x_1_recon, x_1_edit = out
            self.image = (x_1.detach(), x_1_recon.detach(), x_1_edit)
            # self.patches = torch.cat([in_patch, out_patch], dim=2)
        else:
            rg, rs, eg, es, x_1, x_1_recon = out

            self.image = (x_1, x_1_recon)
        self.isTest = not train
        if self.isTest:
            self.num_test = img.shape[0]

        # Loss setting
        w_l2, w_lpips, w_id = self.config['w']['l2'], self.config['w']['lpips'], self.config['w'][
            'id']
        # if 'l2loss_on_real_image' in self.config and self.config['l2loss_on_real_image']:
        # b = x_1.size(0)
        # b = x_1.size(0) // 2
        self.l2_loss = self.L2loss(x_1_recon, x_1) if w_l2 > 0 else torch.tensor(
            0)  # l2 loss only on synthetic data
        # LPIPS
        multiscale_lpips = False if 'multiscale_lpips' not in self.config else self.config[
            'multiscale_lpips']
        self.lpips_loss = self.LPIPS(
            x_1_recon, x_1, multi_scale=multiscale_lpips).mean() if w_lpips > 0 else torch.tensor(0)
        self.id_loss = self.IDloss(x_1_recon, x_1).mean() if w_id > 0 else torch.tensor(0)
        self.landmark_loss = self.landmarkloss(
            x_1_recon, x_1) if self.config['w']['landmark'] > 0 else torch.tensor(0)

        # if train:
        #     w_adv, w_disc = self.config['w']['adv'], self.config['w']['disc']
        #     rand_glob_loss = torch.Tensor(0).cuda()
        #     if w_disc > 0:
        #         fake_pred = self.disc(patch_rand)
        #         rand_glob_loss = F.softplus(-fake_pred).mean()
        #     self.lpips_loss_adv = self.LPIPS(
        #         patch_rand, x_1,
        #         multi_scale=multiscale_lpips).mean() if w_lpips > 0 else torch.tensor(0)
        #     self.id_loss_adv = self.IDloss(patch_rand, x_1).mean() if w_id > 0 else torch.tensor(0)
        if train:
            self.recip_loss = self.L2loss(rs, es)
            self.distribution = self.mmd(rg, eg)

        # Total loss
        w_l2, w_lpips, w_id = self.config['w']['l2'], self.config['w']['lpips'], self.config['w'][
            'id']
        self.loss = w_l2 * self.l2_loss + w_lpips * self.lpips_loss + w_id * self.id_loss

        if train:
            w_dist, w_recip = self.config['w']['dist'], self.config['w']['recip']
            self.loss += w_dist * self.distribution + w_recip * self.recip_loss
        # if train:
        #     self.loss += -(w_lpips * self.lpips_loss_adv + w_id * self.id_loss_adv) * w_adv
        #     if w_disc > 0:
        #         self.loss += rand_glob_loss * w_disc
        #         self.glob_loss = rand_glob_loss
        # if self.l2_loss.item() < 1:
        #     self.loss += w_l2 * self.l2_loss

        if 'l1' in self.config['w'] and self.config['w']['l1'] > 0:
            self.l1_loss = self.L1loss(x_1_recon, x_1)
            self.loss += self.config['w']['l1'] * self.l1_loss
        if 'landmark' in self.config['w']:
            self.loss += self.config['w']['landmark'] * self.landmark_loss
        return self.loss

    def test(self,
             img=None,
             logdir=None,
             iter=None,
             bound=None,
             name="inversion",
             infer_reverse=True):
        # if infer_reverse:
        #     print(img.shape)
        #     old_img = img
        #     b, c, h, w = old_img.shape
        #     # img_new = old_img.reshape(b, c, h, 2, w // 2)
        #     img = torch.cat((old_img[:, :, :, 280:], old_img[:, :, :, :280]), dim=3)
        #     # img = img_new.reshape(b, c, h, w)
        # breakpoint()
        # start = time.time()
        out = self.get_image(img=img, train=False, infer=False, bound=bound, show_ori=False)
        x_1, x_1_recon = out
        # end = time.time()
        # if self.cnt > 100:
        #     self.time += end - start
        #     print()
        #     print(self.time / (self.cnt - 5))
        # self.cnt += 1
        # x_1 = self.face_pool(x_1).detach().cpu()
        # x_1_recon = self.face_pool(x_1recon).detach().cpu()
        # out_img = torch.cat((x_1, x_1_recon, x_1_mix), dim=3)
        utils.save_image(clip_img(x_1_recon), logdir + "/recon/" + str(iter) + f'.png')
        # utils.save_image(clip_img(img), logdir + "/" + str(iter) + f'.png')

        # Total loss
        w_l2, w_lpips, w_id = self.config['w']['l2'], self.config['w']['lpips'], self.config['w'][
            'id']
        self.l2_loss = self.L2loss(x_1_recon, x_1) if w_l2 > 0 else torch.tensor(
            0)  # l2 loss only on synthetic data
        multiscale_lpips = False if 'multiscale_lpips' not in self.config else self.config[
            'multiscale_lpips']
        self.lpips_loss = self.LPIPS(
            x_1_recon, x_1, multi_scale=multiscale_lpips).mean() if w_lpips > 0 else torch.tensor(0)
        self.id_loss = self.IDsim(x_1_recon, x_1).mean() if w_id > 0 else torch.tensor(0)
        print(f"{self.l2_loss}, {self.lpips_loss} {self.id_loss}")
        with open("~/PatchInv/criteria/ssim_record/fse_save", "a") as f:
            f.write(f"{self.l2_loss}, {self.lpips_loss} {self.id_loss}\n")
        # return output

    def log_test_loss(self):
        self.l2_lossAll += self.l2_loss
        self.lpips_lossAll += self.lpips_loss
        self.id_lossAll += self.id_loss
        self.cnt += 1

    def compute_test_loss(self):
        print(f"{self.l2_lossAll / self.cnt=}")
        print(f"{self.lpips_lossAll / self.cnt=}")
        print(f"{self.id_lossAll / self.cnt=}")
        # breakpoint()

    def log_loss(self, logger, n_iter, prefix='train'):
        print(f"{self.l2_lossAll / self.cnt=}")
        print(f"{self.lpips_lossAll / self.cnt=}")
        print(f"{self.id_lossAll / self.cnt=}")
        logger.log_value(prefix + '/l2_loss', self.l2_loss.item(), n_iter + 1)
        logger.log_value(prefix + '/lpips_loss', self.lpips_loss.item(), n_iter + 1)
        logger.log_value(prefix + '/id_loss', self.id_loss.item(), n_iter + 1)
        logger.log_value(prefix + '/distribution', self.distribution.item(), n_iter + 1)
        logger.log_value(prefix + '/recip', self.recip_loss.item(), n_iter + 1)
        # if 'disc' in self.config['w'] and self.config['w']['disc'] > 0:
        #     logger.log_value(prefix + '/glob_loss', self.glob_loss.item(), n_iter + 1)
        logger.log_value(prefix + '/total_loss', self.loss.item(), n_iter + 1)
        if 'l1' in self.config['w'] and self.config['w']['l1'] > 0:
            logger.log_value(prefix + '/l1_loss', self.l1_loss.item(), n_iter + 1)
        if 'landmark' in self.config['w']:
            logger.log_value(prefix + '/landmark_loss', self.landmark_loss.item(), n_iter + 1)

    def save_image(self, log_dir, n_epoch, n_iter, prefix='/train/', img=None, train=False):
        return self.save_image_gan(
            log_dir=log_dir, n_epoch=n_epoch, n_iter=n_iter, prefix=prefix, img=img, train=train)

    def save_image_gan(
        self,
        log_dir,
        n_epoch,
        n_iter,
        prefix='/train/',
        img=None,
        train=False,
    ):
        os.makedirs(log_dir + prefix + 'epoch_' + str(n_epoch), exist_ok=True)
        with torch.no_grad():
            scale = 0 if train else self.rand_scale
            if train:
                if hasattr(self, "image") and self.image is not None:
                    x_1, x_1_recon = self.image
                else:
                    out = self.get_image(img=img, train=train, show_ori=False)
                    _, _, _, _, x_1, x_1_recon = out
                    # x_1, x_1_recon = x_1_full[:, :, :self.patch_generator.
                    #                           pixelspace_step_size, :self.patch_generator.
                    #                           pixelspace_step_size], x_1_recon_full[:, :, :self.
                    #                                                                 patch_generator.
                    #                                                                 pixelspace_step_size, :
                    #                                                                 self.
                    #                                                                 patch_generator.
                    #                                                                 pixelspace_step_size]
                x_1 = self.face_pool(x_1).detach().cpu()
                x_1_recon = self.face_pool(x_1_recon).detach().cpu()
                # x_1_full = self.face_pool(x_1_full).detach().cpu()
                # x_1_recon_full = self.face_pool(x_1_recon_full).detach().cpu()
                out_img = torch.cat((x_1, x_1_recon), dim=3)
            else:
                if hasattr(self, "image") and self.image is not None:
                    x_1, x_1_recon, x_1_edit = self.image
                else:
                    out = self.get_image(img=img, train=train, show_ori=True)
                    # x_1, x_1_pad, x_1_gen, x_1_recon = out
                    x_1, x_1_recon, x_1_edit = out
                x_1 = self.face_pool(x_1).detach().cpu()
                x_1_recon = self.face_pool(x_1_recon).detach().cpu()
                # x_1_pad = self.face_pool(x_1_pad).detach().cpu()
                # x_1_gen = self.face_pool(x_1_gen).detach().cpu()
                x_1_edit = self.face_pool(x_1_edit).detach().cpu()
                out_img = torch.cat((x_1, x_1_recon, x_1_edit), dim=3)
            utils.save_image(
                clip_img(out_img), log_dir + prefix + 'epoch_' + str(n_epoch) + '/iter_' +
                str(scale) + str(n_iter + 1) + '_0.png')
            # breakpoint()
            # utils.save_image(
            #     clip_img(patch), log_dir + prefix + 'epoch_' + str(n_epoch) + '/iter_' +
            #     str(n_iter + 1) + '_patch_0.png')

    def save_model(self, log_dir):
        torch.save(
            {
                'global_enc_state_dict': self.globEnc.state_dict(),
                'local_enc_state_dict': self.locEnc.state_dict(),
                "random_noise": self.randomized_noises,
            }, '{:s}/enc.pth.tar'.format(log_dir))

    def save_checkpoint(self, n_epoch, log_dir):
        checkpoint_state = {
            'n_epoch': n_epoch,
            'global_enc_state_dict': self.globEnc.state_dict(),
            'local_enc_state_dict': self.locEnc.state_dict(),
            'enc_opt_state_dict': self.enc_opt.state_dict(),
            'enc_scheduler_state_dict': self.enc_scheduler.state_dict(),
            "random_noise": self.randomized_noises,
        }
        torch.save(checkpoint_state, '{:s}/checkpoint_{:s}.pth'.format(log_dir, str(n_epoch)))

    def load_model(self, log_dir):
        self.enc.load_state_dict(torch.load('{:s}/enc.pth.tar'.format(log_dir)))

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.globEnc.load_state_dict(state_dict['global_enc_state_dict'])
        self.locEnc.load_state_dict(state_dict['local_enc_state_dict'])
        self.enc_opt.load_state_dict(state_dict['enc_opt_state_dict'])
        self.enc_scheduler.load_state_dict(state_dict['enc_scheduler_state_dict'])
        self.randomized_noises = state_dict['random_noise']
        return state_dict['n_epoch'] + 1

    def load_noise(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.randomized_noises = state_dict['random_noise']
        print(f"[*] Noise Loaded")

    def update(self, n_iter, img=None):
        self.n_iter = n_iter
        self.enc_opt.zero_grad()
        self.compute_loss(img=img, train=True).backward()
        self.enc_opt.step()

    # def log_test_loss(self):
    #     if self.isTest:
    #         if 'landmark' in self.config['w'] and self.config['w']['landmark'] > 0:
    #             self.landmark_lossAll += self.landmark_loss * self.num_test
    #         if 'l1' in self.config['w'] and self.config['w']['l1'] > 0:
    #             self.l1_lossAll += self.l1_loss * self.num_test
    #         self.id_lossAll += self.id_loss * self.num_test
    #         self.lpips_lossAll += self.lpips_loss * self.num_test
    #         self.l2_lossAll += self.l2_loss * self.num_test
    #         self.lossAll += self.loss
    #         self.cnt += self.num_test

    # def compute_test_loss(self, logger, n_iter, prefix='train'):
    #     # embed()
    #     logger.log_value(prefix + '/l2_loss', self.l2_lossAll.item() / self.cnt, n_iter + 1)
    #     logger.log_value(prefix + '/lpips_loss', self.lpips_lossAll.item() / self.cnt, n_iter + 1)
    #     logger.log_value(prefix + '/id_loss', self.id_lossAll.item() / self.cnt, n_iter + 1)
    #     logger.log_value(prefix + '/total_loss', self.lossAll.item() / self.cnt, n_iter + 1)
    #     if 'l1' in self.config['w'] and self.config['w']['l1'] > 0:
    #         logger.log_value(prefix + '/l1_loss', self.l1_lossAll.item() / self.cnt, n_iter + 1)
    #     if 'landmark' in self.config['w'] and self.config['w']['landmark'] > 0:
    #         logger.log_value(prefix + '/landmark_loss',
    #                          self.landmark_lossAll.item() / self.cnt, n_iter + 1)

    #     self.cnt = 0
    #     self.landmark_lossAll = 0
    #     self.l1_lossAll = 0
    #     self.id_lossAll = 0
    #     self.lpips_lossAll = 0
    #     self.l2_lossAll = 0
    #     self.lossAll = 0
    #     self.cntcnt += 1
