import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from torchvision import models, utils
from generator.infinity.custom_ops import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from generator.infinity.infinity_ops import Skip
from generator.infinity.infinity_ops import EqualLinear


class localLatentEncoder(nn.Module):

    def __init__(self, config):
        super(localLatentEncoder, self).__init__()

        self.config = config
        self.size = config.train_params.patch_size
        blur_kernel = [1, 2, 1]

        if config.train_params.training_modality == "patch":
            g_output_res = config.train_params.patch_size
        elif config.train_params.training_modality == "full":
            g_output_res = config.train_params.full_size
        else:
            raise NotImplementedError()

        if g_output_res == 101 and config.train_params.ts_input_size == 11:
            self.convs_specs = [
                dict(out_ch=512, downsample=True, styled=False),  # 161 -> 81
                # skip-node 0
                dict(out_ch=512, downsample=False, styled=False),  # 81 -> 81
                # skip-node 1
                dict(out_ch=512, downsample=True, styled=False),  # 81 -> 41
                # skip-node 2
                dict(out_ch=512, downsample=False, styled=True),  # 41 -> 41
                # skip-node 3
                dict(out_ch=512, downsample=True, styled=True),  # 41 -> 21
                # skip-node 4
                dict(out_ch=512, downsample=False, styled=True),  # 21 -> 21
                # skip-node 5
                dict(out_ch=512, downsample=True, styled=True),  # 21 -> 11
                # skip-node 6
                dict(out_ch=512, downsample=False, styled=True),  # 11 -> 11
                # skip-node 7
            ]
            self.skip_specs = [
                dict(src=1, tgt=3, downsample=True),
                dict(src=3, tgt=5, downsample=True),
                dict(src=5, tgt=7, downsample=True),
                dict(src=7, tgt=8, downsample=True),
            ]

        self.convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.noises = nn.Module()

        in_ch = 3

        for i, conv_spec in enumerate(self.convs_specs):

            self.convs.append(
                ActConv2d(
                    in_ch * 4,
                    conv_spec["out_ch"],
                    kernel_size=1,
                    stride=2 if conv_spec["downsample"] else 1,
                    padding=0,
                    one_side_padding=True),
            )
            in_ch = conv_spec["out_ch"]

        for skip_spec in self.skip_specs:
            src_conv_spec = self.convs_specs[skip_spec["src"]]
            in_ch = src_conv_spec["out_ch"]
            self.skips.append(
                Skip(
                    in_ch,
                    downsample=skip_spec["downsample"],
                    no_zero_pad=True,
                    blur_kernel=blur_kernel))

    def forward(self, patch_image=None):
        cur_skip_idx = 0
        skip = None
        h = patch_image
        # out = []
        for i, conv in enumerate(self.convs):
            B, C, H, W = h.shape
            h1 = h.clone()
            h2 = h.clone()
            h3 = h.clone()
            # h1[:, :, :H - 1, :W - 1] = h[:, :, 1:, 1:]
            # h2[:, :, :H - 1] = h[:, :, 1:]
            # h3[:, :, :, :W - 1] = h[:, :, :, 1:]
            # h1[:, :, 1:, 1:] = h[:, :, :H - 1, :W - 1]
            h1 = torch.roll(h1, (-1, -1), (2, 3))
            h2 = torch.roll(h2, -1, 2)
            h3 = torch.roll(h3, -1, 3)
            # h2[:, :, 1:] = h[:, :, :H - 1]
            # h3[:, :, :, 1:] = h[:, :, :, :W - 1]
            h_final = torch.cat([h, h1, h2, h3], dim=1)
            h = conv(h_final)
            # print(h.shape)
            # out.append(h)

            skip_spec = self.skip_specs[cur_skip_idx]
            skip_src = skip_spec["src"]
            skip_tgt = skip_spec["tgt"]
            if i == skip_src:
                skip_op = self.skips[cur_skip_idx]
                skip = skip_op(h, skip=skip)

                cur_skip_idx += 1

        latent = skip
        # out.append(skip)
        # return latent, out
        return latent


class ActConv2d(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 padding=0,
                 one_side_padding=False):
        super(ActConv2d, self).__init__()
        self.Conv2d = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=0)
        self.reside = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.one_side_pad = one_side_padding
        # if one_side_padding:
        #     self.rm_pad = padding // stride,
        self.pad_mode = "replicate"
        self.pad = padding
        self.activation = FusedLeakyReLU(out_ch)

    def forward(self, input_):
        input_ = F.pad(input_, (0, self.pad, 0, self.pad), mode=self.pad_mode)
        out = self.Conv2d(input_)
        out = self.activation(out)
        out = self.reside(out)
        out = self.activation(out)
        return out


class CondConv2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, one_side_padding=False):
        super(CondConv2d, self).__init__()
        self.Conv2d = StyledConv(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            style_dim=512,
            stride=stride,
            padding=padding,
            padding_mode="replicate")
        self.reside = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.one_side_pad = one_side_padding
        if one_side_padding:
            self.rm_pad = padding // stride
        self.activation = FusedLeakyReLU(out_ch)

    def forward(self, input_, cond):
        out = self.Conv2d(input_, cond)
        # if self.one_side_pad and self.rm_pad != 0:
        #     out = out[:, :, self.rm_pad:, self.rm_pad:]
        out = self.reside(out)
        out = self.activation(out)
        return out


class StyledConv(nn.Module):

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 2, 1],
        demodulate=True,
        stride=1,
        dilation=1,
        padding=0,
        padding_mode="replicate",
        activation="LeakyReLU",
    ):
        super().__init__()
        self.upsample = upsample

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
            stride=stride,
            dilation=dilation,
            padding=padding,
            padding_mode=padding_mode)

        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style):
        out = self.conv(input, style)
        out = self.activate(out)
        return out


class ModulatedConv2d(nn.Module):

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        stride=1,
        dilation=1,
        padding=0,
        padding_mode="replicate",
        blur_kernel=[1, 2, 1],
        no_zero_pad=False,
        config=None,
        side=None,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.style_dim = style_dim
        self.no_zero_pad = no_zero_pad
        self.config = config
        self.side = side
        self.verbose = True
        self.stride = stride

        fan_in = in_channel * kernel_size**2
        self.scale = 1 / math.sqrt(fan_in)
        factor = 2
        self.pad = padding
        self.pad_mode = padding_mode
        # assert kernel_size == 3, \
        #     "Lets assume kernel size = 3 first"
        # if len(blur_kernel) % 2 == 1:  # used
        #     pad0 = pad1 = len(blur_kernel) // 2
        # else:  # Original StyleGAN2
        #     p = (len(blur_kernel) - factor) + (kernel_size - 1)
        #     pad0 = (p + 1) // 2
        #     pad1 = p // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))

        self.demodulate = demodulate
        if style_dim > 0:
            self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        else:
            self.modulation = None

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        # Special case for spatially-shaped style
        # Here, we early justify whether the whole feature uses the same style.
        # If that's the case, we simply use the same style, otherwise, it will use another slower logic.
        if (style is not None) and (style.ndim == 4):
            mean_style = style.mean([2, 3], keepdim=True)
            is_mono_style = ((style - mean_style) < 1e-8).all()
            if is_mono_style:
                style = mean_style.squeeze()

        # Regular forward
        if style.ndim == 2:
            style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
            # if self.verbose:
            #     print(f"style dim 2 {style.shape}")
            #     self.verbose = False
            # (1, ) * (1, out_ch, in_ch, k, k) * (B, 1, in_ch, 1, 1)
            # => (B, out_ch, in_ch, k, k)
            weight = self.scale * self.weight * style

            if self.demodulate:
                demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
                weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

            weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size,
                                 self.kernel_size)

            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            input = F.pad(input, (0, self.pad, 0, self.pad), mode=self.pad_mode)
            out = F.conv2d(input, weight, padding=0, stride=self.stride, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            assert (not self.training), \
                "Only accepts spatially-shaped global-latent for testing-time manipulation!"
            assert (style.ndim == 4), \
                "Only considered BxCxHxW case, but got shape {}".format(style.shape)

            # For simplicity (and laziness), we sometimes feed spatial latents
            # that are larger than the input, center-crop for such kind of cases.
            style = self._auto_shape_align(source=style, target=input)

            # [Note]
            # Original (lossy expression):   input * (style * weight)
            # What we equivalently do here (still lossy): (input * style) * weight
            sb, sc, sh, sw = style.shape
            flat_style = style.permute(0, 2, 3, 1).reshape(-1, sc)  # (BxHxW, C)
            style_mod = self.modulation(flat_style)  # (BxHxW, C)
            style_mod = style_mod.view(sb, sh, sw, self.in_channel).permute(0, 3, 1,
                                                                            2)  # (B, C, H, W)

            input_st = (style_mod * input)  # (B, C, H, W)
            weight = self.scale * self.weight

            if self.demodulate:
                # [Hubert]
                # This will be an estimation if spatilly fused styles are different.
                # In practice, the interpolation of styles do not (numerically) change drastically, so the approximation here is invisible.
                """
                # This is the implementation we shown in the paper Appendix, the for-loop is slow.
                # But this version surely allocates a constant amount of memory.
                for i in range(sh):
                    for j in range(sw):
                        style_expand_s = style_mod[:, :, i, j].view(sb, 1, self.in_channel, 1, 1) # shape: (B, 1, in_ch, 1, 1)
                        simulated_weight_s = weight * style_expand_s # shape: (B, out_ch, in_ch, k, k)
                        demod_s[:, :, i, j] = torch.rsqrt(simulated_weight_s.pow(2).sum([2, 3, 4]) + 1e-8) # shape: (B, out_ch)
                """
                """
                Logically equivalent version, omits one for-loop by batching one spatial dimension.
                """
                demod = torch.zeros(sb, self.out_channel, sh, sw).to(style.device)
                for i in range(sh):
                    style_expand = style_mod[:, :, i, :].view(sb, 1, self.in_channel,
                                                              sw).pow(2)  # shape: (B, 1, in_ch, W)
                    weight_expand = weight.pow(2).sum([3, 4]).unsqueeze(
                        -1)  # shape: (B, out_ch, in_ch, 1)
                    simulated_weight = weight_expand * style_expand  # shape: (B, out_ch, in_ch, W)
                    demod[:, :, i, :] = torch.rsqrt(simulated_weight.sum(2) +
                                                    1e-8)  # shape: (B, out_ch, W)
                """
                # An even faster version that batches both height and width dimension, but allocates too much memory that is impractical in reality.
                # For instance, it allocates 40GB memory with shape (8, 512, 128, 3, 3, 31, 31).
                style_expand = style_mod.view(sb, 1, self.in_channel, 1, 1, sh, sw) # (B,      1  in_ch, 1, 1, H, W)
                weight_expand = weight.unsqueeze(5).unsqueeze(6)                    # (B, out_ch, in_ch, k, k, 1, 1)
                simulated_weight = weight_expand * style_expand # shape: (B, out_ch, in_ch, k, k, H, W)
                demod = torch.rsqrt(simulated_weight.pow(2).sum([2, 3, 4]) + 1e-8) # shape: (B, out_ch, H, W)
                """
                """
                # Just FYI. If you use the mean style over the patch, it creates blocky artifacts
                mean_style = style_mod.mean([2,3]).view(sb, 1, self.in_channel, 1, 1)
                simulated_weight_ = weight * mean_style # shape: (B, out_ch, in_ch, k, k)
                demod_ = torch.rsqrt(simulated_weight_.pow(2).sum([2, 3, 4]) + 1e-8)
                demod_ = demod_.unsqueeze(2).unsqueeze(3)
                """

            weight = weight.view(self.out_channel, in_channel, self.kernel_size, self.kernel_size)

            input = F.pad(input_st, (0, self.pad, 0, self.pad), mode=self.pad_mode)
            out = F.conv2d(input_st, weight, padding=self.padding, stride=self.stride, groups=1)
            if self.demodulate:
                raise NotImplementedError("Unused, not implemented!")
                out = out * demod

            out = out.contiguous()  # Don't know where causes discontiguity.

        return out
