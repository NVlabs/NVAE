# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict

from neural_operations import ConvBNSwish, normalize_weight_jit

AROPS = OrderedDict([
    ('conv_3x3', lambda C, masked, zero_diag: ELUConv(C, C, 3, 1, 1, masked=masked, zero_diag=zero_diag))
])


class Identity(nn.Module):
    def __init__(self, masked, zero_diag):
        super(Identity, self).__init__()
        if zero_diag:
            raise ValueError('Skip connection with zero diag is just a zero operation.')

    def forward(self, x):
        return x


def channel_mask(c_in, g_in, c_out, zero_diag):
    assert c_in % c_out == 0 or c_out % c_in == 0, "%d - %d" % (c_in, c_out)
    assert g_in == 1 or g_in == c_in

    if g_in == 1:
        mask = np.ones([c_out, c_in], dtype=np.float32)
        if c_out >= c_in:
            ratio = c_out // c_in
            for i in range(c_in):
                mask[i * ratio:(i + 1) * ratio, i + 1:] = 0
                if zero_diag:
                    mask[i * ratio:(i + 1) * ratio, i:i + 1] = 0
        else:
            ratio = c_in // c_out
            for i in range(c_out):
                mask[i:i + 1, (i + 1) * ratio:] = 0
                if zero_diag:
                    mask[i:i + 1, i * ratio:(i + 1) * ratio:] = 0
    elif g_in == c_in:
        mask = np.ones([c_out, c_in // g_in], dtype=np.float32)
        if zero_diag:
            mask = 0. * mask

    return mask


def create_conv_mask(kernel_size, c_in, g_in, c_out, zero_diag, mirror):
    m = (kernel_size - 1) // 2
    mask = np.ones([c_out, c_in // g_in, kernel_size, kernel_size], dtype=np.float32)
    mask[:, :, m:, :] = 0
    mask[:, :, m, :m] = 1
    mask[:, :, m, m] = channel_mask(c_in, g_in, c_out, zero_diag)
    if mirror:
        mask = np.copy(mask[:, :, ::-1, ::-1])
    return mask


def norm(t, dim):
    return torch.sqrt(torch.sum(t * t, dim))


class ARConv2d(nn.Conv2d):
    """Allows for weights as input."""

    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 masked=False, zero_diag=False, mirror=False):
        """
        Args:
            use_shared (bool): Use weights for this layer or not?
        """
        super(ARConv2d, self).__init__(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias)

        self.masked = masked
        if self.masked:
            assert kernel_size % 2 == 1, 'kernel size should be an odd value.'
            self.mask = torch.from_numpy(create_conv_mask(kernel_size, C_in, groups, C_out, zero_diag, mirror)).cuda()
            init_mask = self.mask.cpu()
        else:
            self.mask = 1.0
            init_mask = 1.0

        # init weight normalizaition parameters
        init = torch.log(norm(self.weight * init_mask, dim=[1, 2, 3]).view(-1, 1, 1, 1) + 1e-2)
        self.log_weight_norm = nn.Parameter(init, requires_grad=True)
        self.weight_normalized = None

    def normalize_weight(self):
        weight = self.weight
        if self.masked:
            assert self.mask.size() == weight.size()
            weight = weight * self.mask

        # weight normalization
        weight = normalize_weight_jit(self.log_weight_norm, weight)
        return weight

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): of size (B, C_in, H, W).
            params (ConvParam): containing `weight` and `bias` (optional) of conv operation.
        """
        self.weight_normalized = self.normalize_weight()
        bias = self.bias
        return F.conv2d(x, self.weight_normalized, bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ELUConv(nn.Module):
    """ReLU + Conv2d + BN."""

    def __init__(self, C_in, C_out, kernel_size, padding=0, dilation=1, masked=True, zero_diag=True,
                 weight_init_coeff=1.0, mirror=False):
        super(ELUConv, self).__init__()
        self.conv_0 = ARConv2d(C_in, C_out, kernel_size, stride=1, padding=padding, bias=True, dilation=dilation,
                               masked=masked, zero_diag=zero_diag, mirror=mirror)
        # change the initialized log weight norm
        self.conv_0.log_weight_norm.data += np.log(weight_init_coeff)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): of size (B, C_in, H, W)
        """
        out = F.elu(x)
        out = self.conv_0(out)
        return out


class ARInvertedResidual(nn.Module):
    def __init__(self, inz, inf, ex=6, dil=1, k=5, mirror=False):
        super(ARInvertedResidual, self).__init__()
        hidden_dim = int(round(inz * ex))
        padding = dil * (k - 1) // 2
        layers = []
        layers.extend([ARConv2d(inz, hidden_dim, kernel_size=3, padding=1, masked=True, mirror=mirror, zero_diag=True),
                       nn.ELU(inplace=True)])
        layers.extend([ARConv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=k, padding=padding, dilation=dil,
                                masked=True, mirror=mirror, zero_diag=False),
                      nn.ELU(inplace=True)])
        self.convz = nn.Sequential(*layers)
        self.hidden_dim = hidden_dim

    def forward(self, z, ftr):
        z = self.convz(z)
        return z


class MixLogCDFParam(nn.Module):
    def __init__(self, num_z, num_mix, num_ftr, mirror):
        super(MixLogCDFParam, self).__init__()

        num_out = num_z * (3 * num_mix + 3)
        self.conv = ELUConv(num_ftr, num_out, kernel_size=1, padding=0, masked=True, zero_diag=False,
                            weight_init_coeff=0.1, mirror=mirror)
        self.num_z = num_z
        self.num_mix = num_mix

    def forward(self, ftr):
        out = self.conv(ftr)
        b, c, h, w = out.size()
        out = out.view(b, self.num_z, c // self.num_z,  h, w)
        m = self.num_mix
        logit_pi, mu, log_s, log_a, b, _ = torch.split(out, [m, m, m, 1, 1, 1], dim=2)  # the last one is dummy
        return logit_pi, mu, log_s, log_a, b


def mix_log_cdf_flow(z1, logit_pi, mu, log_s, log_a, b):
    # z         b, n, 1, h, w
    # logit_pi  b, n, k, h, w
    # mu        b, n, k, h, w
    # log_s     b, n, k, h, w
    # log_a     b, n, 1, h, w
    # b         b, n, 1, h, w

    log_s = torch.clamp(log_s, min=-7)

    z = z1.unsqueeze(dim=2)
    log_pi = torch.log_softmax(logit_pi, dim=2)  # normalize log_pi
    u = - (z - mu) * torch.exp(-log_s)
    softplus_u = F.softplus(u)
    log_mix_cdf = log_pi - softplus_u
    log_one_minus_mix_cdf = log_mix_cdf + u
    log_mix_cdf = torch.logsumexp(log_mix_cdf, dim=2)
    log_one_minus_mix_cdf = torch.logsumexp(log_one_minus_mix_cdf, dim=2)

    log_a = log_a.squeeze_(dim=2)
    b = b.squeeze_(dim=2)
    new_z = torch.exp(log_a) * (log_mix_cdf - log_one_minus_mix_cdf) + b

    # compute log determinant Jac
    log_mix_pdf = torch.logsumexp(log_pi + u - log_s - 2 * softplus_u, dim=2)
    log_det = log_a - log_mix_cdf - log_one_minus_mix_cdf + log_mix_pdf

    return new_z, log_det
