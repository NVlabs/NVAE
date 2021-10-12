# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import logging
import os
import shutil
import time
from datetime import timedelta
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist

import torch.nn.functional as F
from tensorboardX import SummaryWriter


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class ExpMovingAvgrageMeter(object):

    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.avg = 0

    def update(self, val):
        self.avg = (1. - self.momentum) * self.avg + self.momentum * val


class DummyDDP(nn.Module):
    def __init__(self, model):
        super(DummyDDP, self).__init__()
        self.module = model

    def forward(self, *input, **kwargs):
        return self.module(*input, **kwargs)


def count_parameters_in_M(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class Logger(object):
    def __init__(self, rank, save):
        # other libraries may set logging before arriving at this line.
        # by reloading logging, we can get rid of previous configs set by other libraries.
        from importlib import reload
        reload(logging)
        self.rank = rank
        if self.rank == 0:
            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                format=log_format, datefmt='%m/%d %I:%M:%S %p')
            fh = logging.FileHandler(os.path.join(save, 'log.txt'))
            fh.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(fh)
            self.start_time = time.time()

    def info(self, string, *args):
        if self.rank == 0:
            elapsed_time = time.time() - self.start_time
            elapsed_time = time.strftime(
                '(Elapsed: %H:%M:%S) ', time.gmtime(elapsed_time))
            if isinstance(string, str):
                string = elapsed_time + string
            else:
                logging.info(elapsed_time)
            logging.info(string, *args)


class Writer(object):
    def __init__(self, rank, save):
        self.rank = rank
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=save, flush_secs=20)

    def add_scalar(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_scalar(*args, **kwargs)

    def add_figure(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_figure(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_image(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_histogram(*args, **kwargs)

    def add_histogram_if(self, write, *args, **kwargs):
        if write and False:   # Used for debugging.
            self.add_histogram(*args, **kwargs)

    def close(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.close()


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def get_stride_for_cell_type(cell_type):
    if cell_type.startswith('normal') or cell_type.startswith('combiner'):
        stride = 1
    elif cell_type.startswith('down'):
        stride = 2
    elif cell_type.startswith('up'):
        stride = -1
    else:
        raise NotImplementedError(cell_type)

    return stride


def get_cout(cin, stride):
    if stride == 1:
        cout = cin
    elif stride == -1:
        cout = cin // 2
    elif stride == 2:
        cout = 2 * cin

    return cout


def kl_balancer_coeff(num_scales, groups_per_scale, fun):
    if fun == 'equal':
        coeff = torch.cat([torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)], dim=0).cuda()
    elif fun == 'linear':
        coeff = torch.cat([(2 ** i) * torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)], dim=0).cuda()
    elif fun == 'sqrt':
        coeff = torch.cat([np.sqrt(2 ** i) * torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)], dim=0).cuda()
    elif fun == 'square':
        coeff = torch.cat([np.square(2 ** i) / groups_per_scale[num_scales - i - 1] * torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)], dim=0).cuda()
    else:
        raise NotImplementedError
    # convert min to 1.
    coeff /= torch.min(coeff)
    return coeff


def kl_per_group(kl_all):
    kl_vals = torch.mean(kl_all, dim=0)
    kl_coeff_i = torch.abs(kl_all)
    kl_coeff_i = torch.mean(kl_coeff_i, dim=0, keepdim=True) + 0.01

    return kl_coeff_i, kl_vals


def kl_balancer(kl_all, kl_coeff=1.0, kl_balance=False, alpha_i=None):
    if kl_balance and kl_coeff < 1.0:
        alpha_i = alpha_i.unsqueeze(0)

        kl_all = torch.stack(kl_all, dim=1)
        kl_coeff_i, kl_vals = kl_per_group(kl_all)
        total_kl = torch.sum(kl_coeff_i)

        kl_coeff_i = kl_coeff_i / alpha_i * total_kl
        kl_coeff_i = kl_coeff_i / torch.mean(kl_coeff_i, dim=1, keepdim=True)
        kl = torch.sum(kl_all * kl_coeff_i.detach(), dim=1)

        # for reporting
        kl_coeffs = kl_coeff_i.squeeze(0)
    else:
        kl_all = torch.stack(kl_all, dim=1)
        kl_vals = torch.mean(kl_all, dim=0)
        kl = torch.sum(kl_all, dim=1)
        kl_coeffs = torch.ones(size=(len(kl_vals),))

    return kl_coeff * kl, kl_coeffs, kl_vals


def kl_coeff(step, total_step, constant_step, min_kl_coeff):
    return max(min((step - constant_step) / total_step, 1.0), min_kl_coeff)


def log_iw(decoder, x, log_q, log_p, crop=False):
    recon = reconstruction_loss(decoder, x, crop)
    return - recon - log_q + log_p


def reconstruction_loss(decoder, x, crop=False):
    from distributions import Normal, DiscMixLogistic

    recon = decoder.log_prob(x)
    if crop:
        recon = recon[:, :, 2:30, 2:30]
    
    if isinstance(decoder, DiscMixLogistic):
        return - torch.sum(recon, dim=[1, 2])    # summation over RGB is done.
    else:
        return - torch.sum(recon, dim=[1, 2, 3])


def tile_image(batch_image, n):
    assert n * n == batch_image.size(0)
    channels, height, width = batch_image.size(1), batch_image.size(2), batch_image.size(3)
    batch_image = batch_image.view(n, n, channels, height, width)
    batch_image = batch_image.permute(2, 0, 3, 1, 4)                              # n, height, n, width, c
    batch_image = batch_image.contiguous().view(channels, n * height, n * width)
    return batch_image


def average_gradients(params, is_distributed):
    """ Gradient averaging. """
    if is_distributed:
        size = float(dist.get_world_size())
        for param in params:
            if param.requires_grad:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= size


def average_params(params, is_distributed):
    """ parameter averaging. """
    if is_distributed:
        size = float(dist.get_world_size())
        for param in params:
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= size


def average_tensor(t, is_distributed):
    if is_distributed:
        size = float(dist.get_world_size())
        dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
        t.data /= size


def one_hot(indices, depth, dim):
    indices = indices.unsqueeze(dim)
    size = list(indices.size())
    size[dim] = depth
    y_onehot = torch.zeros(size).cuda()
    y_onehot.zero_()
    y_onehot.scatter_(dim, indices, 1)

    return y_onehot


def num_output(dataset):
    if dataset in {'mnist', 'omniglot'}:
        return 28 * 28
    elif dataset == 'cifar10':
        return 3 * 32 * 32
    elif dataset.startswith('celeba') or dataset.startswith('imagenet') or dataset.startswith('lsun'):
        size = int(dataset.split('_')[-1])
        return 3 * size * size
    elif dataset == 'ffhq':
        return 3 * 256 * 256
    else:
        raise NotImplementedError


def get_input_size(dataset):
    if dataset in {'mnist', 'omniglot'}:
        return 32
    elif dataset == 'cifar10':
        return 32
    elif dataset.startswith('celeba') or dataset.startswith('imagenet') or dataset.startswith('lsun'):
        size = int(dataset.split('_')[-1])
        return size
    elif dataset == 'ffhq':
        return 256
    else:
        raise NotImplementedError


def pre_process(x, num_bits):
    if num_bits != 8:
        x = torch.floor(x * 255 / 2 ** (8 - num_bits))
        x /= (2 ** num_bits - 1)
    return x


def get_arch_cells(arch_type):
    if arch_type == 'res_elu':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_elu', 'res_elu']
        arch_cells['down_enc'] = ['res_elu', 'res_elu']
        arch_cells['normal_dec'] = ['res_elu', 'res_elu']
        arch_cells['up_dec'] = ['res_elu', 'res_elu']
        arch_cells['normal_pre'] = ['res_elu', 'res_elu']
        arch_cells['down_pre'] = ['res_elu', 'res_elu']
        arch_cells['normal_post'] = ['res_elu', 'res_elu']
        arch_cells['up_post'] = ['res_elu', 'res_elu']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_bnelu':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnelu', 'res_bnelu']
        arch_cells['down_enc'] = ['res_bnelu', 'res_bnelu']
        arch_cells['normal_dec'] = ['res_bnelu', 'res_bnelu']
        arch_cells['up_dec'] = ['res_bnelu', 'res_bnelu']
        arch_cells['normal_pre'] = ['res_bnelu', 'res_bnelu']
        arch_cells['down_pre'] = ['res_bnelu', 'res_bnelu']
        arch_cells['normal_post'] = ['res_bnelu', 'res_bnelu']
        arch_cells['up_post'] = ['res_bnelu', 'res_bnelu']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_bnswish':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_dec'] = ['res_bnswish', 'res_bnswish']
        arch_cells['up_dec'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_post'] = ['res_bnswish', 'res_bnswish']
        arch_cells['up_post'] = ['res_bnswish', 'res_bnswish']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'mbconv_sep':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['mconv_e6k5g0']
        arch_cells['down_enc'] = ['mconv_e6k5g0']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['mconv_e3k5g0']
        arch_cells['down_pre'] = ['mconv_e3k5g0']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'mbconv_sep11':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['mconv_e6k11g0']
        arch_cells['down_enc'] = ['mconv_e6k11g0']
        arch_cells['normal_dec'] = ['mconv_e6k11g0']
        arch_cells['up_dec'] = ['mconv_e6k11g0']
        arch_cells['normal_pre'] = ['mconv_e3k5g0']
        arch_cells['down_pre'] = ['mconv_e3k5g0']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res53_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish5', 'res_bnswish']
        arch_cells['down_enc'] = ['res_bnswish5', 'res_bnswish']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['res_bnswish5', 'res_bnswish']
        arch_cells['down_pre'] = ['res_bnswish5', 'res_bnswish']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res35_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish', 'res_bnswish5']
        arch_cells['down_enc'] = ['res_bnswish', 'res_bnswish5']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['res_bnswish', 'res_bnswish5']
        arch_cells['down_pre'] = ['res_bnswish', 'res_bnswish5']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res55_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish5', 'res_bnswish5']
        arch_cells['down_enc'] = ['res_bnswish5', 'res_bnswish5']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['res_bnswish5', 'res_bnswish5']
        arch_cells['down_pre'] = ['res_bnswish5', 'res_bnswish5']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_mbconv9':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_dec'] = ['mconv_e6k9g0']
        arch_cells['up_dec'] = ['mconv_e6k9g0']
        arch_cells['normal_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_post'] = ['mconv_e3k9g0']
        arch_cells['up_post'] = ['mconv_e3k9g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'mbconv_res':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['mconv_e6k5g0']
        arch_cells['down_enc'] = ['mconv_e6k5g0']
        arch_cells['normal_dec'] = ['res_bnswish', 'res_bnswish']
        arch_cells['up_dec'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_pre'] = ['mconv_e3k5g0']
        arch_cells['down_pre'] = ['mconv_e3k5g0']
        arch_cells['normal_post'] = ['res_bnswish', 'res_bnswish']
        arch_cells['up_post'] = ['res_bnswish', 'res_bnswish']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'mbconv_den':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['mconv_e6k5g0']
        arch_cells['down_enc'] = ['mconv_e6k5g0']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['mconv_e3k5g8']
        arch_cells['down_pre'] = ['mconv_e3k5g8']
        arch_cells['normal_post'] = ['mconv_e3k5g8']
        arch_cells['up_post'] = ['mconv_e3k5g8']
        arch_cells['ar_nn'] = ['']
    else:
        raise NotImplementedError

    return arch_cells


def groups_per_scale(num_scales, num_groups_per_scale, is_adaptive, divider=2, minimum_groups=1):
    g = []
    n = num_groups_per_scale
    for s in range(num_scales):
        assert n >= 1
        g.append(n)
        if is_adaptive:
            n = n // divider
            n = max(minimum_groups, n)
    return g
