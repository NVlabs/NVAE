# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the PyTorch library.
#
# Source:
# https://github.com/pytorch/pytorch/blob/881c1adfcd916b6cd5de91bc343eb86aff88cc80/torch/nn/modules/batchnorm.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_PyTorch). The modifications
# to this file are subject to the NVIDIA Source Code License for
# NVAE located at the root directory.
# ---------------------------------------------------------------

from __future__ import division

import torch
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F

from .functions import SyncBatchNorm as sync_batch_norm
from .swish import Swish as swish


class SyncBatchNormSwish(_BatchNorm):
    r"""Applies Batch Normalization over a N-Dimensional input (a mini-batch of [N-2]D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over all
    mini-batches of the same process groups. :math:`\gamma` and :math:`\beta`
    are learnable parameter vectors of size `C` (where `C` is the input size).
    By default, the elements of :math:`\gamma` are sampled from
    :math:`\mathcal{U}(0, 1)` and the elements of :math:`\beta` are set to 0.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momemtum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, +)` slices, it's common terminology to call this Volumetric Batch Normalization
    or Spatio-temporal Batch Normalization.

    Currently SyncBatchNorm only supports DistributedDataParallel with single GPU per process. Use
    torch.nn.SyncBatchNorm.convert_sync_batchnorm() to convert BatchNorm layer to SyncBatchNorm before wrapping
    Network with DDP.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, +)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``
        process_group: synchronization of stats happen within each process group
            individually. Default behavior is synchronization across the whole
            world

    Shape:
        - Input: :math:`(N, C, +)`
        - Output: :math:`(N, C, +)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.SyncBatchNorm(100)
        >>> # creating process group (optional)
        >>> # process_ids is a list of int identifying rank ids.
        >>> process_group = torch.distributed.new_group(process_ids)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm3d(100, affine=False, process_group=process_group)
        >>> input = torch.randn(20, 100, 35, 45, 10)
        >>> output = m(input)

        >>> # network is nn.BatchNorm layer
        >>> sync_bn_network = nn.SyncBatchNorm.convert_sync_batchnorm(network, process_group)
        >>> # only single gpu per process is currently supported
        >>> ddp_sync_bn_network = torch.nn.parallel.DistributedDataParallel(
        >>>                         sync_bn_network,
        >>>                         device_ids=[args.local_rank],
        >>>                         output_device=args.local_rank)

    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, process_group=None):
        super(SyncBatchNormSwish, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.process_group = process_group
        # gpu_size is set through DistributedDataParallel initialization. This is to ensure that SyncBatchNorm is used
        # under supported condition (single GPU per process)
        self.ddp_gpu_size = None

    def _check_input_dim(self, input):
        if input.dim() < 2:
            raise ValueError('expected at least 2D input (got {}D input)'
                             .format(input.dim()))

    def _specify_ddp_gpu_num(self, gpu_size):
        if gpu_size > 1:
            raise ValueError('SyncBatchNorm is only supported for DDP with single GPU per process')
        self.ddp_gpu_size = gpu_size

    def forward(self, input):
        # currently only GPU input is supported
        if not input.is_cuda:
            raise ValueError('SyncBatchNorm expected input tensor to be on GPU')

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        need_sync = self.training or not self.track_running_stats
        if need_sync:
            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        # fallback to framework BN when synchronization is not necessary
        if not need_sync:
            out = F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
            return swish.apply(out)
        else:
            # av: I only use it in this setting.
            if not self.ddp_gpu_size and False:
                raise AttributeError('SyncBatchNorm is only supported within torch.nn.parallel.DistributedDataParallel')

            return sync_batch_norm.apply(
                input, self.weight, self.bias, self.running_mean, self.running_var,
                self.eps, exponential_average_factor, process_group, world_size)