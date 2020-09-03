# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the PyTorch library.
#
# Source:
# https://github.com/pytorch/pytorch/blob/2a54533c64c409b626b6c209ed78258f67aec194/torch/nn/modules/_functions.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_PyTorch). The modifications
# to this file are subject to the NVIDIA Source Code License for
# NVAE located at the root directory.
# ---------------------------------------------------------------

import torch
from torch.autograd.function import Function
import torch.distributed as dist


class SyncBatchNorm(Function):

    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
        input = input.contiguous()

        count = torch.empty(1,
                            dtype=running_mean.dtype,
                            device=input.device).fill_(input.numel() // input.size(1))

        # calculate mean/invstd for input.
        mean, invstd = torch.batch_norm_stats(input, eps)

        num_channels = input.shape[1]
        # C, C, 1 -> (2C + 1)
        combined = torch.cat([mean, invstd, count], dim=0)
        # world_size * (2C + 1)
        combined_list = [
            torch.empty_like(combined) for k in range(world_size)
        ]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(combined_list, combined, async_op=False)
        combined = torch.stack(combined_list, dim=0)
        # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
        mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)

        size = count_all.view(-1).long().sum()
        if size == 1:
            raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))

        # calculate global mean & invstd
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            count_all.view(-1)
        )

        self.save_for_backward(input, weight, mean, invstd, bias, count_all)
        self.process_group = process_group

        # apply element-wise normalization
        out = torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)

        # av: apply swish
        assert eps == 1e-5, "I assumed below that eps is 1e-5"
        out = out * torch.sigmoid(out)
        # av: end

        return out

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd, bias, count_tensor = self.saved_tensors

        # av: re-compute batch normalized out
        eps = 1e-5
        out = torch.batch_norm_elemt(saved_input, weight, bias, mean, invstd, eps)
        sigmoid_out = torch.sigmoid(out)
        grad_output *= (sigmoid_out * (1 + out * (1 - sigmoid_out)))
        # av: end

        grad_input = grad_weight = grad_bias = None
        process_group = self.process_group

        # calculate local stats as well as grad_weight / grad_bias
        sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
            grad_output,
            saved_input,
            mean,
            invstd,
            weight,
            self.needs_input_grad[0],
            self.needs_input_grad[1],
            self.needs_input_grad[2]
        )

        if self.needs_input_grad[0]:
            # synchronizing stats used to calculate input gradient.
            # TODO: move div_ into batch_norm_backward_elemt kernel
            num_channels = sum_dy.shape[0]
            combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
            torch.distributed.all_reduce(
                combined, torch.distributed.ReduceOp.SUM, process_group, async_op=False)
            sum_dy, sum_dy_xmu = torch.split(combined, num_channels)

            divisor = count_tensor.sum()
            mean_dy = sum_dy / divisor
            mean_dy_xmu = sum_dy_xmu / divisor
            # backward pass for gradient calculation
            grad_input = torch.batch_norm_backward_elemt(
                grad_output,
                saved_input,
                mean,
                invstd,
                weight,
                mean_dy,
                mean_dy_xmu
            )

        # synchronizing of grad_weight / grad_bias is not needed as distributed
        # training would handle all reduce.
        if weight is None or not self.needs_input_grad[1]:
            grad_weight = None

        if weight is None or not self.needs_input_grad[2]:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None
