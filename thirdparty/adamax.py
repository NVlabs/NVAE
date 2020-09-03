# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the PyTorch library.
#
# Source:
# https://github.com/pytorch/pytorch/blob/6e2bb1c05442010aff90b413e21fce99f0393727/torch/optim/adamax.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_PyTorch). The modifications
# to this file are subject to the NVIDIA Source Code License for
# NVAE located at the root directory.
# ---------------------------------------------------------------

import torch
from torch.optim import Optimizer

@torch.jit.script
def fusion1(exp_avg :torch.Tensor, grad :torch.Tensor, beta1: float):
    return exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)


class Adamax(Optimizer):
    """Implements Adamax algorithm (a variant of Adam based on infinity norm).

    It has been proposed in `Adam: A Method for Stochastic Optimization`__.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    __ https://arxiv.org/abs/1412.6980
    """

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adamax, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        params, grads, exp_avg, exp_inf = {},{},{},{}

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adamax does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # state['exp_inf'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_inf'] = torch.zeros_like(p.data)

                state['step'] += 1

                if p.shape not in params:
                    params[p.shape]  = {'idx': 0, 'data': []}
                    grads[p.shape]   = []
                    exp_avg[p.shape] = []
                    exp_inf[p.shape] = []

                params[p.shape]['data'].append(p.data)
                grads[p.shape].append(grad)
                exp_avg[p.shape].append(state['exp_avg'])
                exp_inf[p.shape].append(state['exp_inf'])

        for i in params:
            params[i]['data'] = torch.stack(params[i]['data'], dim=0)
            grads[i]          = torch.stack(grads[i], dim=0)
            exp_avg[i]        = torch.stack(exp_avg[i], dim=0)
            exp_inf[i]        = torch.stack(exp_inf[i], dim=0)

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            bias_correction = 1 - beta1 ** self.state[group['params'][0]]['step']
            clr = group['lr'] / bias_correction

            for i in params:
                if group['weight_decay'] != 0:
                    grads[i] = grads[i].add_(params[i]['data'], alpha=group['weight_decay'])
                # Update biased first moment estimate.
                exp_avg[i].mul_(beta1).add_(grads[i], alpha=1 - beta1)
                # Update the exponentially weighted infinity norm.
                torch.max(exp_inf[i].mul_(beta2), grads[i].abs_().add_(eps), out=exp_inf[i])
                params[i]['data'].addcdiv_(exp_avg[i], exp_inf[i], value=-clr)

        for group in self.param_groups:
            for p in group['params']:
                idx = params[p.shape]['idx']
                p.data = params[p.shape]['data'][idx, :]
                self.state[p]['exp_avg'] = exp_avg[p.shape][idx, :]
                self.state[p]['exp_inf'] = exp_inf[p.shape][idx, :]
                params[p.shape]['idx'] += 1

        return loss
