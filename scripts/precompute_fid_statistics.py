# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import os
import argparse
from fid.fid_score import compute_statistics_of_generator, save_statistics
from datasets import get_loaders_eval
from fid.inception import InceptionV3
from itertools import chain


def main(args):
    device = 'cuda'
    dims = 2048
    # for binary datasets including MNIST and OMNIGLOT, we don't apply binarization for FID computation
    train_queue, valid_queue, _ = get_loaders_eval(args.dataset, args)
    print('len train queue', len(train_queue), 'len val queue', len(valid_queue), 'batch size', args.batch_size)
    if args.dataset in {'celeba_256', 'omniglot'}:
        train_queue = chain(train_queue, valid_queue)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], model_dir=args.fid_dir).to(device)
    m, s = compute_statistics_of_generator(train_queue, model, args.batch_size, dims, device, args.max_samples)
    file_path = os.path.join(args.fid_dir, args.dataset + '.npz')
    print('saving fid stats at %s' % file_path)
    save_statistics(file_path, m, s)


if __name__ == '__main__':
    # python precompute_fid_statistics.py --dataset cifar10
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'celeba_64', 'celeba_256', 'omniglot', 'mnist',
                                 'imagenet_32', 'ffhq', 'lsun_bedroom_128', 'lsun_church_256'],
                        help='which dataset to use')
    parser.add_argument('--data', type=str, default='/tmp/nvae-diff/data',
                        help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size per GPU')
    parser.add_argument('--max_samples', type=int, default=50000,
                        help='batch size per GPU')
    parser.add_argument('--fid_dir', type=str, default='/tmp/fid-stats',
                        help='A dir to store fid related files')

    args = parser.parse_args()
    args.distributed = False

    main(args)