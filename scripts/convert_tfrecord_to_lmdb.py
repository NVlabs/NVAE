# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import argparse
import torch
import lmdb
import os

from tfrecord.torch.dataset import TFRecordDataset


def main(dataset, split, tfr_path, lmdb_path):
    assert split in {'train', 'validation'}

    # create target directory
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path, exist_ok=True)
    if dataset == 'celeba' and split in {'train', 'validation'}:
        num_shards = {'train': 120, 'validation': 40}[split]
        lmdb_path = os.path.join(lmdb_path, '%s.lmdb' % split)
        tfrecord_path_template = os.path.join(tfr_path, '%s/%s-r08-s-%04d-of-%04d.tfrecords')
    elif dataset == 'imagenet-oord_32':
        num_shards = {'train': 2000, 'validation': 80}[split]
        # imagenet_oord_lmdb_path += '_32'
        lmdb_path = os.path.join(lmdb_path, '%s.lmdb' % split)
        tfrecord_path_template = os.path.join(tfr_path, '%s/%s-r05-s-%04d-of-%04d.tfrecords')
    elif dataset == 'imagenet-oord_64':
        num_shards = {'train': 2000, 'validation': 80}[split]
        # imagenet_oord_lmdb_path += '_64'
        lmdb_path = os.path.join(lmdb_path, '%s.lmdb' % split)
        tfrecord_path_template = os.path.join(tfr_path, '%s/%s-r06-s-%04d-of-%04d.tfrecords')
    else:
        raise NotImplementedError

    # create lmdb
    env = lmdb.open(lmdb_path, map_size=1e12)
    count = 0
    with env.begin(write=True) as txn:
        for tf_ind in range(num_shards):
            # read tf_record
            tfrecord_path = tfrecord_path_template % (split, split, tf_ind, num_shards)
            index_path = None
            description = {'shape': 'int', 'data': 'byte', 'label': 'int'}
            dataset = TFRecordDataset(tfrecord_path, index_path, description)
            loader = torch.utils.data.DataLoader(dataset, batch_size=1)

            # put the data in lmdb
            for data in loader:
                im = data['data'][0].cpu().numpy()
                txn.put(str(count).encode(), im)
                count += 1
                if count % 100 == 0:
                    print(count)

        print('added %d items to the LMDB dataset.' % count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LMDB creator using TFRecords from GLOW.')
    # experimental results
    parser.add_argument('--dataset', type=str, default='imagenet-oord_32',
                        help='dataset name', choices=['imagenet-oord_32', 'imagenet-oord_32', 'celeba'])
    parser.add_argument('--tfr_path', type=str, default='/data1/datasets/imagenet-oord/mnt/host/imagenet-oord-tfr',
                        help='location of TFRecords')
    parser.add_argument('--lmdb_path', type=str, default='/data1/datasets/imagenet-oord/imagenet-oord-lmdb_32',
                        help='target location for storing lmdb files')
    parser.add_argument('--split', type=str, default='train',
                        help='training or validation split', choices=['train', 'validation'])
    args = parser.parse_args()
    main(args.dataset, args.split, args.tfr_path, args.lmdb_path)



