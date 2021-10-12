# The Official PyTorch Implementation of "NVAE: A Deep Hierarchical Variational Autoencoder" [(NeurIPS 2020 Spotlight Paper)](https://arxiv.org/abs/2007.03898)

<div align="center">
  <a href="http://latentspace.cc/arash_vahdat/" target="_blank">Arash&nbsp;Vahdat</a> &emsp; <b>&middot;</b> &emsp;
  <a href="http://jankautz.com/" target="_blank">Jan&nbsp;Kautz</a> 
</div>
<br>
<br>

[NVAE](https://arxiv.org/abs/2007.03898) is a deep hierarchical variational autoencoder that enables training SOTA 
likelihood-based generative models on several image datasets.

<p align="center">
    <img src="img/celebahq.png" width="800">
</p>

## Requirements
NVAE is built in Python 3.7 using PyTorch 1.6.0. Use the following command to install the requirements:
```
pip install -r requirements.txt
``` 

## Set up file paths and data
We have examined NVAE on several datasets. For large datasets, we store the data in LMDB datasets
for I/O efficiency. Click below on each dataset to see how you can prepare your data. Below, `$DATA_DIR` indicates
the path to a data directory that will contain all the datasets and `$CODE_DIR` refers to the code directory:

<details><summary>MNIST and CIFAR-10</summary>

These datasets will be downloaded automatically, when you run the main training for NVAE using `train.py`
for the first time. You can use `--data=$DATA_DIR/mnist` or `--data=$DATA_DIR/cifar10`, so that the datasets
are downloaded to the corresponding directories.
</details>

<details><summary>CelebA 64</summary>
Run the following commands to download the CelebA images and store them in an LMDB dataset:

```shell script
cd $CODE_DIR/scripts
python create_celeba64_lmdb.py --split train --img_path $DATA_DIR/celeba_org --lmdb_path $DATA_DIR/celeba64_lmdb
python create_celeba64_lmdb.py --split valid --img_path $DATA_DIR/celeba_org --lmdb_path $DATA_DIR/celeba64_lmdb
python create_celeba64_lmdb.py --split test  --img_path $DATA_DIR/celeba_org --lmdb_path $DATA_DIR/celeba64_lmdb
```
Above, the images will be downloaded to `$DATA_DIR/celeba_org` automatically and then then LMDB datasets are created
at `$DATA_DIR/celeba64_lmdb`.
</details>
 
<details><summary>ImageNet 32x32</summary>

Run the following commands to download tfrecord files from [GLOW](https://github.com/openai/glow) and to convert them
to LMDB datasets
```shell script
mkdir -p $DATA_DIR/imagenet-oord
cd $DATA_DIR/imagenet-oord
wget https://storage.googleapis.com/glow-demo/data/imagenet-oord-tfr.tar
tar -xvf imagenet-oord-tfr.tar
cd $CODE_DIR/scripts
python convert_tfrecord_to_lmdb.py --dataset=imagenet-oord_32 --tfr_path=$DATA_DIR/imagenet-oord/mnt/host/imagenet-oord-tfr --lmdb_path=$DATA_DIR/imagenet-oord/imagenet-oord-lmdb_32 --split=train
python convert_tfrecord_to_lmdb.py --dataset=imagenet-oord_32 --tfr_path=$DATA_DIR/imagenet-oord/mnt/host/imagenet-oord-tfr --lmdb_path=$DATA_DIR/imagenet-oord/imagenet-oord-lmdb_32 --split=validation
```
</details>

<details><summary>CelebA HQ 256</summary>

Run the following commands to download tfrecord files from [GLOW](https://github.com/openai/glow) and to convert them
to LMDB datasets
```shell script
mkdir -p $DATA_DIR/celeba
cd $DATA_DIR/celeba
wget https://storage.googleapis.com/glow-demo/data/celeba-tfr.tar
tar -xvf celeba-tfr.tar
cd $CODE_DIR/scripts
python convert_tfrecord_to_lmdb.py --dataset=celeba --tfr_path=$DATA_DIR/celeba/celeba-tfr --lmdb_path=$DATA_DIR/celeba/celeba-lmdb --split=train
python convert_tfrecord_to_lmdb.py --dataset=celeba --tfr_path=$DATA_DIR/celeba/celeba-tfr --lmdb_path=$DATA_DIR/celeba/celeba-lmdb --split=validation
```
</details>


<details><summary>FFHQ 256</summary>

Visit [this Google drive location](https://drive.google.com/drive/folders/1WocxvZ4GEZ1DI8dOz30aSj2zT6pkATYS) and download
`images1024x1024.zip`. Run the following commands to unzip the images and to store them in LMDB datasets:
```shell script
mkdir -p $DATA_DIR/ffhq
unzip images1024x1024.zip -d $DATA_DIR/ffhq/
cd $CODE_DIR/scripts
python create_ffhq_lmdb.py --ffhq_img_path=$DATA_DIR/ffhq/images1024x1024/ --ffhq_lmdb_path=$DATA_DIR/ffhq/ffhq-lmdb --split=train
python create_ffhq_lmdb.py --ffhq_img_path=$DATA_DIR/ffhq/images1024x1024/ --ffhq_lmdb_path=$DATA_DIR/ffhq/ffhq-lmdb --split=validation
```
</details>

<details><summary>LSUN</summary>

We use LSUN datasets in our follow-up works. Visit [LSUN](https://www.yf.io/p/lsun) for 
instructions on how to download this dataset. Since the LSUN scene datasets come in the
LMDB format, they are ready to be loaded using torchvision data loaders.

</details>


## Running the main NVAE training and evaluation scripts
We use the following commands on each dataset for training NVAEs on each dataset for 
Table 1 in the [paper](https://arxiv.org/pdf/2007.03898.pdf). In all the datasets but MNIST
normalizing flows are enabled. Check Table 6 in the paper for more information on training
details. Note that for the multinode training (more than 8-GPU experiments), we use the `mpirun` 
command to run the training scripts on multiple nodes. Please adjust the commands below according to your setup. 
Below `IP_ADDR` is the IP address of the machine that will host the process with rank 0 
(see [here](https://pytorch.org/tutorials/intermediate/dist_tuto.html#initialization-methods)). 
`NODE_RANK` is the index of each node among all the nodes that are running the job.

<details><summary>MNIST</summary>

Two 16-GB V100 GPUs are used for training NVAE on dynamically binarized MNIST. Training takes about 21 hours.

```shell script
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR
cd $CODE_DIR
python train.py --data $DATA_DIR/mnist --root $CHECKPOINT_DIR --save $EXPR_ID --dataset mnist --batch_size 200 \
        --epochs 400 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 3 --num_preprocess_cells 3 \
        --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 20 --num_preprocess_blocks 2 \
        --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 32 --num_channels_dec 32 --num_nf 0 \
        --ada_groups --num_process_per_node 2 --use_se --res_dist --fast_adamax 
```
</details>

<details><summary>CIFAR-10</summary>

Eight 16-GB V100 GPUs are used for training NVAE on CIFAR-10. Training takes about 55 hours.

```shell script
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR
cd $CODE_DIR
python train.py --data $DATA_DIR/cifar10 --root $CHECKPOINT_DIR --save $EXPR_ID --dataset cifar10 \
        --num_channels_enc 128 --num_channels_dec 128 --epochs 400 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_latent_scales 1 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
        --num_preprocess_blocks 1 --num_postprocess_blocks 1 --num_groups_per_scale 30 --batch_size 32 \
        --weight_decay_norm 1e-2 --num_nf 1 --num_process_per_node 8 --use_se --res_dist --fast_adamax 
```
</details>

<details><summary>CelebA 64</summary>

Eight 16-GB V100 GPUs are used for training NVAE on CelebA 64. Training takes about 92 hours.

```shell script
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR
cd $CODE_DIR
python train.py --data $DATA_DIR/celeba64_lmdb --root $CHECKPOINT_DIR --save $EXPR_ID --dataset celeba_64 \
        --num_channels_enc 64 --num_channels_dec 64 --epochs 90 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_latent_scales 3 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
        --num_preprocess_blocks 1 --num_postprocess_blocks 1 --weight_decay_norm 1e-1 --num_groups_per_scale 20 \
        --batch_size 16 --num_nf 1 --ada_groups --num_process_per_node 8 --use_se --res_dist --fast_adamax
```
</details>

<details><summary>ImageNet 32x32</summary>

24 16-GB V100 GPUs are used for training NVAE on ImageNet 32x32. Training takes about 70 hours.

```shell script
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR
export IP_ADDR=IP_ADDRESS
export NODE_RANK=NODE_RANK_BETWEEN_0_TO_2
cd $CODE_DIR
mpirun --allow-run-as-root -np 3 -npernode 1 bash -c \
        'python train.py --data $DATA_DIR/imagenet-oord/imagenet-oord-lmdb_32 --root $CHECKPOINT_DIR --save $EXPR_ID --dataset imagenet_32 \
        --num_channels_enc 192 --num_channels_dec 192 --epochs 45 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_latent_scales 1 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
        --num_preprocess_blocks 1 --num_postprocess_blocks 1 --num_groups_per_scale 28 \
        --batch_size 24 --num_nf 1 --warmup_epochs 1 \
        --weight_decay_norm 1e-2 --weight_decay_norm_anneal --weight_decay_norm_init 1e0 \
        --num_process_per_node 8 --use_se --res_dist \
        --fast_adamax --node_rank $NODE_RANK --num_proc_node 3 --master_address $IP_ADDR '
```
</details>

<details><summary>CelebA HQ 256</summary>

24 32-GB V100 GPUs are used for training NVAE on CelebA HQ 256. Training takes about 94 hours.

```shell script
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR
export IP_ADDR=IP_ADDRESS
export NODE_RANK=NODE_RANK_BETWEEN_0_TO_2
cd $CODE_DIR
mpirun --allow-run-as-root -np 3 -npernode 1 bash -c \
        'python train.py --data $DATA_DIR/celeba/celeba-lmdb --root $CHECKPOINT_DIR --save $EXPR_ID --dataset celeba_256 \
        --num_channels_enc 30 --num_channels_dec 30 --epochs 300 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_latent_scales 5 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
        --num_preprocess_blocks 1 --num_postprocess_blocks 1 --weight_decay_norm 1e-2 --num_groups_per_scale 16 \
        --batch_size 4 --num_nf 2 --ada_groups --min_groups_per_scale 4 \
        --weight_decay_norm_anneal --weight_decay_norm_init 1. --num_process_per_node 8 --use_se --res_dist \
        --fast_adamax --num_x_bits 5 --node_rank $NODE_RANK --num_proc_node 3 --master_address $IP_ADDR '
```

In our early experiments, a smaller model with 24 channels instead of 30, could be trained on only 8 GPUs in 
the same time (with the batch size of 6). The smaller models obtain only 0.01 bpd higher 
negative log-likelihood.
</details>

<details><summary>FFHQ 256</summary>

24 32-GB V100 GPUs are used for training NVAE on FFHQ 256. Training takes about 160 hours. 

```shell script
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR
export IP_ADDR=IP_ADDRESS
export NODE_RANK=NODE_RANK_BETWEEN_0_TO_2
cd $CODE_DIR
mpirun --allow-run-as-root -np 3 -npernode 1 bash -c \
        'python train.py --data $DATA_DIR/ffhq/ffhq-lmdb --root $CHECKPOINT_DIR --save $EXPR_ID --dataset ffhq \
        --num_channels_enc 30 --num_channels_dec 30 --epochs 200 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_latent_scales 5 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
        --num_preprocess_blocks 1 --num_postprocess_blocks 1 --weight_decay_norm 1e-1  --num_groups_per_scale 16 \
        --batch_size 4 --num_nf 2  --ada_groups --min_groups_per_scale 4 \
        --weight_decay_norm_anneal --weight_decay_norm_init 1. --num_process_per_node 8 --use_se --res_dist \
        --fast_adamax --num_x_bits 5 --learning_rate 8e-3 --node_rank $NODE_RANK --num_proc_node 3 --master_address $IP_ADDR '
```

In our early experiments, a smaller model with 24 channels instead of 30, could be trained on only 8 GPUs in 
the same time (with the batch size of 6). The smaller models obtain only 0.01 bpd higher 
negative log-likelihood.
</details>

**If for any reason your training is stopped, use the exact same commend with the addition of `--cont_training`
to continue training from the last saved checkpoint. If you observe NaN, continuing the training using this flag
usually will not fix the NaN issue.**

## Known Issues
<details><summary>Cannot build CelebA 64 or training gives NaN right at the beginning on this dataset </summary>

Several users have reported issues building CelebA 64 or have encountered NaN at the beginning of training on this dataset.
If you face similar issues on this dataset, you can download this dataset manually and build LMDBs using instructions
on this issue https://github.com/NVlabs/NVAE/issues/2 .
</details>

<details><summary>Getting NaN after a few epochs of training </summary>

One of the main challenges in training very deep hierarchical VAEs is training instability that we discussed in the paper.
We have verified that the settings in the commands above can be trained in a stable way. If you modify the settings
above and you encounter NaN after a few epochs of training, you can use these tricks to stabilize your training:
i) increase the spectral regularization coefficient, `--weight_decay_norm`. ii) Use exponential decay on 
`--weight_decay_norm` using  `--weight_decay_norm_anneal` and `--weight_decay_norm_init`. iii) Decrease learning rate.
</details>

<details><summary>Training freezes with no NaN </summary>

In some very rare cases, we observed that training freezes after 2-3 days of training. We believe the root cause
of this is because of a racing condition that is happening in one of the low-level libraries. If for any reason the training 
is stopped, kill your current run, and use the exact same commend with the addition of `--cont_training`
to continue training from the last saved checkpoint.
</details>

## Monitoring the training progress
While running any of the commands above, you can monitor the training progress using Tensorboard:

<details><summary>Click here</summary>

```shell script
tensorboard --logdir $CHECKPOINT_DIR/eval-$EXPR_ID/
```
Above, `$CHECKPOINT_DIR` and `$EXPR_ID` are the same variables used for running the main training script.

</details> 

## Post-training sampling, evaluation, and checkpoints

<details><summary>Evaluating Log-Likelihood</summary>

You can use the following command to load a trained model and evaluate it on the test datasets:

```shell script
cd $CODE_DIR
python evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --data $DATA_DIR/mnist --eval_mode=evaluate --num_iw_samples=1000
```
Above, `--num_iw_samples` indicates the number of importance weighted samples used in evaluation. 
`$CHECKPOINT_DIR` and `$EXPR_ID` are the same variables used for running the main training script.
Set `--data` to the same argument that was used when training NVAE (our example is for MNIST).

</details> 

<details><summary>Sampling</summary>

You can also use the following command to generate samples from a trained model:

```shell script
cd $CODE_DIR
python evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --eval_mode=sample --temp=0.6 --readjust_bn
```
where `--temp` sets the temperature used for sampling and `--readjust_bn` enables readjustment of the BN statistics
as described in the paper. If you remove `--readjust_bn`, the sampling will proceed with BN layer in the eval mode 
(i.e., BN layers will use running mean and variances extracted during training).

</details>

<details><summary>Computing FID</summary>

You can compute the FID score using 50K samples. To do so, you will need to create
a mean and covariance statistics file on the training data using a command like:

```shell script
cd $CODE_DIR
python scripts/precompute_fid_statistics.py --data $DATA_DIR/cifar10 --dataset cifar10 --fid_dir /tmp/fid-stats/
```
The command above computes the references statistics on the CIFAR-10 dataset and stores them in the `--fid_dir` durectory.
Given the reference statistics file, we can run the following command to compute the FID score:

```shell script
cd $CODE_DIR
python evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --data $DATA_DIR/cifar10 --eval_mode=evaluate_fid  --fid_dir /tmp/fid-stats/ --temp=0.6 --readjust_bn
```
where `--temp` sets the temperature used for sampling and `--readjust_bn` enables readjustment of the BN statistics
as described in the paper. If you remove `--readjust_bn`, the sampling will proceed with BN layer in the eval mode 
(i.e., BN layers will use running mean and variances extracted during training).
Above, `$CHECKPOINT_DIR` and `$EXPR_ID` are the same variables used for running the main training script.
Set `--data` to the same argument that was used when training NVAE (our example is for MNIST).

</details> 

<details><summary>Checkpoints</summary> 

We provide checkpoints on MNIST, CIFAR-10, CelebA 64, CelebA HQ 256, FFHQ in 
[this Google drive directory](https://drive.google.com/drive/folders/1KVpw12AzdVjvbfEYM_6_3sxTy93wWkbe?usp=sharing). 
For CIFAR10, we provide two checkpoints as we observed that a multiscale NVAE provides better qualitative
results than a single scale model on this dataset. The multiscale model is only slightly worse in terms
of log-likelihood (0.01 bpd). We also observe that one of our early models on CelebA HQ 256 with 0.01 bpd 
worse likelihood generates much better images in low temperature on this dataset.

You can use the commands above to evaluate or sample from these checkpoints.

</details> 

## How to construct smaller NVAE models
In the commands above, we are constructing big NVAE models that require several days of training
in most cases. If you'd like to construct smaller NVAEs, you can use these tricks:

* Reduce the network width: `--num_channels_enc` and `--num_channels_dec` are controlling the number
of initial channels in the bottom-up and top-down networks respectively. Recall that we halve the
number of channels with every spatial downsampling layer in the bottom-up network, and we double the number of
channels with every upsampling layer in the top-down network. By reducing
`--num_channels_enc` and `--num_channels_dec`, you can reduce the overall width of the networks.

* Reduce the number of residual cells in the hierarchy: `--num_cell_per_cond_enc` and 
`--num_cell_per_cond_dec` control the number of residual cells used between every latent variable
group in the bottom-up and top-down networks respectively. In most of our experiments, we are using
two cells per group for both networks. You can reduce the number of residual cells to one to make the model
smaller.

* Reduce the number of epochs: You can reduce the training time by reducing `--epochs`.

* Reduce the number of groups: You can make NVAE smaller by using a smaller number of latent variable groups. 
We use two schemes for setting the number of groups:
    1. An equal number of groups: This is set by `--num_groups_per_scale` which indicates the number of groups 
    in each scale of latent variables. Reduce this number to have a small NVAE.
    
    2. An adaptive number of groups: This is enabled by `--ada_groups`. In this case, the highest
    resolution of latent variables will have `--num_groups_per_scale` groups and 
    the smaller scales will get half the number of groups successively (see groups_per_scale in utils.py).
    We don't let the number of groups go below `--min_groups_per_scale`. You can reduce
    the total number of groups by reducing `--num_groups_per_scale` and `--min_groups_per_scale`
    when `--ada_groups` is enabled.

## Understanding the implementation
If you are modifying the code, you can use the following figure to map the code to the paper.

<p align="center">
    <img src="img/model_diagram.png" width="900">
</p>


## Traversing the latent space
We can generate images by traversing in the latent space of NVAE. This sequence is generated using our model
trained on CelebA HQ, by interpolating between samples generated with temperature 0.6. 
Some artifacts are due to color quantization in GIFs.

<p align="center">
    <img src="https://drive.google.com/uc?id=1k_s_TCdblNRI6MG_X1tji9VoOPumCzz9" width="512">
</p>

## License
Please check the LICENSE file. NVAE may be used non-commercially, meaning for research or 
evaluation purposes only. For business inquiries, please contact 
[researchinquiries@nvidia.com](mailto:researchinquiries@nvidia.com).

You should take into consideration that VAEs are trained to mimic the training data distribution, and, any 
bias introduced in data collection will make VAEs generate samples with a similar bias. Additional bias could be 
introduced during model design, training, or when VAEs are sampled using small temperatures. Bias correction in 
generative learning is an active area of research, and we recommend interested readers to check this area before 
building applications using NVAE.

## Bibtex:
Please cite our paper, if you happen to use this codebase:

```
@inproceedings{vahdat2020NVAE,
  title={{NVAE}: A Deep Hierarchical Variational Autoencoder},
  author={Vahdat, Arash and Kautz, Jan},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```
