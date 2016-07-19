This is my own development branch. It merges the mpi-parallel and the batch-normalization branches, and adds some other things such as the python script (tools/show_log.py) to plot the training curves.

Below are the README merged from the mpi-parallel and batch-normalization branches.


# MPI Parallel

This branch provides data parallelization for Caffe based on MPI.

## Installation

- Install `openmpi` with `apt-get`, or `pacman`, or `yum`, etc.
- Uncomment the MPI parallel block in the Makefile.config and set the `MPI_INCLUDE` and `MPI_LIB` correspondingly.
- `make clean && make -j`

## Usage

You don't need to change your prototxt. Just provide the GPU ids in the `-gpu` option (separated by commas). For example:

    mpirun -n 2 build/tools/caffe train \
      -solver examples/mnist/lenet_solver.prototxt \
      -gpu 0,1


# Batch Normalization

This branch provides implementation of Batch Normalization (BN). Most of the codes are adpated from Chenglong Chen's [caffe-windows](https://github.com/ChenglongChen/caffe-windows).

## Usage

Just add a BN layer before each activation function. The configuration of a BN layer looks like:

    layer {
      name: "conv1_bn"
      type: "BN"
      bottom: "conv1"
      top: "conv1_bn"
      param {
        lr_mult: 1
        decay_mult: 0
      }
      param {
        lr_mult: 1
        decay_mult: 0
      }
      bn_param {
        slope_filler {
          type: "constant"
          value: 1
        }
        bias_filler {
          type: "constant"
          value: 0
        }
      }
    }

We also implement a simple version of local data shuffling in the data layer. It's recommended to set `shuffle_pool_size: 10` in `data_param` of the training data layer.
