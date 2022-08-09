#!/usr/bin/env bash
PATH=~/miniconda3/bin:$PATH
export PYTHONPATH=$PWD:$PWD/taichi_three:$PWD/PointFlow:$PWD/setvae:$PYTHONPATH
export PATH=$PWD/taichi_three:$PATH:/opt/cuda/11.1.1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/11.1.1/lib64

. activate plb
