#!/bin/sh

python cnnmatching.py

source ~/anaconda3/etc/profile.d/conda.sh
conda activate bim_view

cd /home/houhao/workspace/bim_view_o3d/scripts

python rigid_transform_svd.py