# absolute path that contains all datasets
DATA_ROOT=/root/yangbinchao/program

# kitti
DATASET=$DATA_ROOT/data/KITTI-sc
CONFIG=configs/v1/kitti_raw.txt

# nyu
# DATASET=$DATA_ROOT/nyu
# CONFIG=configs/v2/nyu.txt

python train.py --config $CONFIG --dataset_dir $DATASET 