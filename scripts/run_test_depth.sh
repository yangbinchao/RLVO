# absolute path that contains all datasets
DATA_ROOT=/root/yangbinchao/program

# kitti
DATASET=$DATA_ROOT/data/KITTI-sc
CONFIG=configs/v1/kitti_raw.txt
CKPT=ckpts/kitti_scv1/version_12/epoch=64-val_loss=0.1384.ckpt  #last.ckpt #epoch=87-val_loss=0.1457.ckpt

# nyu
# DATASET=$DATA_ROOT/data/NYU-sc
# CONFIG=configs/v2/nyu.txt
# CKPT=demo/nyu_scv2/version_10/epoch=101-val_loss=0.1580.ckpt

# run
python test_depth.py --config $CONFIG --dataset_dir $DATASET --ckpt_path $CKPT

