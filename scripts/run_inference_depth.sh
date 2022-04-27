# absolute path that contains all datasets
DATA_ROOT=/root/yangbinchao/program/rlvo/sc_depth_pl

# kitti
# INPUT=$DATA_ROOT/kitti/testing/color
# OUTPUT=results/kitti
# CKPT=ckpts/kitti_scv1/version_0/epoch=99-val_loss=0.1411.ckpt
# CONFIG=configs/v1/kitti_raw.txt

# nyu
INPUT=$DATA_ROOT/demo/input
OUTPUT=demo/output
CONFIG=configs/v2/nyu.txt
CKPT=demo/nyu_scv2/version_10/epoch=101-val_loss=0.1580.ckpt

# inference depth
python inference_depth.py --config $CONFIG \
--input_dir $INPUT --output_dir $OUTPUT \
--ckpt_path $CKPT #--save-vis --save-depth
