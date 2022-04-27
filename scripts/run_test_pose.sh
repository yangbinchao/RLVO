# absolute path that contains all datasets
DATA_ROOT=/data/yangbinchao_data/KITTI-odometry/dataset/sequences

# kitti
SEQ_NUM=09
DATASET=$DATA_ROOT/$SEQ_NUM
CONFIG=configs/v1/kitti_raw.txt
OUTPUT=demo/pose
CKPT=ckpts/kitti_scv1/version_12/epoch=64-val_loss=0.1384.ckpt  #last.ckpt #epoch=87-val_loss=0.1457.ckpt

# run
python test_pose.py --config $CONFIG --dataset_dir $DATASET --ckpt_path $CKPT --output_dir_pose $OUTPUT --pose_sequences $SEQ_NUM

