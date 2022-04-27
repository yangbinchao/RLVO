import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from path import Path

# pytorch-lightning
from config import get_opts

from SC_Depth import SC_Depth
from SC_DepthV2 import SC_DepthV2

from datasets.test_folder_pose import TestSet
import datasets.custom_transforms as custom_transforms

from losses.loss_functions import compute_errors
from losses.inverse_warp import pose_vec2mat

from visualization import *


@torch.no_grad()
def main():
    hparams = get_opts()

    # initialize network
    if hparams.model_version == 'v1':
        system = SC_Depth(hparams)
    elif hparams.model_version == 'v2':
        system = SC_DepthV2(hparams)
    model = system.load_from_checkpoint(hparams.ckpt_path)
    model.cuda()
    model.eval()

    # dataset
    if hparams.dataset_name == 'nyu':
        training_size = [256, 320]
    elif hparams.dataset_name == 'kitti':
        training_size = [256, 832] # [128, 416] # [160, 512] # [256, 832]
    elif hparams.dataset_name == 'ddad':
        training_size = [384, 640]

    '''
    data loader
    '''
    test_transform = custom_transforms.Compose([
        custom_transforms.RescaleTo(training_size),
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize()]
    )
    test_dataset = TestSet(
        hparams.dataset_dir,
        transform=test_transform,
        dataset=hparams.dataset_name
    )
    print('{} samples found in test scenes'.format(len(test_dataset)))

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True
                             )

    all_errs = []
    predictions_array = np.zeros((len(test_loader), 3, 3, 4))
    for i, (tgt_img, ref_imgs, intrinsics) in enumerate(tqdm(test_loader)):
        poses = [model.inference_pose(tgt_img.cuda(), im.cuda()) for im in ref_imgs]
        pose_1 = poses[0].cpu()
        pose_2 = poses[1].cpu()
        poses = torch.cat([pose_1, torch.zeros(1,6).float(), pose_2])
        inv_transform_matrices = pose_vec2mat(poses).numpy().astype(np.float64) # [B,6] -> [B,3,4]

        rot_matrices = np.linalg.inv(inv_transform_matrices[:,:,:3])
        tr_vectors = -rot_matrices @ inv_transform_matrices[:,:,-1:]
        transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)
        first_inv_transform = inv_transform_matrices[0]
        final_poses = first_inv_transform[:,:3] @ transform_matrices
        final_poses[:,:,-1:] += first_inv_transform[:,-1:]  # final_poses done

        predictions_array[i] = final_poses
        np.save(str(hparams.output_dir_pose) + '/predictions.npy', predictions_array)
        data=np.load(str(hparams.output_dir_pose) + '/predictions.npy') 

        for i in range(1,len(data)):
            r = data[i-1,1]
            data[i] = r[:,:3] @ data[i]
            data[i,:,:,-1] = data[i,:,:,-1] + r[:,-1]

        data = data.reshape(-1,12)
        pose_prediction_path = str(hparams.output_dir_pose) + '/kitti_odometry_test/' + str(hparams.pose_sequences) + '.txt'
        np.savetxt(pose_prediction_path,data,delimiter=' ') 

    print('=> the result save on {}'.format(pose_prediction_path))


if __name__ == '__main__':
    main()
