import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
import os


def load_as_float(path):
    return imread(path).astype(np.float32)


class TestSet(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, train=True, sequence_length=3, transform=None, skip_frames=1, dataset='kitti'):
        np.random.seed(0)
        random.seed(0)
        self.root = Path(root)/'image_2'
        self.cam_root = Path(root)

        self.transform = transform
        self.dataset = dataset
        self.k = skip_frames
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.k,
                      demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)

        # intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
        with open(os.path.join(self.cam_root,'calib.txt'),'r') as f:
            iPx = f.readlines()
        # homography matrix 
        # iP0 = np.array([float(i) for i in iPx[0].strip('\n').split(' ')[1:]]).reshape(3,4)
        iP2 = np.array([float(i) for i in iPx[2].strip('\n').split(' ')[1:]]).reshape(3,4)
        intrinsics = iP2[:3,:3]


        imgs = sorted(self.root.files('*.png'))

        if len(imgs) < sequence_length:
            print('=> the length of seq is too short!')
        for i in range(demi_length * self.k, len(imgs)-demi_length * self.k):
            sample = {'intrinsics': intrinsics,'tgt': imgs[i], 'ref_imgs': []}
            for j in shifts:
                sample['ref_imgs'].append(imgs[i+j])
            sequence_set.append(sample)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])  # (370, 1226, 3)
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]

        # if self.transform is not None:    
        #     tgt_img, _ = self.transform([tgt_img], None)
        #     tgt_img = tgt_img[0]  # torch.Size([3, 256, 832])

        #     ref_imgs, _ = self.transform([ref_imgs], None)
        #     ref_imgs = ref_imgs[0]

        if self.transform is not None:
            imgs, intrinsics = self.transform(
                [tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])

        return tgt_img, ref_imgs, intrinsics

    def __len__(self):
        return len(self.samples)
