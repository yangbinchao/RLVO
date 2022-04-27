import torchvision.transforms as T
import torch
import numpy as np
import cv2
from PIL import Image


def visualize_image(image):
    """
    tensor image: (3, H, W)
    """
    x = (image.cpu() * 0.225 + 0.45)
    return x

def tensor2numpy(input_tensor):
	input_tensor = input_tensor[0].detach().cpu().numpy()# input_tensor[0].to(torch.device('cpu')).numpy()
	in_arr = np.transpose(input_tensor,(1,2,0)) #将(c,w,h)转换为(w,h,c)。但此时如果是全精度模型，转化出来的dtype=float64 范围是[0,1]。后续要转换为cv2对象，需要乘以255
	return np.uint8(in_arr*255) # , cv2.COLOR_BGR2BGR

def numpy2tensor(images):
    tensors = []
    for im in images:
        # put it from HWC to CHW format
        im = np.transpose(im, (2, 0, 1))
        # handle numpy array
        tensors.append(torch.from_numpy(im).float()/255)
    return tensors

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8)  # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_
