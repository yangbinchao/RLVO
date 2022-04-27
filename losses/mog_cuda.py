import cv2
import numpy as np
import keyboard

def MOG_mask(source_img, target_img):
    #gpu_frame = cv2.cuda_GpuMat()
    mog = cv2.createBackgroundSubtractorMOG2(2, 25, detectShadows=False)
    # 参数：
    # history：用于训练背景的帧数，默认为500帧,
    # varThreshold：方差阈值，用于判断当前像素是前景还是背景
    # detectShadows：是否检测影子，设为true为检测，false为不检测

    frame = source_img
    for i in range(2):
        mask_img = mog.apply(frame)
        cv2.imshow("mog", mask_img)
        # th = cv2.threshold(np.copy(mask_img), 244, 255, cv2.THRESH_BINARY)[1]     # cv2.threshold二值化函数
        # th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)      # 腐蚀操作
        # dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)    # 膨胀操作 
        # cv2.imshow("thresh", th)
        # cv2.imshow("diff", frame & cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)) #  融合原图可视化
        # cv2.imshow("detection", frame)
        frame = target_img
    cv2.waitKey(10000)   
    return mask_img

# for i in range(1589):            #测试连续数据集，对应range()为数据集大小-1
#     source_img = cv2.imread("E://image_2/%06d.png"%(i))
#     target_img = cv2.imread("E://image_2/%06d.png"%(i+1))
#     mask = GMG_mask(source_img, target_img)
#     if keyboard.is_pressed('Esc'):        # 按键退出
#         break

#     # print("E://image_2/%06d.png"%(i))
#     # cv2.imshow("haha",source_img)


'''
测试两张图像
'''
def main():
    img1_path = "../demo/mog_test/1.jpg"
    img2_path = "../demo/mog_test/2.jpg"
    source_img = cv2.imread(img1_path)
    target_img = cv2.imread(img2_path)
    mask = MOG_mask(source_img, target_img)


if __name__ == '__main__':
    main()