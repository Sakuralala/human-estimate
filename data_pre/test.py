import cv2
import numpy as np
from utils.utils import *
from multiprocessing import freeze_support
from data_loader.coco_preprocess import COCOPreProcess
from data_loader.mpii_preprocess import MPIIPreProcess

if __name__ == '__main__':
    freeze_support()
    data_pre = COCOPreProcess(
        'D:/迅雷下载/mpii_human_pose_v1/images/',
        'mpii_train.json',
        resized_size=(512, 768),
        batch_size=1)
    '''
    mpii_pre = MPIIPreProcess('mpii_train.h5',
                              'D:/迅雷下载/mpii_human_pose_v1/images/',batch_size=1)
    '''
    gen = batch_generator(data_pre)
    while True:
        batch = next(gen)
        img_kpt = batch[2][0].draw_on_image(batch[0][0], size=5)
        cv2.imshow('sample', img_kpt)
        cv2.waitKey()
