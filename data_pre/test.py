import cv2
import numpy as np
from utils.utils import *
from multiprocessing import freeze_support
from data_loader.coco_preprocess import COCOPreProcess
from data_loader.mpii_preprocess import MPIIPreProcess

if __name__ == '__main__':
    freeze_support()
    data_pre = COCOPreProcess(
        'D:/迅雷下载/coco/val2017/',
        'D:/迅雷下载/coco/annotations_trainval2017/annotations/person_keypoints_val2017.json',
        resized_size=(1024, 1024),
        is_multi=True)
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
