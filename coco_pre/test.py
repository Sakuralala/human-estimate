import cv2
import numpy as np
from utils_new import *
from multiprocessing import freeze_support
from data_gen import *
from coco_preprocess import COCOPreProcess

if __name__ == '__main__':
    freeze_support()
    coco_pre = COCOPreProcess(
        'D:/迅雷下载/coco/annotations_trainval2017/annotations/person_keypoints_val2017.json',
        'D:/迅雷下载/coco/val2017/')
    gen = batch_generator(coco_pre)
    while True:
        batch = next(gen)
        img_kpt = batch[2][0].draw_on_image(batch[0][0], size=5)
        cv2.imshow('sample',img_kpt)
        cv2.waitKey()
