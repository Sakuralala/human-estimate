#测试数据的预读取
#encoding=utf-8
from data_gen import *
import numpy as np
import cv2
from multiprocessing import freeze_support
import multiprocessing as mp


def main(ann_dir, img_dir):
    batch_gen = batch(ann_dir, img_dir, batch_size=1, slavers=2)
    while True:
        ret = next(batch_gen)
        ann_img = ret[2][0].draw_on_image(ret[0][0], size=7)
        cv2.imshow('test', ann_img)
        cv2.waitKey()


if __name__ == '__main__':
    freeze_support()
    main(
        'D:/迅雷下载/coco/annotations_trainval2017/annotations/person_keypoints_val2017.json',
        'D:/迅雷下载/coco/val2017/')
