#使用np的一些杂的功能
import numpy as np
import cv2
import tensorflow as tf


def coor_translate_np(coor, old_size, new_size):
    '''
    desc:使用np来进行的坐标点转换
    args:
        coor:原坐标
        old_size:[2],原图的大小
        new_center:[2],新图的大小
    '''
    new_size = new_size.astype(np.float)
    old_size = old_size.astype(np.float)
    coor = coor.astype(np.float)
    old_center = old_size / 2
    new_center = new_size / 2
    #相对于原图中心的坐标
    coor_relative = coor - old_center
    scale = new_size / old_size
    new_coor_relative = (coor_relative + 0.5) * scale - 0.5
    new_coor = new_coor_relative + new_center
    #保证不超出边界
    new_coor = np.minimum(
        np.maximum(new_coor, [0.0, 0.0]), [new_size[0] - 1, new_size[1] - 1])

    return np.round(new_coor).astype(np.int)


def rotate_and_scale(image, angle, center=None, scale=1.0):
    '''
    desc:对图像进行旋转和scale操作，返回对应的转换矩阵和转换后的图像
    '''
    (h, w) = image.shape[:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
    #得到仿射变换矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale)
    #进行仿射变换
    rotscal= cv2.warpAffine(image, M, (w, h))

    return M,rotscal


def make_gt_heatmaps(kpt_coor,map_size,sigma,mask):
    '''
    desc:生成真实的heatmaps
    args:
        kpt_coor:真实关键点坐标 先x后y
        map_size:图片大小
        sigma:方差
        mask:表示对应关键点是否被遮挡
    '''
    kpt_num=kpt_coor.shape[0]
    kpt_coor=kpt_coor.astype(np.int)
    x=np.linspace(0,map_size[1]-1,map_size[1])
    y=np.linspace(0,map_size[0]-1,map_size[0])
    y_t,x_t=np.meshgrid(y,x)
    y_t=np.transpose(y_t,(1,0))
    x_t=np.transpose(x_t,(1,0))
    #[h,w,kpt_num]
    y_t=np.tile(np.expand_dims(y_t,2),(1,1,kpt_num))
    #[h,w,kpt_num]
    x_t=np.tile(np.expand_dims(x_t,2),(1,1,kpt_num))
    x_rel=x_t-kpt_coor[:,0]
    y_rel=y_t-kpt_coor[:,1]
    #(x-u)**2+(y-v)**2
    #[h,w,kpt_num]
    distance=np.square(x_rel)+np.square(y_rel)
    #[h,w,kpt_num]
    #* 除去被遮挡或不在图片范围内的点
    gt_heatmaps=np.exp(-distance/sigma**2)*mask
    #3个像素点以外的值均为0
    gt_heatmaps=np.where(gt_heatmaps>np.exp(-18),gt_heatmaps,0)

    return gt_heatmaps


def variable_summaries(var):
    '''
    desc:可视化。
    args:
        var:tensor.
    '''
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)