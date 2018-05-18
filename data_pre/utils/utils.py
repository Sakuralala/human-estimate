#一些杂的辅助函数
#encoding=utf-8
import numpy as np
import cv2
import tensorflow as tf
#用于data augment
import imgaug as ia
import os
from imgaug import augmenters as iaa
from utils.parallel_gen import GeneratorEnqueuer


#batch生成器 ...
def batch_generator(data_preprocess, slavers=10, max_queue_size=16):
    '''
    description：生成batch
    parameters:
        data_preprocess: object
            数据集预处理类，必须有get_batch_gen方法(linux下使用)和next方法(windows下使用)
        slavers: int
            线程数量
        max_epoch: int
            最大迭代轮数
    return: generator
        返回从队列中取得预处理batch的generator
    '''
    try:
        if os.name is 'posix':
            gen = data_preprocess.get_batch_gen()
        #for fuck windows
        elif os.name is 'nt':
            gen = data_preprocess
        else:
            raise ValueError('Unknown system.')
        gen_enq = GeneratorEnqueuer(gen)
        #产生了多个slaver并开始工作
        gen_enq.start(slaver=slavers, max_queue_size=max_queue_size)
        while True:
            #is_running/empty在get中已经判断了
            batch_output = gen_enq.get()
            #返回的也是一个生成器 所以需要next
            batch_output = next(batch_output)
            #能用生成器 表明父进程的这个类实例未被传给子进程？
            yield batch_output
            batch_output = None
    except Exception as e:
        raise e
        #if gen_enq is not None:
        #gen_enq.stop()


def isArrayLike(obj):
    '''
    description:判断一个obj是否为类array np的也算
    return: bool
        true if obj is array like else false
    '''
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


def augment_both(img, coords, prob=0.5):
    '''
    description:在对img进行augment的同时相应地变换关键点的坐标,注意关键点需要先x后y.
    parameters:
        img: ndarray
            图片[h,w,3]或[b,h,w,3]
        coords: ndarray or list
            关键点坐标,[n,2]的array或[b,n,2] 或[b,x,2] x表示改维度的长度不一定一致
        prob: float
            做某一项augment的概率
    return: tuple of list
        增强后的imgs和转换后的coords的tuple
    '''
    #ia.seed(1)
    sometimes = lambda aug: iaa.Sometimes(prob, aug)
    #print(coords.shape, img.shape)
    if len(img.shape) == 3:
        img = [img]
    coords = [
        ia.KeypointsOnImage.from_coords_array(coords[i], img[i].shape)
        for i in range(coords.shape[0])
    ]
    #做brightness、旋转和放缩，概率为prob
    seq = iaa.Sequential([
        sometimes(iaa.Multiply((0.8, 1.2))),
        sometimes(
            iaa.OneOf([
                iaa.GaussianBlur((0, 3.0)),
                iaa.AverageBlur(k=(2, 7)),
                iaa.MedianBlur(k=(3, 11))
            ])),
        sometimes(iaa.Affine(scale=(0.75, 1.25), rotate=(-30, 30)))
    ])
    #保证每次aug不一样
    seq_det = seq.to_deterministic()
    img_aug = seq_det.augment_images(img)
    coords_aug = seq_det.augment_keypoints(coords)
    coords = [elem.get_coords_array() for elem in coords_aug]

    return img_aug, coords


def scale_coords_trans(coor, old_size, new_size):
    '''
    description:缩放变换的坐标点转换,注意coor、old_size、new_size的xy的先后顺序要一致
    parameters:
        coor: ndarray [k,2]/[b,k,2]
            原坐标
        old_size: ndarray 
            [2]/[b,2],原图的大小
        new_center: ndarray
            [2]/[b,2],新图的大小
    return: ndarray 
        返回处理好后的坐标
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
    description:对图像进行旋转和scale操作，返回对应的转换矩阵和转换后的图像
    parameters:
        image: ndarray
            输入的图片
        angle: int 
            旋转的角度
        center: tuple
            旋转中心 先x后y
        scale: float or int
            放缩尺寸
    return: tuple
        返回对应的仿射变换矩阵和变换后的图片
    '''
    (h, w) = image.shape[:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
    #得到仿射变换矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale)
    #进行仿射变换
    rotscal = cv2.warpAffine(image, M, (w, h))

    return M, rotscal


def _get_grid(image_size):
    '''
    description:用以生成坐标maps的辅助函数,用以取得每个图片的每个像素点的位置。
    parameters:
        image_size: tuple or array like
            图片大小
    return: tuple
        返回每个像素点的x坐标与y坐标ndarray
    '''
    x = np.linspace(0, image_size[1] - 1, image_size[1])
    y = np.linspace(0, image_size[0] - 1, image_size[0])
    y_t, x_t = np.meshgrid(y, x)
    y_t = np.transpose(y_t, (1, 0))
    x_t = np.transpose(x_t, (1, 0))

    return x_t, y_t


def _get_rel(kpt_coor, image_size):
    '''
    description:用以生成坐标maps的辅助函数,用以取得每个图片的每个像素点的位置相对关键点位置的偏移。
    parameters:
        image_size: array like
            图片大小
        kpt_coor: ndarray
            关键点坐标 先x后y
    return: tuple of ndarray
        返回图片中每个像素点坐标相对关节点位置的偏移
    '''
    kpt_num = kpt_coor.shape[0]
    kpt_coor = kpt_coor.astype(np.int)
    x_t, y_t = _get_grid(image_size)
    #[h,w,kpt_num]
    y_t = np.tile(np.expand_dims(y_t, 2), (1, 1, kpt_num))
    #[h,w,kpt_num]
    x_t = np.tile(np.expand_dims(x_t, 2), (1, 1, kpt_num))
    x_rel = x_t - kpt_coor[:, 0]
    y_rel = y_t - kpt_coor[:, 1]

    return x_rel, y_rel


def make_gt_heatmaps(kpt_coor, image_size, sigma, mask):
    '''
    description:生成真实的heatmaps
    parameters:
        kpt_coor: ndarray
            真实关键点坐标 先x后y
        image_size: array like
            图片大小
        sigma: int or float
            方差
        mask: ndarray
            表示对应关键点是否被遮挡
    return: ndarray
        返回生成的gt heatmaps
    '''
    x_rel, y_rel = _get_rel(kpt_coor, image_size)
    #(x-u)**2+(y-v)**2
    #[h,w,kpt_num]
    distance = np.square(x_rel) + np.square(y_rel)
    #[h,w,kpt_num]
    #* 除去被遮挡或不在图片范围内的点
    gt_heatmaps = np.exp(-distance / sigma**2) * mask
    #3个像素点以外的值均为0
    gt_heatmaps = np.where(gt_heatmaps > np.exp(-18), gt_heatmaps, 0)

    return gt_heatmaps


def make_probmaps_and_offset(kpt_coor, image_size, radius, mask):
    '''
    description:产生以关键点坐标为圆心，半径为r范围内值为1其他点的值为0的probmaps,以及对应的offset vector。
    parameters:
        kpt_coor:关键点坐标，先x后y。
        image_size:图片大小。
        radius:半径。
        mask:遮挡及不存在的点的掩码。
    '''
    x_rel, y_rel = _get_rel(kpt_coor, image_size)
    distance = np.square(x_rel) + np.square(y_rel)
    #[h,w,kpt_num]  不存在/被遮挡的关键点对应的图置为全0
    probmaps = np.where(distance <= radius**2, 1, 0) * mask

    #在半径外的像素点的x和y的相对坐标全置为零
    #取-是表示从各个像素点位置指向关键点的向量
    x_rel, y_rel = -x_rel, -y_rel
    #[h,w,kpt_num,2]
    offset = np.stack([y_rel, x_rel], -1)
    shape = offset.shape
    #[h,w,kpt_num*2]
    offset = np.reshape(offset, (shape[0], shape[1], shape[2] * shape[3]))
    #完整的真实标签
    #[h,w,kpt_num*3]
    total = np.concatenate((probmaps, offset), -1)

    return total


def get_locmap(predicted):
    '''
    description:通过预测的概率heatmaps及offset vectors,并通过hough voting的方式来获取最终的关键点定位图。
    parameters:
        predicted:概率heatmaps和offset的合体。
    '''
    shape = predicted.shape
    #[b,h,w,kpt_num]
    probmaps = predicted[:, :, :, 0:shape[-1] // 3]
    #[b,h,w,kpt_num*2]
    offsets = predicted[:, :, :, shape[-1] // 3:]
    size = (shape[1], shape[2])
    x_t, y_t = _get_grid(size)
    #[h,w,2]
    pixel_coor = np.stack([y_t, x_t], -1)
    #[h,w,kpt_num*2]
    pixel_coor = np.tile(pixel_coor, (1, 1, shape[-1] // 3))
    #[b,h,w,kpt_num*2]            [h,w,kpt_num*2]
    predicted_kpt_coor = offsets + pixel_coor
    #要返回的图 [b,h,w,kpt_num]
    locmaps = np.zeros(
        (shape[0], shape[1], shape[2], shape[3] // 3), dtype=np.float)
    predicted_kpt_coor = np.expand_dims(
        np.expand_dims(predicted_kpt_coor, 0), 0)
    #[h,w,b,h,w,n]
    #memory error.........
    predicted_kpt_coor = np.tile(predicted_kpt_coor,
                                 (shape[1], shape[2], 1, 1, 1, 1))
    #[h,w,1,1,1,n]
    pixel_coor = np.expand_dims(
        np.expand_dims(np.expand_dims(pixel_coor, 2), 2), 2)
    #xj+F(xj)-x [h,w,b,h,w,n]
    tmp = predicted_kpt_coor - pixel_coor
    #B(xj+F(xj)-x) [h,w,b,h,w,n/2]
    tmp = 1.0 - np.sqrt(
        np.square(tmp[:, :, :, :, :, 0::2]) +
        np.square(tmp[:, :, :, :, :, 1::2]))
    tmp = np.where(tmp >= 0, tmp, 0)
    #h(xj)*
    tmp *= probmaps
    #[b,h,w,n/2]
    locmaps = np.sum(tmp, (0, 1))
    #暂时就先用循环把
    #循环太慢了。。
    '''
    for i in range(predicted.shape[1]):
        for j in range(predicted.shape[2]):
            #xj+F(xj)-x [b,h,w,kpt_num*2]
            tmp = predicted_kpt_coor - pixel_coor[i, j]
            #B(xj+F(xj)-x)  三角核函数
            #[b,h,w,kpt_num]
            tmp = 1.0 - np.sqrt(
                np.square(tmp[:, :, :, 0::2]) + np.square(tmp[:, :, :, 1::2]))
            #超出范围的置0
            tmp = np.where(tmp >= 0, tmp, 0)
            #h(xj)*
            tmp *= probmaps
            #[b,kpt_num] 所有像素点对(i,j)点的投票最终得分
            tmp = np.sum(tmp, (1, 2))
            locmaps[:, i, j, :] += tmp
            tmp = None
    '''
    return locmaps


#用这个的话产生的event文件就太大了。。。。。
def variable_summaries(var):
    '''
    description:可视化。
    parameters:
        var:tensor.
    return:
        none
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
