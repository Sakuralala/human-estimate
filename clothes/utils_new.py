#一些杂的辅助函数
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


def augment(image):
    '''
    desc:对原始图片进行一些hue和brightness的增强
    args:
        image:输入的图片
    '''
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    size = np.random.randint(1, 5)
    if size % 2:
        random_bright = .2 + np.random.uniform(0.6,1.0)
        random_hue = .2 + np.random.uniform(0.6,1.0)
        image1[:, :, 2] = image1[:, :, 2] * random_bright
        image1[:, :, 1] = image1[:, :, 1] * random_hue
        image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
        image1 = cv2.GaussianBlur(image1, (size, size), 0)
    return image1


def rotate_and_scale(image, angle, center=None, scale=1.0):
    '''
    desc:对图像进行旋转和scale操作，返回对应的转换矩阵和转换后的图像
    args:
        image:输入的图片
        angle:旋转的角度
        center:旋转中心
        scale:放缩尺寸
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
    desc:用以生成坐标maps的辅助函数,用以取得每个图片的每个像素点的位置。
    args:
        image_size:图片大小
    '''
    x = np.linspace(0, image_size[1] - 1, image_size[1])
    y = np.linspace(0, image_size[0] - 1, image_size[0])
    y_t, x_t = np.meshgrid(y, x)
    y_t = np.transpose(y_t, (1, 0))
    x_t = np.transpose(x_t, (1, 0))

    return x_t, y_t


def _get_rel(kpt_coor, image_size):
    '''
    desc:用以生成坐标maps的辅助函数,用以取得每个图片的每个像素点的位置相对关键点位置的偏移。
    args:
        image_size:图片大小
        kpt_coor:关键点坐标 先x后y
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
    desc:生成真实的heatmaps
    args:
        kpt_coor:真实关键点坐标 先x后y
        image_size:图片大小
        sigma:方差
        mask:表示对应关键点是否被遮挡
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
    desc:产生以关键点坐标为圆心，半径为r范围内值为1其他点的值为0的probmaps,以及对应的offset vector。
    args:
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
    shape=offset.shape
    #[h,w,kpt_num*2]
    offset=np.reshape(offset,(shape[0],shape[1],shape[2]*shape[3]))
    #完整的真实标签
    #[h,w,kpt_num*3]
    total = np.concatenate((probmaps, offset), -1)

    return total


def get_locmap(predicted):
    '''
    desc:通过预测的概率heatmaps及offset vectors,并通过hough voting的方式来获取最终的关键点定位图。
    args:
        predicted:概率heatmaps和offset的合体。
    '''
    shape=predicted.shape
    #[b,h,w,kpt_num]
    probmaps = predicted[:, :, :, 0:shape[-1]//3]
    #[b,h,w,kpt_num*2]
    offsets = predicted[:, :, :,shape[-1]//3:]
    size = (shape[1], shape[2])
    x_t, y_t = _get_grid(size)
    #[h,w,2]
    pixel_coor = np.stack([y_t, x_t], -1)
    #[h,w,kpt_num*2]
    pixel_coor=np.tile(pixel_coor,(1,1,shape[-1]//3))
    #[b,h,w,kpt_num*2]            [h,w,kpt_num*2]
    predicted_kpt_coor = offsets + pixel_coor
    #要返回的图 [b,h,w,kpt_num]
    locmaps = np.zeros(
        (shape[0], shape[1], shape[2],
         shape[3]//3),
        dtype=np.float)
    predicted_kpt_coor=np.expand_dims(np.expand_dims(predicted_kpt_coor,0),0)
    #[h,w,b,h,w,n]
    #memory error.........
    predicted_kpt_coor=np.tile(predicted_kpt_coor,(shape[1],shape[2],1,1,1,1))
    #[h,w,1,1,1,n]
    pixel_coor=np.expand_dims(np.expand_dims(np.expand_dims(pixel_coor,2),2),2)
    #xj+F(xj)-x [h,w,b,h,w,n]
    tmp=predicted_kpt_coor-pixel_coor
    #B(xj+F(xj)-x) [h,w,b,h,w,n/2]
    tmp=1.0-np.sqrt(np.square(tmp[:,:,:,:,:,0::2])+np.square(tmp[:,:,:,:,:,1::2]))
    tmp=np.where(tmp>=0,tmp,0)
    #h(xj)*
    tmp*=probmaps
    #[b,h,w,n/2]
    locmaps=np.sum(tmp,(0,1))
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