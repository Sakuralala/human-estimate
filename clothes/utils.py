#一些杂项
import tensorflow as tf
import tensorflow.contrib.distributions as tfd
from params_clothes import *

#使用tf的内建类来进行二维Gaussian Map的制作
'''
keypoint_coordinates:关节点的坐标，note:是在原图的尺寸下的坐标。[21,2]先height再width tf.int32
map_size:Gaussian Map的大小，[h,w]
sigma:标准差，因为是制作二维Gaussian Map,两个维度间的坐标值相互独立,故协方差为0，相同维度间的方差取相同。
'''


def make_gt_heatmap(keypoint_coordinates, map_size, sigma, valid_vec):
    with tf.name_scope('ground_truth_heatmap'):
        sigma = tf.cast(sigma, tf.float32)
        dims = keypoint_coordinates.get_shape().as_list()
        if valid_vec is not None:
            valid_vec = tf.cast(valid_vec, tf.float32)
            valid_vec = tf.squeeze(valid_vec)
            cond_val = tf.greater(valid_vec, 0.5)
        else:
            cond_val = tf.ones_like(
                keypoint_coordinates[:, 0], dtype=tf.float32)
            cond_val = tf.greater(cond_val, 0.5)
        # 在map_size height范围内的坐标
        cond_1_in = tf.logical_and(
            tf.less(keypoint_coordinates[:, 0], map_size[0] ),
            tf.greater(keypoint_coordinates[:, 0], -1))
        # 在map_size width范围内的坐标
        cond_2_in = tf.logical_and(
            tf.less(keypoint_coordinates[:, 1], map_size[1] ),
            tf.greater(keypoint_coordinates[:, 1], -1))
        cond_in = tf.logical_and(cond_1_in, cond_2_in)
        #即 可见又在crop_image范围内的坐标
        cond = tf.logical_and(cond_val, cond_in)

        #[2,2]
        covariance = tf.eye(2, name='covariance') * sigma
        #[k,2,2]
        covariance_all_jt = tf.tile(
            tf.expand_dims(covariance, 0), [dims[0], 1, 1],
            name='covariance_all_jt')
        #keypoint_coordinates即为中心位置，即均值点
        pdf_mvn = tfd.MultivariateNormalFullCovariance(
            tf.cast(keypoint_coordinates,tf.float32), covariance, name='2dim-normal-distribution')
        #以下几步生成了一个[256,256,2]的三维坐标枚举，即从[0,0]到[255,255]
        h = tf.range(map_size[0])
        w = tf.range(map_size[1])
        h = tf.tile(tf.expand_dims(h, -1), [1, map_size[1]])
        w = tf.reshape(tf.tile(w, [map_size[0]]), [map_size[0], map_size[1]])
        #[map_size[0],map_size[1],2]
        coor = tf.concat([tf.expand_dims(h, -1), tf.expand_dims(w, -1)], -1)
        #[map_size[0]*map_size[1],2]
        coor = tf.reshape(coor, [map_size[0] * map_size[1], 2])
        #[map_size[0]*map_size[1],all_jt,2]
        coor_all_jt = tf.tile(tf.expand_dims(coor, 1), [1, dims[0], 1])
        #gt_heamap:[map_size[0]*map_size[1],all_jt]
        gt_heatmap_all_jt = pdf_mvn.prob(tf.cast(coor_all_jt, tf.float32))
        #[map_size[0],map_size[1],all_jt]
        gt_heatmap_all_jt = tf.reshape(
            gt_heatmap_all_jt, [map_size[0], map_size[1], dims[0]],
            'ground_truth_heatmap_all_jt') * tf.cast(cond, tf.float32)
        #背景:[map_size[0],map_size[1],1]
        background = tf.expand_dims(tf.zeros([map_size[0], map_size[1]]), -1)
        #[map_size[0],map_size[1],NUM_OF_HEAMAPS]
        gt_heatmap_all_jt = tf.concat([gt_heatmap_all_jt, background], -1)
        gt_heatmap_all_jt = tf.image.resize_bilinear(
            tf.expand_dims(gt_heatmap_all_jt, 0),
            [map_size[0] // 4, map_size[1] // 4])
        gt_heatmap_all_jt=tf.squeeze(gt_heatmap_all_jt)
        return gt_heatmap_all_jt


def make_gaussian_map(keypoint_coordinates, map_size, sigma, valid_vec=None):
    '''
    key_point_coordinates:各个关节点的坐标 [21,2] 先y后x
    map_size:输出的heatmap大小
    sigma:正态分布的方差
    return:kp张gaussian_map
    '''
    with tf.name_scope('gaussian_map'):
        sigma = tf.cast(sigma, tf.float32)
        assert len(map_size) == 2, 'Gaussian map size must be 2.'
        keypoint_coordinates = tf.cast(keypoint_coordinates, tf.int32)
        if valid_vec is not None:
            valid_vec = tf.cast(valid_vec, tf.float32)
            valid_vec = tf.squeeze(valid_vec)
            cond_val = tf.greater(valid_vec, 0.5)
        else:
            cond_val = tf.ones_like(
                keypoint_coordinates[:, 0], dtype=tf.float32)
            cond_val = tf.greater(cond_val, 0.5)

        # 在map_size width范围内的坐标
        cond_1_in = tf.logical_and(
            tf.less(keypoint_coordinates[:, 0], map_size[1] - 1),
            tf.greater(keypoint_coordinates[:, 0], 0))
        # 在map_size height范围内的坐标
        cond_2_in = tf.logical_and(
            tf.less(keypoint_coordinates[:, 1], map_size[0] - 1),
            tf.greater(keypoint_coordinates[:, 1], 0))
        cond_in = tf.logical_and(cond_1_in, cond_2_in)
        #即 可见又在crop_image范围内的坐标
        cond = tf.logical_and(cond_val, cond_in)

        keypoint_coordinates = tf.cast(keypoint_coordinates, tf.float32)
        keypoint_number = keypoint_coordinates.get_shape().as_list()[0]
        Y = tf.reshape(
            tf.tile(tf.range(map_size[1]), [map_size[0]]),
            [map_size[1], map_size[0]])
        X = tf.transpose(Y)
        #[256,256,21]
        X = tf.tile(tf.expand_dims(X, -1), [1, 1, keypoint_number])
        Y = tf.tile(tf.expand_dims(Y, -1), [1, 1, keypoint_number])
        #[256,256,21]
        X_relative = tf.cast(X, tf.float32) - keypoint_coordinates[:, 1]
        Y_relative = tf.cast(Y, tf.float32) - keypoint_coordinates[:, 0]
        #每个gaussian_map上仅坐标等于关节点坐标的位置distance为0，强度最大，其他位置按照正态分布递减
        distance = tf.square(X_relative) + tf.square(Y_relative)
        gaussian_map = tf.exp(-distance / sigma) * tf.cast(cond, tf.float32)
        #背景 [256,256,1]
        #background_gaussian_map=tf.expand_dims(tf.zeros([map_size[0],map_size[1]]),-1)
        background_gaussian_map = tf.expand_dims(
            tf.ones([map_size[0], map_size[1]]), -1)
        #[size,size,1]
        total = tf.reduce_sum(gaussian_map, -1, keep_dims=True)
        background_gaussian_map -= total
        #[256,256,22]
        gaussian_map = tf.concat([gaussian_map, background_gaussian_map], -1)

        return gaussian_map


def boundary_calculate(label,height,width):
    '''
    description:根据图像和对应坐标计算该图像的边界，返回左上角和右下角坐标(y,x)
    args:
        label:标签
        height:图像高
        width:图像宽
    '''
    #可见的点 先x后y
    kpt_visible=tf.boolean_mask(label[:,:2],tf.cast(label[:,2],tf.bool))
    min_coor=tf.maximum(tf.reduce_min(kpt_visible,0),0)
    max_coor=tf.minimum(tf.reduce_max(kpt_visible,0),[width-1,height-1])
    #[y,x]
    return min_coor[::-1], max_coor[::-1]

def kpt_coor_translate(coor,scale,old_center,new_center):
    '''
    des:计算并返回resize后的各个关键点的新坐标(tf.int32)
    args:
        coor:原坐标
        scale:[2]，分别表示在height上的放缩比例和在width上的放缩比例
        old_center:原图的中心
        new_center:新图的中心
    '''
    #和后面scale乘时需要变为float
    coor_relative=tf.cast(coor-old_center,tf.float32)
    #tf.Print(coor_relative,[coor_relative],'coor_rel:')
    #new_coor_relative=tf.cast((coor_relative+0.5)*scale-0.5,tf.int32)
    new_coor_relative=(coor_relative+0.5)*scale-0.5
    new_coor=new_coor_relative+new_center
    new_size=new_center*2+1
    new_coor_x=tf.minimum(tf.maximum(new_coor[:,1],0.0),new_size[1])
    new_coor_y=tf.minimum(tf.maximum(new_coor[:,0],0),new_size[0])
    #print(new_coor_y.shape)
    new_coor=tf.stack([new_coor_y,new_coor_x],-1)    
    new_coor=tf.cast(tf.round(new_coor),tf.int32)
    #print(new_coor.shape)

    return new_coor


#根据位置裁剪手部并resize之
def image_crop_resize(image, crop_location, crop_size):
    """
    Crops an image. When factor is not given does an central crop.

    Inputs:
        image: 4D tensor, [ 1,height, width, channels] which will be cropped in height and width dimension
        crop_location: tensor, [2, 2] which represent the height and width location of the crop
        crop_size: int, describes the extension of the crop
    Outputs:
        image_crop: 3D tensor, [crop_size, crop_size, channels]
    """
    with tf.name_scope('crop_image_from_xy'):
        s = image.get_shape().as_list()
        assert len(
            s
        ) == 4, "Image needs to be of shape [all_jt, width, height, channel]"
        #scale=crop_size/crop_size_best
        crop_location = tf.cast(crop_location, tf.float32)
        crop_size = tf.cast(crop_size, tf.float32)

        crop_size_best = tf.maximum(crop_location[1, 0] - crop_location[0, 0],
                                    crop_location[1, 1] - crop_location[0, 1])
        #[2,1]
        center = tf.cast(
            tf.expand_dims([(crop_location[1, 0] - crop_location[0, 0]) / 2,
                            (crop_location[1, 1] - crop_location[0, 1]) / 2],
                           -1), tf.float32)
        #note 此处切片之后 y/x变为了1维的
        y_min = tf.maximum(center[0] - crop_size_best // 2, 0.0)
        y_max = tf.minimum(y_min + crop_size_best, input_para['height'])
        x_min = tf.maximum(center[1] - crop_size_best // 2, 0.0)
        x_max = tf.minimum(x_min + crop_size_best, input_para['width'])
        boxes = tf.stack([
            y_min / (input_para['height'] - 1), x_min /
            (input_para['width'] - 1), y_max /
            (input_para['height'] - 1), x_max / (input_para['width'] - 1)
        ], -1)
        box_ind = tf.range(s[0])
        #先从原图像中提取box指定的crop_image再进行resize到crop_size
        image_cropped_and_resized = tf.image.crop_and_resize(
            image, boxes, box_ind, tf.cast([crop_size, crop_size], tf.int32))
        image_cropped_and_resized = tf.squeeze(image_cropped_and_resized)
        #[resized_height,resized_width,channels]
        return image_cropped_and_resized
