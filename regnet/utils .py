#一些杂项
import tensorflow as tf


def make_gaussian_map(keypoint_coordinates, map_size, sigma, valid_vec=None):
    '''
    key_point_coordinates:各个关节点的坐标 [21,2] 先y后x
    map_size:输出的heatmap大小
    sigma:正态分布的方差
    return:kp张gaussian_map
    '''
    with tf.name_scope('gaussian_map'):
        sigma = tf.cast(sigma, tf.float32)
        assert len(map_size) == 2,'Gaussian map size must be 2.'
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
            tf.tile(tf.range(map_size[1]), [map_size[0]]), [map_size[1], map_size[0]])
        X = tf.transpose(Y)
        #[256,256,21]
        X = tf.tile(tf.expand_dims(X, -1), [1, 1, keypoint_number])
        Y = tf.tile(tf.expand_dims(Y, -1), [1, 1, keypoint_number])
        #[256,256,21]
        X_relative = tf.cast(X,tf.float32) - keypoint_coordinates[:, 1]
        Y_relative = tf.cast(Y,tf.float32) - keypoint_coordinates[:, 0]
        #每个gaussian_map上仅坐标等于关节点坐标的位置distance为0，强度最大，其他位置按照正态分布递减
        distance = tf.square(X_relative) + tf.square(Y_relative)
        gaussian_map=tf.exp(-distance/sigma)*tf.cast(cond,tf.float32)
        #背景 [256,256,1]
        #background_gaussian_map=tf.expand_dims(tf.zeros([map_size[0],map_size[1]]),-1)
        background_gaussian_map=tf.expand_dims(tf.ones([map_size[0],map_size[1]]),-1)
	#[size,size,1]
        total=tf.reduce_sum(gaussian_map,-1,keep_dims=True)      
        background_gaussian_map-=total
        #[256,256,22]
        gaussian_map=tf.concat([gaussian_map,background_gaussian_map],-1)

        return gaussian_map



#根据位置裁剪手部并resize之
def crop_image_from_xy(image, crop_location, crop_size, scale=1.0):
    """
    Crops an image. When factor is not given does an central crop.

    Inputs:
        image: 4D tensor, [batch, height, width, channels] which will be cropped in height and width dimension
        crop_location: tensor, [batch, 2] which represent the height and width location of the crop
        crop_size: int, describes the extension of the crop
    Outputs:
        image_crop: 4D tensor, [batch, crop_size, crop_size, channels]
    """
    with tf.name_scope('crop_image_from_xy'):
        s = image.get_shape().as_list()
        assert len(
            s
        ) == 4, "Image needs to be of shape [batch, width, height, channel]"
        #scale=crop_size/crop_size_best
        scale = tf.reshape(scale, [-1])
        crop_location = tf.cast(crop_location, tf.float32)
        crop_location = tf.reshape(crop_location, [s[0], 2])
        crop_size = tf.cast(crop_size, tf.float32)

        #Q:然后这里就变成了crop_size_best....为啥不直接传进crop_size_best呢？
        crop_size_scaled = crop_size / scale
        #左边界
        y1 = crop_location[:, 0] - crop_size_scaled // 2
        #右边界
        y2 = y1 + crop_size_scaled
        #上边界
        x1 = crop_location[:, 1] - crop_size_scaled // 2
        #下边界
        x2 = x1 + crop_size_scaled
        #在原图像height轴上占的比例
        y1 /= s[1]
        #在原图像height轴上占的比例
        y2 /= s[1]
        #在原图像width轴上占的比例
        x1 /= s[2]
        #在原图像width轴上占的比例
        x2 /= s[2]
        #[1,4] 可以有[batch_size,4]个boxes分别用来对应batch_size张图片
        boxes = tf.stack([y1, x1, y2, x2], -1)
        #[1,2]
        crop_size = tf.cast(tf.stack([crop_size, crop_size]), tf.int32)
        #[1] 指定第几个图片
        box_ind = tf.range(s[0])
        #先从原图像中提取box指定的crop_image再进行resize到crop_size
        image_c = tf.image.crop_and_resize(
            tf.cast(image, tf.float32), boxes, box_ind, crop_size, name='crop')
        return image_c


class LearningRateScheduler:
    """
        Provides scalar tensors at certain iteration as is needed for a multistep learning rate schedule.
    """

    def __init__(self, steps, values):
        self.steps = steps
        self.values = values

        assert len(steps) + 1 == len(
            values), "There must be one more element in value as step."

    def get_lr(self, global_step):
        with tf.name_scope('lr_scheduler'):

            if len(self.values) == 1:  #1 value -> no step
                learning_rate = tf.constant(self.values[0])
            elif len(self.values) == 2:  #2 values -> one step
                cond = tf.greater(global_step, self.steps[0])
                learning_rate = tf.where(cond, self.values[1], self.values[0])
            else:  # n values -> n-1 steps
                cond_first = tf.less(global_step, self.steps[0])

                cond_between = list()
                for ind, step in enumerate(range(0, len(self.steps) - 1)):
                    cond_between.append(
                        tf.logical_and(
                            tf.less(global_step, self.steps[ind + 1]),
                            tf.greater_equal(global_step, self.steps[ind])))

                cond_last = tf.greater_equal(global_step, self.steps[-1])

                cond_full = [cond_first]
                cond_full.extend(cond_between)
                cond_full.append(cond_last)

                cond_vec = tf.stack(cond_full)
                lr_vec = tf.stack(self.values)

                learning_rate = tf.where(cond_vec, lr_vec,
                                         tf.zeros_like(lr_vec))

                learning_rate = tf.reduce_sum(learning_rate)

            return learning_rate
