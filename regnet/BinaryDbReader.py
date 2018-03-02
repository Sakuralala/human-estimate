from __future__ import print_function, unicode_literals
import os

import tensorflow as tf

from utils import *
#from utils.canonical_trafo import canonical_trafo, flip_right_hand
#from utils.relative_trafo import bone_rel_trafo


class BinaryDbReader(object):
    """
        Reads data from a binary dataset created by create_binary_db.py
    """
    def __init__(self, mode=None, batch_size=1, shuffle=True, use_wrist_coord=True, sigma=25.0, hand_crop=False,
                 random_crop_to_size=False,
                 scale_to_size=False,
                 hue_aug=False,
                 coord_uv_noise=False,
                 crop_center_noise=False, crop_scale_noise=False, crop_offset_noise=False,
                 gaussian_map_dropout=False):
        """ Inputs:
                mode: string, Indicates which binary file to read. Can be 'training' or 'evaluation'
                batch_size: int, Number of samples forming a batch
                shuffle: boolean, If true samples of binary file are shuffled while reading
                use_wrist_coord: boolean, When true keypoint #0 is the wrist(手腕), palm(手掌) center otherwise
                hand_crop: boolean, When true calculates a tight hand crop using the gt keypoint annotations
                                    Updates/Sets the following output items: image_crop, 
                                    crop_scale, gaussian_map, cam_mat, keypoint_uv21
                #Q sigma不是方差的大小吗。。。？
                sigma: float, Size of the ground truth gaussian_maps
                random_crop_to_size: boolean, Takes randomly sampled crops from the image & the mask
                scale_to_size: boolean, Scales down image and keypoints
                hue_aug: boolean, Random hue augmentation
                coord_uv_noise: boolean, Adds some gaussian noise on the gt uv coordinate
                crop_center_noise: boolean, Adds gaussian noise on the hand crop center location (all keypoints still lie within the crop)
                crop_scale_noise: boolean, Adds gaussian noise on the hand crop size
                crop_offset_noise: boolean, Offsets the hand crop center randomly (keypoints can lie outside the crop)
                gaussian_map_dropout: boolean, Randomly drop gaussian_map channels
        """
        self.path_to_db = '../hand3d/data/bin/'

        self.num_samples = 0
        if mode == 'training':
            self.path_to_db += 'rhd_training.bin'
            self.num_samples = 41258
        elif mode == 'evaluation':
            self.path_to_db += 'rhd_evaluation.bin'
            self.num_samples = 2728
        else:
            assert 0, "Unknown dataset mode."

        assert os.path.exists(self.path_to_db), "Could not find the binary data file!"

        # general parameters
        self.batch_size = batch_size
        self.sigma = sigma
        self.shuffle = shuffle
        self.use_wrist_coord = use_wrist_coord
        self.random_crop_to_size = random_crop_to_size
        #提取手部后再resize到的图片大小
        self.random_crop_size = 256
        # 缩小
        self.scale_to_size = scale_to_size
        self.scale_target_size = (240, 320)  # size its scaled down to if scale_to_size=True

        # data augmentation parameters
        # 数据增强  随机增加0.1的color？
        self.hue_aug = hue_aug
        self.hue_aug_max = 0.1

        self.hand_crop = hand_crop
        self.coord_uv_noise = coord_uv_noise
        # 增加一些噪声
        self.coord_uv_noise_sigma = 2.5  # std dev in px of noise on the uv coordinates
        self.crop_center_noise = crop_center_noise
        # 中心增加的噪声
        self.crop_center_noise_sigma = 20.0  # std dev in px: this moves what is in the "center", but the crop always contains all keypoints

        self.crop_scale_noise = crop_scale_noise
        self.crop_offset_noise = crop_offset_noise
        # 不知道啥玩意
        self.crop_offset_noise_sigma = 10.0  # translates the crop after size calculation (this can move keypoints outside)
        self.gaussian_map_dropout = gaussian_map_dropout
        self.gaussian_map_dropout_prob = 0.8

        # these are constants of the dataset and therefore must not be changed
        self.image_size = (320, 320)
        self.crop_size = 256
        self.num_kp = 42

    def get(self):
        """ Provides input data to the graph. """
        # calculate size of each record (this lists what is contained in the db and how many bytes are occupied)
        record_bytes = 2

        encoding_bytes = 4
        kp_xyz_entries = 3 * self.num_kp
        record_bytes += encoding_bytes*kp_xyz_entries

        encoding_bytes = 4
        kp_uv_entries = 2 * self.num_kp
        record_bytes += encoding_bytes*kp_uv_entries

        cam_matrix_entries = 9
        record_bytes += encoding_bytes*cam_matrix_entries

        image_bytes = self.image_size[0] * self.image_size[1] * 3
        record_bytes += image_bytes

        hand_parts_bytes = self.image_size[0] * self.image_size[1]
        record_bytes += hand_parts_bytes

        kp_vis_bytes = self.num_kp
        record_bytes += kp_vis_bytes

        """ READ DATA ITEMS"""
        # Start reader
        reader = tf.FixedLengthRecordReader(header_bytes=0, record_bytes=record_bytes)
        _, value = reader.read(tf.train.string_input_producer([self.path_to_db]))

        # decode to floats
        bytes_read = 0
        data_dict = dict()
        record_bytes_float32 = tf.decode_raw(value, tf.float32)

        # 1. Read keypoint xyz
        keypoint_xyz = tf.reshape(tf.slice(record_bytes_float32, [bytes_read//4], [kp_xyz_entries]), [self.num_kp, 3])
        bytes_read += encoding_bytes*kp_xyz_entries

        # calculate palm coord
        if not self.use_wrist_coord:
            palm_coord_l = tf.expand_dims(0.5*(keypoint_xyz[0, :] + keypoint_xyz[12, :]), 0)
            palm_coord_r = tf.expand_dims(0.5*(keypoint_xyz[21, :] + keypoint_xyz[33, :]), 0)
            keypoint_xyz = tf.concat([palm_coord_l, keypoint_xyz[1:21, :], palm_coord_r, keypoint_xyz[-20:, :]], 0)

        data_dict['keypoint_xyz'] = keypoint_xyz

        # 2. Read keypoint uv
        keypoint_uv = tf.cast(tf.reshape(tf.slice(record_bytes_float32, [bytes_read//4], [kp_uv_entries]), [self.num_kp, 2]), tf.int32)
        bytes_read += encoding_bytes*kp_uv_entries

        keypoint_uv = tf.cast(keypoint_uv, tf.float32)

        # calculate palm coord
        if not self.use_wrist_coord:
            palm_coord_uv_l = tf.expand_dims(0.5*(keypoint_uv[0, :] + keypoint_uv[12, :]), 0)
            palm_coord_uv_r = tf.expand_dims(0.5*(keypoint_uv[21, :] + keypoint_uv[33, :]), 0)
            keypoint_uv = tf.concat([palm_coord_uv_l, keypoint_uv[1:21, :], palm_coord_uv_r, keypoint_uv[-20:, :]], 0)

        if self.coord_uv_noise:
            noise = tf.truncated_normal([42, 2], mean=0.0, stddev=self.coord_uv_noise_sigma)
            keypoint_uv += noise

        data_dict['keypoint_uv'] = keypoint_uv

        # 3. Camera intrinsics
        cam_mat = tf.reshape(tf.slice(record_bytes_float32, [bytes_read//4], [cam_matrix_entries]), [3, 3])
        bytes_read += encoding_bytes*cam_matrix_entries
        data_dict['cam_mat'] = cam_mat

        # decode to uint8
        bytes_read += 2
        record_bytes_uint8 = tf.decode_raw(value, tf.uint8)

        # 4. Read image
        image = tf.reshape(tf.slice(record_bytes_uint8, [bytes_read], [image_bytes]),
                               [self.image_size[0], self.image_size[1], 3])
        image = tf.cast(image, tf.float32)
        bytes_read += image_bytes

        # subtract mean
        image = image / 255.0 - 0.5
        if self.hue_aug:
            image = tf.image.random_hue(image, self.hue_aug_max)
        data_dict['image'] = image

        # 5. Read mask
        hand_parts_mask = tf.reshape(tf.slice(record_bytes_uint8, [bytes_read], [hand_parts_bytes]),
                               [self.image_size[0], self.image_size[1]])
        hand_parts_mask = tf.cast(hand_parts_mask, tf.int32)
        bytes_read += hand_parts_bytes
        data_dict['hand_parts'] = hand_parts_mask
        #返回手部类
        hand_mask = tf.greater(hand_parts_mask, 1)
        #返回背景类+人
        bg_mask = tf.logical_not(hand_mask)
        #背景类+手部类 [H,W,2]
        data_dict['hand_mask'] = tf.cast(tf.stack([bg_mask, hand_mask], 2), tf.int32)

        # 6. Read visibilty
        keypoint_vis = tf.reshape(tf.slice(record_bytes_uint8, [bytes_read], [kp_vis_bytes]),
                               [self.num_kp])
        #[42]
        keypoint_vis = tf.cast(keypoint_vis, tf.bool)
        bytes_read += kp_vis_bytes

        # calculate palm visibility
        if not self.use_wrist_coord:
            palm_vis_l = tf.expand_dims(tf.logical_or(keypoint_vis[0], keypoint_vis[12]), 0)
            palm_vis_r = tf.expand_dims(tf.logical_or(keypoint_vis[21], keypoint_vis[33]), 0)
            #维度：[84]
            keypoint_vis = tf.concat([palm_vis_l, keypoint_vis[1:21], palm_vis_r, keypoint_vis[-20:]], 0)
        data_dict['keypoint_vis'] = keypoint_vis

        assert bytes_read == record_bytes, "Doesnt add up."

        """ DEPENDENT DATA ITEMS: SUBSET of 21 keypoints"""
        # figure out dominant hand(主手) by analysis of the segmentation mask
        one_map, zero_map = tf.ones_like(hand_parts_mask), tf.zeros_like(hand_parts_mask)
        cond_l = tf.logical_and(tf.greater(hand_parts_mask, one_map), tf.less(hand_parts_mask, one_map*18))
        #右手的mask 从18开始
        cond_r = tf.greater(hand_parts_mask, one_map*17)
        # where(condition,x,y,name)
        hand_map_l = tf.where(cond_l, one_map, zero_map)
        hand_map_r = tf.where(cond_r, one_map, zero_map)
        #左手占的总像素点数
        num_px_left_hand = tf.reduce_sum(hand_map_l)
        #右手占的总像素点数
        num_px_right_hand = tf.reduce_sum(hand_map_r)

        # PRODUCE the 21 subset using the segmentation masks
        # We only deal with the more prominent hand for each frame and discard the second set of keypoints
        kp_coord_xyz_left = keypoint_xyz[:21, :]
        kp_coord_xyz_right = keypoint_xyz[-21:, :]

        #Q:直接根据num_px_right_hand和num_px_left_hand的大小判断选择左右手不就行了，搞得这么麻烦？
        # [21,3]的全true或全false矩阵
        cond_left = tf.logical_and(tf.cast(tf.ones_like(kp_coord_xyz_left), tf.bool), tf.greater(num_px_left_hand, num_px_right_hand))
        kp_coord_xyz21 = tf.where(cond_left, kp_coord_xyz_left, kp_coord_xyz_right)

        #选像素点更多的那边作为要判断的手
        #左手标记为0 右手标记为1
        hand_side = tf.where(tf.greater(num_px_left_hand, num_px_right_hand),
                             tf.constant(0, dtype=tf.int32),
                             tf.constant(1, dtype=tf.int32))  # left hand = 0; right hand = 1
        #变为一个vector 其中hand_size指向的位置为1
        data_dict['hand_side'] = tf.one_hot(hand_side, depth=2, on_value=1.0, off_value=0.0, dtype=tf.float32)

        data_dict['keypoint_xyz21'] = kp_coord_xyz21

        # make coords relative to root joint
        #以palm作为根部
        kp_coord_xyz_root = kp_coord_xyz21[0, :] # this is the palm coord
        kp_coord_xyz21_rel = kp_coord_xyz21 - kp_coord_xyz_root  # relative coords in metric coords
        #以中指最底下那个指节作为参考长度
        index_root_bone_length = tf.sqrt(tf.reduce_sum(tf.square(kp_coord_xyz21_rel[12, :] - kp_coord_xyz21_rel[11, :])))
        data_dict['keypoint_scale'] = index_root_bone_length
        #以中指的根指节将坐标点进行归一化
        data_dict['keypoint_xyz21_normed'] = kp_coord_xyz21_rel / index_root_bone_length  # normalized by length of 12->11

        '''
        # calculate local coordinates
        #更为相对的坐标 每个手指的根部作为一个原点
        kp_coord_xyz21_local = bone_rel_trafo(data_dict['keypoint_xyz21_normed'])
        kp_coord_xyz21_local = tf.squeeze(kp_coord_xyz21_local)
        data_dict['keypoint_xyz21_local'] = kp_coord_xyz21_local

        # calculate viewpoint and coords in canonical(规范的) coordinates
        # TODO:旋转什么的 之后再说
        kp_coord_xyz21_rel_can, rot_mat = canonical_trafo(data_dict['keypoint_xyz21_normed'])
        kp_coord_xyz21_rel_can, rot_mat = tf.squeeze(kp_coord_xyz21_rel_can), tf.squeeze(rot_mat)
        kp_coord_xyz21_rel_can = flip_right_hand(kp_coord_xyz21_rel_can, tf.logical_not(cond_left))
        data_dict['keypoint_xyz21_can'] = kp_coord_xyz21_rel_can
        data_dict['rot_mat'] = tf.matrix_inverse(rot_mat)
        '''
        # Set of 21 for visibility
        keypoint_vis_left = keypoint_vis[:21]
        keypoint_vis_right = keypoint_vis[-21:]
        keypoint_vis21 = tf.where(cond_left[:, 0], keypoint_vis_left, keypoint_vis_right)
        data_dict['keypoint_vis21'] = keypoint_vis21

        # Set of 21 for UV coordinates
        keypoint_uv_left = keypoint_uv[:21, :]
        keypoint_uv_right = keypoint_uv[-21:, :]
        #因为uv坐标系第三维是visualble的标志，所以取前两个
        keypoint_uv21 = tf.where(cond_left[:, :2], keypoint_uv_left, keypoint_uv_right)
        data_dict['keypoint_uv21'] = keypoint_uv21

        """ DEPENDENT DATA ITEMS: HAND CROP """
        if self.hand_crop:
            #中指根部 [y,x]
            crop_center = keypoint_uv21[12, ::-1]

            # catch problem, when no valid kp available (happens almost never)
            crop_center = tf.cond(tf.reduce_all(tf.is_finite(crop_center)), lambda: crop_center,
                                  lambda: tf.constant([0.0, 0.0]))
            crop_center.set_shape([2, ])

            #增加均值为0标准差为crop_center_noise_sigma
            #移动了中心的坐标
            if self.crop_center_noise:
                noise = tf.truncated_normal([2], mean=0.0, stddev=self.crop_center_noise_sigma)
                crop_center += noise

            #Q:随机对比增强 啥玩意
            crop_scale_noise = tf.constant(1.0)
            if self.crop_scale_noise:
                crop_scale_noise = tf.squeeze(tf.random_uniform([1], minval=1.0, maxval=1.2))

            # select visible coords only
            #返回height方向上可见的部分
            kp_coord_h = tf.boolean_mask(keypoint_uv21[:, 1], keypoint_vis21)
            #返回width方向上可见的部分
            kp_coord_w = tf.boolean_mask(keypoint_uv21[:, 0], keypoint_vis21)
            #维度:[visuable_kp,2] 顺序是height到width
            kp_coord_hw = tf.stack([kp_coord_h, kp_coord_w], 1)

            # determine size of crop (measure spatial extend of hw coords first)
            #返回最小值的坐标值 reduce_min返回维度[2]，表示height轴和width轴的最小值
            min_coord = tf.maximum(tf.reduce_min(kp_coord_hw, 0), 0.0)
            #返回最大值的坐标值 reduce_max返回维度[2],表示height轴和width轴的最大值
            max_coord = tf.minimum(tf.reduce_max(kp_coord_hw, 0), self.image_size)

            # find out larger distance wrt the center of crop
            #寻找最佳裁剪尺寸[2],具体的值为[deltaH,deltaW]
            crop_size_best = 2*tf.maximum(max_coord - crop_center, crop_center - min_coord)
            #为了剪出一个正方形再求一次最大值
            crop_size_best = tf.reduce_max(crop_size_best)
            #正方形的尺寸应该在50-500间
            crop_size_best = tf.minimum(tf.maximum(crop_size_best, 50.0), 500.0)

            # catch problem, when no valid kp available
            crop_size_best = tf.cond(tf.reduce_all(tf.is_finite(crop_size_best)), lambda: crop_size_best,
                                  lambda: tf.constant(200.0))
            crop_size_best.set_shape([])

            # calculate necessary scaling
            # 放缩尺寸 256/最佳大小
            scale = tf.cast(self.crop_size, tf.float32) / crop_size_best
            scale = tf.minimum(tf.maximum(scale, 1.0), 10.0)
            scale *= crop_scale_noise
            data_dict['crop_scale'] = scale

            #Q:又来一个噪声？
            if self.crop_offset_noise:
                noise = tf.truncated_normal([2], mean=0.0, stddev=self.crop_offset_noise_sigma)
                crop_center += noise

            # Crop image
            # crop_image_from_xy(image,crop_location,crop_size,scale=1.0)
            # 从原图中先把手部提取出来再统一resize到crop_size大小，默认resize方法为bilinear,即双线性插值法。
            img_crop = crop_image_from_xy(tf.expand_dims(image, 0), crop_center, self.crop_size, scale)
            data_dict['image_crop'] = tf.squeeze(img_crop)

            # Modify uv21 coordinates
            crop_center_float = tf.cast(crop_center, tf.float32)
            #Q 这个坐标改的有点奇怪
            #A 先计算在原图中各个关节点相对于中指根部的关节点的坐标，变为相对坐标后再乘以放缩比例成为放缩后的相对坐标
            #  最后加一个resize_image的中心点的值就得到了在resize_image中各个关节点的相应坐标。
            # 维度:[21]
            keypoint_uv21_u = (keypoint_uv21[:, 0] - crop_center_float[1]) * scale + self.crop_size // 2
            keypoint_uv21_v = (keypoint_uv21[:, 1] - crop_center_float[0]) * scale + self.crop_size // 2
            #维度:[21,2]
            keypoint_uv21 = tf.stack([keypoint_uv21_u, keypoint_uv21_v], 1)
            data_dict['keypoint_uv21'] = keypoint_uv21

            # Modify camera intrinsics
            scale = tf.reshape(scale, [1, ])
            scale_matrix = tf.dynamic_stitch([[0], [1], [2],
                                              [3], [4], [5],
                                              [6], [7], [8]], [scale, [0.0], [0.0],
                                                               [0.0], scale, [0.0],
                                                               [0.0], [0.0], [1.0]])
            scale_matrix = tf.reshape(scale_matrix, [3, 3])

            crop_center_float = tf.cast(crop_center, tf.float32)
            trans1 = crop_center_float[0] * scale - self.crop_size // 2
            trans2 = crop_center_float[1] * scale - self.crop_size // 2
            trans1 = tf.reshape(trans1, [1, ])
            trans2 = tf.reshape(trans2, [1, ])
            trans_matrix = tf.dynamic_stitch([[0], [1], [2],
                                              [3], [4], [5],
                                              [6], [7], [8]], [[1.0], [0.0], -trans2,
                                                               [0.0], [1.0], -trans1,
                                                               [0.0], [0.0], [1.0]])
            trans_matrix = tf.reshape(trans_matrix, [3, 3])

            data_dict['cam_mat'] = tf.matmul(trans_matrix, tf.matmul(scale_matrix, cam_mat))

        """ DEPENDENT DATA ITEMS: Scoremap from the SUBSET of 21 keypoints"""
        # create gaussian_maps from the subset of 2D annoataion
        #Q:维度:[21,2] 先y再x
        keypoint_hw21 = tf.stack([keypoint_uv21[:, 1], keypoint_uv21[:, 0]], -1)
        gaussian_map_size = self.image_size

        if self.hand_crop:
            gaussian_map_size = (self.crop_size, self.crop_size)
        #Chaged: 2018.01.29
        #gaussian_map=make_gaussian_map(keypoint_hw21,gaussian_map_size,self.sigma,keypoint_vis21)
        gaussian_map=make_gt_heatmap(keypoint_hw21,(64,64),1.0,keypoint_vis21)

        if self.gaussian_map_dropout:
            gaussian_map = tf.nn.dropout(gaussian_map, self.gaussian_map_dropout_prob,
                                        noise_shape=[1, 1, 21])
            gaussian_map *= self.gaussian_map_dropout_prob

        data_dict['gaussian_map'] = gaussian_map

        if self.scale_to_size:
            image, keypoint_uv21, keypoint_vis21 = data_dict['image'], data_dict['keypoint_uv21'], data_dict['keypoint_vis21']
            s = image.get_shape().as_list()
            image = tf.image.resize_images(image, self.scale_target_size)
            scale = (self.scale_target_size[0]/float(s[0]), self.scale_target_size[1]/float(s[1]))
            keypoint_uv21 = tf.stack([keypoint_uv21[:, 0] * scale[1],
                                      keypoint_uv21[:, 1] * scale[0]], 1)

            data_dict = dict()  # delete everything else because the scaling makes the data invalid anyway
            data_dict['image'] = image
            data_dict['keypoint_uv21'] = keypoint_uv21
            data_dict['keypoint_vis21'] = keypoint_vis21

        elif self.random_crop_to_size:
            tensor_stack = tf.concat([data_dict['image'],
                                      tf.expand_dims(tf.cast(data_dict['hand_parts'], tf.float32), -1),
                                      tf.cast(data_dict['hand_mask'], tf.float32)], 2)
            s = tensor_stack.get_shape().as_list()
            tensor_stack_cropped = tf.random_crop(tensor_stack,
                                                  [self.random_crop_size, self.random_crop_size, s[2]])
            data_dict = dict()  # delete everything else because the random cropping makes the data invalid anyway
            data_dict['image'], data_dict['hand_parts'], data_dict['hand_mask'] = tensor_stack_cropped[:, :, :3],\
                                                                                  tf.cast(tensor_stack_cropped[:, :, 3], tf.int32),\
                                                                                  tf.cast(tensor_stack_cropped[:, :, 4:], tf.int32)

        names, tensors = zip(*data_dict.items())

        if self.shuffle:
            tensors = tf.train.shuffle_batch_join([tensors],
                                                  batch_size=self.batch_size,
                                                  capacity=100,
                                                  min_after_dequeue=50,
                                                  enqueue_many=False)
        else:
            tensors = tf.train.batch_join([tensors],
                                          batch_size=self.batch_size,
                                          capacity=100,
                                          enqueue_many=False)

        return dict(zip(names, tensors))

