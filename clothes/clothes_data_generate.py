import tensorflow as tf
import numpy as np
import scipy
import csv
import os
from PIL import Image
from params_clothes import *
from utils import *
import sys
import cv2


def _int64_feature(val):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))


def _bytes_feature(val):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))


def _float_feature(val):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[val]))


class clothes_batch_generater():
    def __init__(self,
                 category,
                 batch_size=8,
                 sigma=1.0,
                 cropped_resized_size=256,
                 crop_and_resize=True,
                 coor_noise=True,
                 crop_center_noise=True):
        '''
        args:
            batch_size:批大小
            crop_and_resize:是否根据目标坐标裁剪
            cropped_resized_size:裁剪后的图像大小
            sigma:二维高斯核的方差
	    coor_noise:是否在关节点的位置上加上一定噪声
	    crop_center_noise:是否在crop图像的中心加上一定噪声
        '''
        self.batch_size = batch_size
        self.crop_and_resize = crop_and_resize
        self.sigma = sigma
        self.cropped_resized_size = cropped_resized_size
        self.coor_noise = coor_noise
        self.crop_center_noise = crop_center_noise
        self.category = category

    def _coor_read(self, cat, coor):
        '''
        description:用以将标签中的字符串三元组坐标转换为np.ndarray并返回之
        args:
            cat:类别
            coor:读取csv文件中的关节点部分
        '''
        ret = []
        for i, xyv in enumerate(coor):
            #忽略对应类别没有的关节点
            if xyv[0] == '-':
                if all_kpt_list[i] in categories_dict[cat]:
                    #对于相同类别下可能不存在的关节点直接置全零
                    #xyv=np.asarray([0,0,0],np.int)
                    ret.append(np.asarray([0, 0, 0], np.int32))
            else:
                xyv = np.asarray([int(i) for i in xyv.split('_')], np.int)
                ret.append(xyv)
        #kx3
        return np.asarray(ret, np.int32)

    def _coor_read_full(self, coor):
        '''
        description:用以将标签中的字符串三元组坐标转换为np.ndarray并返回之,全关键点版本，即不分类别
        args:
            coor:读取csv文件中的关节点部分
        '''
        ret = []
        for i, xyv in enumerate(coor):
            if xyv[0] == '-':
                ret.append(np.asarray([0, 0, 0], np.int32))
            else:
                xyv = np.asarray([int(i) for i in xyv.split('_')], np.int)
                ret.append(xyv)
        #kx3
        return np.asarray(ret, np.int32)

    def _data_generate_single(
            self,
            category,
            path='C:\\Users\\oldhen\\Downloads\\tianchi\\fashionAI_key_points_train_20180227\\train'
    ):
        '''
        description:用以生成对应category的图片与标签，返回两个list,分别为images和labels
        或者在test时返回图片和对应名字
        args:
            path:训练的目录
            category:服饰的类别，为了节省内存，还是一个个来把。。。
        '''
        images = []
        if train_para['is_train']:
            labels = []
            csv_file = csv.reader(open(path + '/Annotations/train.csv', 'r'))
        else:
            img_names = []
            csv_file = csv.reader(open(path + '/Annotations/test.csv', 'r'))
        header = None
        for i, row in enumerate(csv_file):
            #先把表头读出来
            if i == 0:
                header = row
            else:
                if category != row[1]:
                    continue
                img_name = path + '/' + row[0]
                img = np.asarray(Image.open(img_name))
                images.append(img)
                if train_para['is_train']:
                    #k*3
                    coor = self._coor_read(category, row[2:])
                    #size*k*3
                    labels.append(coor)
                    return images, labels
                else:
                    img_names.append(img_name)
                    return images, img_names

    def _data_generate(
            self,
            path='C:\\Users\\oldhen\\Downloads\\tianchi\\fashionAI_key_points_train_20180227\\train'
    ):
        '''
        description:用以生成对应的图片与标签，返回两个dict，每个dict内部分别为5种类别的list，两个dict中的image和label一一对应
        args:
            path:训练的目录
        '''
        csv_file = csv.reader(open(path + '/Annotations/train.csv', 'r'))
        #下次不要这样写了。。。dict的hash以及多个list到后面太占内存了。。。。64gb都差点被爆了
        clothes_labels = {
            'blouse': [],
            'skirt': [],
            'outwear': [],
            'trousers': [],
            'dress': []
        }
        clothes_images = {
            'blouse': [],
            'skirt': [],
            'outwear': [],
            'trousers': [],
            'dress': []
        }
        for i, row in enumerate(csv_file):
            #先把表头读出来
            if i == 0:
                header = row
            else:
                category = row[1]
                img_name = path + '/' + row[0]
                coor = self._coor_read(row[2:])
                img = Image.open(img_name)
                #在对应的类别下添加
                clothes_labels[category].append(coor)
                clothes_images[category].append(np.asarray(img, np.float32))
                return clothes_images, clothes_labels

    def convert_to_tfrecords_single(self,
                                    category,
                                    data_path='',
                                    tf_path='D:\\tfrecords\\'):
        '''
        description:用以将images和labels转换为tensorflow的专用二进制格式便于后续读取
        args:
            category:类别名
            data_path:数据的路径
            tf_path:生成的tfrec文件的存放路径
        '''
        print("Start coverting to tfrecord.")
        sys.stdout.flush()
        if category == None:
            raise ValueError("Please give an correct category.")
        ret = self._data_generate_single(category, data_path)
        file_name = tf_path + '/' + category + '.tfrec'
        num = len(ret[0])
        #print(num)
        #sys.stdout.flush()
        with tf.python_io.TFRecordWriter(file_name) as writer:
            #序列化并填充至example  然后写入磁盘
            for i in range(num):
                image_raw = ret[0][i].tostring()
                features = {
                    'height': _int64_feature(ret[0][i].shape[0]),
                    'width': _int64_feature(ret[0][i].shape[1]),
                    'channels': _int64_feature(ret[0][i].shape[2]),
                    'image_raw': _bytes_feature(image_raw)
                }
                if train_para['is_train']:
                    label_raw = ret[1][i].tostring()
                    features['label_raw'] = _bytes_feature(label_raw)
                else:  #str类型不需要再进行tostring操作
                    features['image_name'] = _bytes_feature(ret[1][i])
                example = tf.train.Example(features=tf.train.Features(features))
                writer.write(example.SerializeToString())
        print("Covert %s done." % category)
        sys.stdout.flush()

    def convert_to_tfrecords(
            self,
            images_dict,
            labels_dict,
            file_path='C:\\Users\\oldhen\\Downloads\\tianchi'):
        '''
        description:用以将images和labels转换为tensorflow的专用二进制格式便于后续读取
        args:
            images_dict:包含各个类别的images的字典
            labels_dict:包含各个类别的labels的字典
            file_path:生成的文件存放的路径
        '''
        print("Start coverting to tfrecord.")
        sys.stdout.flush()
        for key in labels_dict.keys():
            images = images_dict[key]
            labels = labels_dict[key]
            file_name = file_path + '/' + key + '.tfrec'
            num = len(images)
            height = images[0].shape[0]
            width = images[0].shape[1]
            with tf.python_io.TFRecordWriter(file_name) as writer:
                # 序列化并填充至example  然后写入磁盘
                #方向
                #labels.shape 返回元祖  在len以取得维数
                for i in range(num):
                    image_raw = images[i].tostring()
                    label_raw = labels[i].tostring()
                    if labels[i].shape[0] != 13 or labels[i].shape[1] != 3:
                        print(i)
                        sys.stdout.flush()
                    if images[i].shape[0] != 512 or images[i].shape[1] != 512:
                        print(i)
                        sys.stdout.flush()
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'height': _int64_feature(height),
                                'width': _int64_feature(width),
                                'label_raw': _bytes_feature(label_raw),
                                'image_raw': _bytes_feature(image_raw)
                            }))
                    writer.write(example.SerializeToString())
            print("Covert %s done." % file_name)
            sys.stdout.flush()
        print("Covert all done.")
        sys.stdout.flush()

    def _parse_function(self, example_proto):
        features = {
            "image_raw": tf.FixedLenFeature((), tf.string),
            "label_raw": tf.FixedLenFeature((), tf.string),
            "height": tf.FixedLenFeature((), tf.int64),
            "width": tf.FixedLenFeature((), tf.int64),
            'channels': tf.FixedLenFeature((), tf.int64)
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.decode_raw(parsed_features['image_raw'], tf.uint8)
        height = tf.cast(parsed_features['height'], tf.int32)
        width = tf.cast(parsed_features['width'], tf.int32)
        channels = tf.cast(parsed_features['channels'], tf.int32)
        shape = tf.stack([height, width, channels])
        #normalize [-.5,.5]
        reshaped_image = tf.reshape(tf.cast(image, tf.float32),
                                    shape) / 255.0 - 0.5
        #k*3
        #注意是int32
        label = tf.decode_raw(parsed_features['label_raw'], tf.int32)
        reshaped_label = tf.reshape(label,
                                    [len(categories_dict[self.category]), 3])
        #return reshaped_image, reshaped_label
        #关键点数量
        is_visible = tf.cast(reshaped_label[:, -1], tf.bool)
        #[k,2] 在原尺寸图像上的坐标
        coor_yx = tf.stack([reshaped_label[:, 1], reshaped_label[:, 0]], -1)
        if self.coor_noise:
            coor_yx += tf.truncated_normal(
                [tf.shape(coor_yx)[0],
                 tf.shape(coor_yx)[1]],
                mean=0.0,
                stddev=1.5)
            #保证在图像范围内
            coor_yx = tf.maximum(
                tf.minimum(coor_yx, [height - 1, width - 1]), [0, 0])
        #TODO:加上数据增强(data augement)
#加上rotate、flip等，那么对应的坐标点如何计算？
        reshaped_image = tf.image.random_contrast(
            reshaped_image, lower=0.5, upper=1.2)

        #要crop的话
        if self.crop_and_resize:
            min_coor, max_coor = boundary_calculate(reshaped_label, height,
                                                    width)
            sub = max_coor - min_coor
            #tf.Print(sub,[sub],'sub:')
            #不crop成正方形直接resize试一下
            #cropped_resized_size_square=tf.maximum(sub[0],sub[1])
            #old_center = (max_coor+min_coor)//2
            old_center = tf.cast(min_coor,
                                 tf.float32) + tf.cast(sub, tf.float32) / 2
            if self.crop_center_noise:
                old_center += tf.truncated_normal(
                    [tf.shape(old_center)[0],
                     tf.shape(old_center)[1]],
                    mean=0.0,
                    stddev=10)
                sub = tf.maximum(old_center - tf.cast(min_coor, tf.float32),
                                 tf.cast(max_coor - old_center,
                                         tf.float32)) * 2
#不能超出图片尺寸
#sub=tf.maximum(tf.minimum(sub,[height,width]),[0,0])

#新的边界点
#min_coor=old_center-crop_size/2
#min_coor=tf.maximum(tf.minimum(min_coor,[height-1,width-1]),[0,0])
#max_coor=old_center+crop_size/2
#max_coor=tf.maximum(tf.minimum(max_coor,[height-1,width-1]),[0,0])

#scale必须是浮点数
            cropped_resized_size = tf.stack(
                [self.cropped_resized_size, self.cropped_resized_size])
            scale = tf.cast(cropped_resized_size, tf.float32) / tf.cast(
                sub, tf.float32)
            #tf.Print(scale,[scale],'scale:')
            #经过resize后新的关键点坐标
            coor_yx = kpt_coor_translate(
                tf.cast(coor_yx, tf.float32), scale, old_center,
                tf.cast(cropped_resized_size, tf.float32) / 2 - 1)
            #coor_yx=tf.cast(tf.cast(coor_yx,tf.float32)*scale,tf.int32)
            min_coor = tf.cast(min_coor, tf.float32)
            max_coor = tf.cast(max_coor, tf.float32)
            height = tf.cast(height, tf.float32)
            width = tf.cast(width, tf.float32)
            #在原图上所占的比例
            boxes = tf.stack([
                min_coor[0] / height, min_coor[1] / width,
                max_coor[0] / height, max_coor[1] / width
            ])
            boxes_ind = tf.range(1)
            #crop并且resize到指定大小
            cropped_resized_image = tf.image.crop_and_resize(
                tf.expand_dims(reshaped_image, 0), tf.expand_dims(boxes, 0),
                boxes_ind, cropped_resized_size)
            cropped_resized_image = tf.squeeze(cropped_resized_image)
            gt_heatmaps = make_gt_heatmap(
                coor_yx,
                (self.cropped_resized_size, self.cropped_resized_size),
                self.sigma, is_visible)
            #coor=tf.concat([coor_yx,tf.cast(tf.expand_dims(is_visible,-1),tf.int32)],-1)
            cropped_resized_image.set_shape(
                [self.cropped_resized_size, self.cropped_resized_size, 3])
            return cropped_resized_image, gt_heatmaps
        else:  #不crop，直接resize(保证batch中各个的维度一致)
            resized_image = tf.image.resize_bilinear(
                tf.expand_dims(reshaped_image, 0),
                [input_para['height'], input_para['width']])
            old_center = tf.stack([height / 2, width / 2])
            new_center = tf.stack(
                [input_para['height'] / 2, input_para['width'] / 2])
            scale = tf.stack([
                tf.cast(height, tf.float32) / input_para['height'],
                tf.cast(width, tf.float32) / input_para['width']
            ])
            coor_yx = kpt_coor_translate(coor_yx, scale, old_center,
                                         new_center)
            gt_heatmaps = make_gt_heatmap(
                coor_yx, (input_para['height'], input_para['width']),
                self.sigma, is_visible)
            resized_image.set_shape(
                [input_para['height'], input_para['width'], 3])
            return resized_image, gt_heatmaps

    #test时的处理方式
    def _parse_function2(self, example_proto):
        features = {
            "image_raw": tf.FixedLenFeature((), tf.string),
            "image_name": tf.FixedLenFeature((), tf.string),
            "height": tf.FixedLenFeature((), tf.int64),
            "width": tf.FixedLenFeature((), tf.int64),
            'channels': tf.FixedLenFeature((), tf.int64)
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.decode_raw(parsed_features['image_raw'], tf.uint8)
        height = tf.cast(parsed_features['height'], tf.int32)
        width = tf.cast(parsed_features['width'], tf.int32)
        channels = tf.cast(parsed_features['channels'], tf.int32)
        shape = tf.stack([height, width, channels])
        #normalize [-.5,.5]TODO：test时需不需要norm?
        reshaped_image = tf.reshape(tf.cast(image, tf.float32),shape)
        image_name=parsed_features['image_name']
        #TODO卷积操作需要知道shape的每一维
        #reshaped_image.set_shape([])
        return reshaped_image,image_name

    #使用新的tf.Data API来生成batch
    def dataset_input_new(self, tf_file):
        dataset = tf.data.TFRecordDataset(tf_file)
        if train_para['is_train']:
            dataset = dataset.map(self._parse_function)
            dataset = dataset.shuffle(800)
            dataset = dataset.repeat() 
        else:
            dataset = dataset.map(self._parse_function2)
            dataset = dataset.repeat(1) 
        dataset = dataset.batch(self.batch_size)
            
        #迭代生成批次数据
        iterator = dataset.make_one_shot_iterator()
        batched_images, batched_labels = iterator.get_next()
        #生成的batch中各个的维度必须相同
        return batched_images, batched_labels


#生成二进制文件测试
'''
categories = ['blouse', 'skirt', 'outwear', 'trousers', 'dress']
for cat in categories:
    tmp = clothes_batch_generater(cat)
    tmp.convert_to_tfrecords_single(
        cat,
        'C:\\Users\\oldhen\\Downloads\\tianchi\\fashionAI_key_points_train_20180227\\train'
    )
    #for gc
    tmp = None
'''

#从tfrecords中读取测试
with tf.Session() as sess:
    #print(len(categories_dict['blouse']))
    gen = clothes_batch_generater('blouse', 2)
    if not os.path.exists(dir_para['tf_dir']):
        os.mkdir(dir_para['tf_dir'])
        print('Created tfrecords dir:', dir_para['tf_dir'])
        sys.stdout.flush()
    rec_name = 'blouse.tfrec'
    for (root, dirs, files) in os.walk(dir_para['tf_dir']):
        if rec_name not in files:
            gen.convert_to_tfrecords_single(
                'blouse', dir_para['train_data_dir'], dir_para['tf_dir'])
    batch = gen.dataset_input_new(dir_para['tf_dir'] + '\\' + rec_name)
    '''
    sess.run(tf.global_variables_initializer())
    img, lb,coor = sess.run(batch)
    #print(img)
    scipy.misc.imsave('test1.png', img[0])
    scipy.misc.imsave('test2.png', img[1])
    lb_resized=tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(lb[0],0),[256,256]))
    lb_resized=sess.run(lb_resized)
    scipy.misc.imsave('hp01.png', lb_resized[:,:,0])
    scipy.misc.imsave('hp02.png', lb_resized[:,:,1])
    #最大的值
    max_val=tf.reduce_max(lb[0],0)
    max_val=tf.reduce_max(max_val,0)
    print(sess.run(max_val))

    #最大值的位置
    max_idx_list=[]
    for i in range(lb_resized.shape[-1]):
        ind=np.unravel_index(np.argmax(lb_resized[:,:,i]),lb_resized.shape[:2])
        max_idx_list.append(ind)
    print(max_idx_list)

    print(coor[0])
    '''
    