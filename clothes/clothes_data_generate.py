import tensorflow as tf
import numpy as np
import csv
import os
from PIL import Image
from params_clothes import *
from utils import make_gt_heatmap, image_crop_resize, boundary_calculate


def _int64_feature(val):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))


def _bytes_feature(val):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))


def _float_feature(val):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[val]))


#TODO:改为类
class clothes_batch_generater():
    def __init__(self,
                 batch_size=8,
                 sigma=1.0,
                 crop_size=256,
                 crop_and_resize=True):
        '''
        args:
            batch_size:批大小
            crop_and_resize:是否根据目标坐标裁剪
            crop_size:裁剪后的图像大小
            sigma:二维高斯核的方差
        '''
        self.batch_size = batch_size
        self.crop_and_resize = crop_and_resize
        self.sigma = sigma
        self.crop_size = crop_size

    def _coor_read(self, coor):
        '''
        description:用以将标签中的字符串三元组坐标转换为np.ndarray并返回之
        args:
            coor:读取csv文件中的关节点部分
        '''
        ret = []
        for xyv in coor:
            #忽略没有的关节点
            if xyv[0] == '-':
                continue
            xyv = np.asarray([int(i) for i in xyv.split('_')], np.int)
            ret.append(xyv)
        #kx3
        return np.asarray(ret, np.int)

    def _data_generate_single(
            self,
            category,
            path='C:\\Users\\oldhen\\Downloads\\tianchi\\fashionAI_key_points_train_20180227\\train'
    ):
        '''
        description:用以生成对应category的图片与标签，返回两个list,分别为images和labels
        args:
            path:训练的目录
            category:服饰的类别，为了节省内存，还是一个个来把。。。
        '''
        csv_file = csv.reader(open(path + '/Annotations/train.csv', 'r'))

        images = []
        labels = []
        for i, row in enumerate(csv_file):
            #先把表头读出来
            if i == 0:
                header = row
            else:
                if category != row[1]:
                    continue
                img_name = path + '/' + row[0]
                #k*3
                coor = self._coor_read(row[2:])
                images.append(np.asarray(Image.open(img_name), np.float32))
                #size*k*3
                labels.append(coor)
        return images, labels

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
            images:包含单个类别的images
            labels_dict:包含单个类别的labels
            category:类别名
            file_path:生成的文件存放的路径
        '''
        print("Start coverting to tfrecord.")
        if category == None:
            raise ValueError("Please give an correct category.")
        images, labels = self._data_generate_single(category, data_path)
        file_name = tf_path + '/' + category + '.tfrec'
        num = len(images)
        with tf.python_io.TFRecordWriter(file_name) as writer:
            #序列化并填充至example  然后写入磁盘
            #方向
            #labels.shape 返回元祖  在len以取得维数
            for i in range(num):
                image_raw = images[i].tostring()
                label_raw = labels[i].tostring()
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            #'height': _int64_feature(height),
                            #'width': _int64_feature(width),
                            'label_raw': _bytes_feature(label_raw),
                            'image_raw': _bytes_feature(image_raw)
                        }))
                writer.write(example.SerializeToString())
        print("Covert %s done." % category)

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
        for key in labels_dict.keys():
            images = images_dict[key]
            labels = labels_dict[key]
            file_name = file_path + '\\' + key + '.tfrec'
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
        print("Covert all done.")

    def _read_from_tfrecords(self, file_queue):
        reader = tf.TFRecordReader()
        key, val = reader.read(file_queue)
        #返回的feature是一个字典
        feature = tf.parse_single_example(
            val,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label_raw': tf.FixedLenFeature([], tf.string)
            })

        image = tf.decode_raw(feature['image_raw'], tf.float32)
        reshaped_image = tf.reshape(image, [
            input_para['height'], input_para['width'], input_para['channels']
        ])
        #k*3
        label = tf.decode_raw(feature['label_raw'], tf.float32)
        reshaped_label = tf.reshape(label, [-1, 3])
        #关键点数量
        #kpt_num=reshaped_label.get_shape().as_list()[0]
        is_visible = tf.cast(reshaped_label[:, -1], tf.bool)
        #[k,2]
        coor_yx = tf.cast(
            tf.stack([reshaped_label[:, 1], reshaped_label[:, 0]], -1),
            tf.float32)
        gt_heatmaps = make_gt_heatmap(coor_yx, (input_para['height'],
                                                input_para['width'])
                                      if self.crop_and_resize == False else
                                      (self.crop_size,
                                       self.crop_size), self.sigma, is_visible)
        #TODO:加上数据增强(data augement)
        if self.crop_and_resize:
            cropped_image = image_crop_resize(
                tf.expand_dims(reshaped_image, 0),
                [boundary_calculate(reshaped_label)], self.crop_size)
            return cropped_image,gt_heatmaps

        return reshaped_image, gt_heatmaps

    def batch_generate(self, tf_file):
        if not os.path.exists(tf_file):
            raise ValueError("Please give an correct path.")
    #生成输入文件队列   注意参数应该是string tensor  所以加个[]
        if train_para['is_train']:
            file_queue = tf.train.string_input_producer([tf_file])
        else:
            file_queue = tf.train.string_input_producer([tf_file], 2)

        read_result = self._read_from_tfrecords(file_queue)
        if train_para['is_train']:
            min_num = int(input_para['train_num'] * 0.5)
            images_batch, labels_batch = tf.train.shuffle_batch(
                [read_result[0], read_result[1]],
                batch_size=self.batch_size,
                num_threads=16,
                min_after_dequeue=min_num,
                capacity=min_num + self.batch_size * 10)
        else:
            min_num = int(input_para['test_num'] * 0.36)
            images_batch, labels_batch = tf.train.batch(
                [read_result[0], read_result[1]],
                batch_size=self.batch_size,
                num_threads=8,
                capacity=10000)

        return images_batch, labels_batch

    #生成二进制文件测试
    '''
    categories = ['blouse', 'skirt', 'outwear', 'trousers', 'dress']
    for cat in categories:
        result = _data_generate_single(cat)
        convert_to_tfrecords_single(result[0], result[1], cat)
        #for gc
        result = []
    '''
