#新的数据生成
import math
import tensorflow as tf
import numpy as np
import scipy
from scipy import misc
import csv
import os
from PIL import Image
from params_clothes import *
from utils import *
import sys
#import cv2
import platform


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
                 resized_size=256,
                 is_resize=True,
                 coor_noise=True,
                 crop_center_noise=True):
        '''
        args:
            batch_size:批大小
            is_resize:是否根据目标坐标裁剪
            resized_size:裁剪后的图像大小
            sigma:二维高斯核的方差
	    coor_noise:是否在关节点的位置上加上一定噪声
	    crop_center_noise:是否在crop图像的中心加上一定噪声
        '''
        self.batch_size = batch_size
        self.is_resize = is_resize
        self.sigma = sigma
        self.resized_size = resized_size
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
                    ret.append(np.asarray([0, 0, 0], np.int32))
            else:
                xyv = np.asarray([int(i) for i in xyv.split('_')], np.int)
                ret.append(xyv)
        #kx3
        return np.asarray(ret, np.int32)


    def data_generate_single(
            self,
            category,
            path=''
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
            if platform.system() == 'Linux':
                csv_file = csv.reader(
                    open(path + '/Annotations/train.csv', 'r'))
            elif platform.system() == 'Windows':
                csv_file = csv.reader(
                    open(path + '\\Annotations\\train.csv', 'r'))
            else:
                raise ValueError("Unknown system.")
        else:
            img_names = []
            cat_list=[]
            if platform.system() == 'Windows':
                csv_file = csv.reader(open(path + '\\test.csv', 'r'))
            elif platform.system() == 'Linux':
                csv_file = csv.reader(open(path + '/test.csv', 'r'))
            else:
                raise ValueError("Unknown system.")
        header = None
        for i, row in enumerate(csv_file):
            #先把表头读出来
            if i == 0:
                header = row
            else:
                if category != row[1] and category!='full':
                    continue
                if platform.system() == 'Windows':
                    img_name = path + '\\' + row[0]
                elif platform.system() == 'Linux':
                    img_name = path + '/' + row[0]
                img = np.asarray(Image.open(img_name),np.uint8)
                images.append(img)
                if train_para['is_train']:
                    #k*3
                    coor = self._coor_read(category, row[2:])
                    #size*k*3
                    labels.append(coor)
                else:
                    cat_list.append(row[1])
                    img_names.append(row[0])
        if train_para['is_train']:
            return np.asarray(images),labels
        else:
            return np.asarray(images),img_names,cat_list


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
        print("Start coverting  %s to tfrecord."%category)
        sys.stdout.flush()
        if category == None:
            raise ValueError("Please give an correct category.")
        #print(data_path)
        ret = self.data_generate_single(category, data_path)
        if platform.system() == 'Windows':
            file_name = tf_path + '\\' + category + '.tfrec'
        elif platform.system() == 'Linux':
            file_name = tf_path + '/' + category + '.tfrec'
        num = len(ret[0])
        print('image number:%s'%num)
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
                    features['image_name'] = _bytes_feature(ret[1][i].encode())
                    features['cat'] = _bytes_feature(ret[2][i].encode())
                example = tf.train.Example(
                    features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())
        print("Covert %s done." % category)
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
        #print(reshaped_image.shape)
        #k*3
        #注意是int32
        label = tf.decode_raw(parsed_features['label_raw'], tf.int32)
        reshaped_label = tf.reshape(label,
                                    [len(categories_dict[self.category]), 3])
        #return reshaped_image, reshaped_label
        #关键点数量
        is_visible = tf.cast(reshaped_label[:, -1], tf.bool)
        #[k,2] 在原尺寸图像上的坐标
        coor_yx = tf.cast(tf.stack([reshaped_label[:, 1], reshaped_label[:, 0]], -1),tf.float32)
        if self.coor_noise:
            coor_yx += tf.truncated_normal(
                [tf.shape(coor_yx)[0],
                 tf.shape(coor_yx)[1]],
                mean=0.0,
                stddev=1.5)
            #保证在图像范围内
            coor_yx = tf.maximum(
                tf.minimum(coor_yx, tf.cast(tf.stack([height - 1, width - 1]),tf.float32)), [0, 0])
        #数据增强
        reshaped_image = tf.image.random_contrast(
            reshaped_image, lower=0.5, upper=1.2)
        reshaped_image=tf.image.random_hue(reshaped_image,0.2)
        reshaped_image=tf.image.random_brightness(reshaped_image,0.3)

        if self.is_resize:
            #新的大小
            resized_size=tf.cast(tf.stack([self.resized_size,self.resized_size]),tf.float32)
            #原大小
            origin_size=tf.cast(shape[0:2],tf.float32)
            #比例 表示缩放到heatmap大小
            scale=resized_size/origin_size
            #对于heatmap大小的坐标
            coor_yx_hp = kpt_coor_translate(
                coor_yx, scale, origin_size/2-1,
                resized_size/2 - 1)
            resized_image=tf.image.resize_bilinear(tf.expand_dims(reshaped_image,0),
                [self.resized_size,self.resized_size],align_corners=True)
            print(resized_image)
            resized_image=tf.squeeze(resized_image)
            gt_heatmaps = make_gaussian_map(
                coor_yx_hp,
                (self.resized_size, self.resized_size),
                self.sigma, is_visible)
            #在此处加上旋转操作，就不需要计算坐标了
            #随机旋转[-30,30]
            #angle=np.random.uniform(-30.0,30.0)
            angle=tf.random_uniform([1],-30.0,30.0)
            #print(angle)
            radian=angle*math.pi/180
            #print(resized_image)
            #rotate的话 所有维度必须在构建时确定
            resized_image.set_shape(
                [self.resized_size, self.resized_size, 3])
            resized_image=tf.contrib.image.rotate(resized_image,radian,'BILINEAR')
            gt_heatmaps=tf.contrib.image.rotate(gt_heatmaps,radian,'BILINEAR')
            return resized_image, gt_heatmaps,coor_yx_hp,reshaped_label,shape
        else:  
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
            gt_heatmaps = make_gaussian_map(
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
            "cat": tf.FixedLenFeature((), tf.string),
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
        reshaped_image = tf.reshape(tf.cast(image, tf.float32), shape)/255.0-0.5
        resized_size=tf.stack([self.resized_size,self.resized_size])
        old_size=tf.stack([height,width])
        resized_image=tf.image.resize_bilinear(tf.expand_dims(reshaped_image,0),resized_size)
        resized_image=tf.squeeze(resized_image)
        image_name = parsed_features['image_name']
        cat=parsed_features['cat']
        return resized_image, image_name,old_size,cat

    #使用新的tf.Data API来生成batch
    def dataset_input_new(self, tf_file):
        dataset = tf.data.TFRecordDataset(tf_file)
        if train_para['is_train']:
            dataset = dataset.map(self._parse_function)
            dataset = dataset.shuffle(10000)
            dataset = dataset.repeat(train_para['epoch'])
        else:
            dataset = dataset.map(self._parse_function2)
            #dataset = dataset.repeat(10)
        dataset = dataset.batch(self.batch_size)

        #迭代生成批次数据
        iterator = dataset.make_one_shot_iterator()
        if train_para['is_train']:
            batched_images, batched_labels,coor_yx,rl,shape = iterator.get_next()
            #生成的batch中各个的维度必须相同c
            return batched_images, batched_labels,coor_yx,rl,shape
        else:
            image,name,shape,cat=iterator.get_next()
            return image,name,shape,cat


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
#train_para['is_train'] = False
'''
with tf.Session() as sess:
    #print(len(categories_dict['blouse']))
    gen = clothes_batch_generater('skirt', 2)
    if not os.path.exists(dir_para['tf_dir']):
        os.mkdir(dir_para['tf_dir'])
        print('Created tfrecords dir')
        sys.stdout.flush()
    rec_name = 'skirt.tfrec'
    for (root, dirs, files) in os.walk(dir_para['tf_dir']):
        if rec_name not in files:
            gen.convert_to_tfrecords_single(
                'skirt',
                'C:\\Users\\oldhen\\Downloads\\tianchi\\fashionAI_key_points_test_a_20180227\\test',
                'D:\\tfrecords_test')
    batch = gen.dataset_input_new(dir_para['tf_dir'] + '\\' + rec_name)
    sess.run(tf.global_variables_initializer())
    img, lb ,coor,coor2,shape= sess.run(batch)
    #print(name[0].decode())
    #print(img)
    print(shape)
    scipy.misc.imsave('test1.png', img[0])
    scipy.misc.imsave('test2.png', img[1])
    lb_resized=tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(lb[0],0),[256,256],align_corners=True))
    lb_resized=sess.run(lb_resized)
    #lb_resized=lb[0]
    for i in range(lb_resized.shape[-1]):
        scipy.misc.imsave('hp%s.png'%i, lb_resized[:,:,i])
        #print(lb_resized[:,1,i])
        print('%s'%i,np.max(lb_resized[:,:,i]))
#pre=np.asarray(Image.open('prexx0.png','r'),np.uint8)
#最大的值
#max_val=np.max(pre)
#max_val=tf.reduce_max(max_val)
#print(sess.run(max_val))
#print(max_val)
#print(pre)
    #最大值的位置
    max_idx_list=[]
    true_list=[]
    true_list2=[]
    for i in range(lb_resized.shape[-1]-1):
        ind=np.unravel_index(np.argmax(lb_resized[:,:,i]),lb_resized.shape[:2])
        max_idx_list.append(ind)
        true_list.append(coor[0,i,:])
        true_list2.append(coor2[0,i,:])
    print(max_idx_list)
    print(true_list)
    print(true_list2)
'''