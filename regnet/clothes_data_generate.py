import tensorflow as tf
import numpy as np
import csv
from PIL import Image
import sys

def _int64_feature(val):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))


def _bytes_feature(val):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))


def _float_feature(val):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[val]))


'''
description:用以将标签中的字符串三元组坐标转换为np.ndarray并返回之
args:
    coor:读取csv文件中的关节点部分
'''


def coor_read(coor):
    ret = []
    for xyv in coor:
        #忽略没有的关节点
        if xyv[0] == '-':
            continue
        xyv = np.asarray([int(i) for i in xyv.split('_')], np.int)
        ret.append(xyv)
    #kx3
    return np.asarray(ret, np.int)


'''
description:用以生成对应category的图片与标签，返回两个list,分别为images和labels
args:
    path:训练的目录
    category:服饰的类别，为了节省内存，还是一个个来把。。。
'''
def data_generate_single(
        category,
        path='C:\\Users\\oldhen\\Downloads\\tianchi\\fashionAI_key_points_train_20180227\\train'
):
    csv_file = csv.reader(open(path + '\\Annotations\\train.csv', 'r'))

    images=[]
    labels=[]
    for i, row in enumerate(csv_file):
        #先把表头读出来
        if i == 0:
            header = row
        else:
            if category!=row[1]:
                continue
            if i%100==0:
                print(i)
                sys.stdout.flush()

            img_name = path + '\\' + row[0]
            coor = coor_read(row[2:])
            images.append(np.asarray(Image.open(img_name),np.float32))
            labels.append(coor)
    return images,labels

'''
description:用以生成对应的图片与标签，返回两个dict，每个dict内部分别为5种类别的list，两个dict中的image和label一一对应
args:
    path:训练的目录
'''
def data_generate(
        path='C:\\Users\\oldhen\\Downloads\\tianchi\\fashionAI_key_points_train_20180227\\train'
):
    csv_file = csv.reader(open(path + '\\Annotations\\train.csv', 'r'))
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
            #print(category)
            #if i%100==0:
            #    print(i)
            #    sys.stdout.flush()

            img_name = path + '\\' + row[0]
            coor = coor_read(row[2:])
            img=Image.open(img_name)
            #在对应的类别下添加
            clothes_labels[category].append(coor)
            clothes_images[category].append(
                np.asarray(img, np.float32))
            return clothes_images, clothes_labels


'''
description:用以将images和labels转换为tensorflow的专用二进制格式便于后续读取
args:
    images:包含单个类别的images
    labels_dict:包含单个类别的labels
    category:类别名
    file_path:生成的文件存放的路径
'''
def convert_to_tfrecords_single(
        images,
        labels,
        category,
        file_path='D:\\tfrecords\\'):
    print("Start coverting to tfrecord.")
    sys.stdout.flush()
    if category==None:
        category='tmp'
    file_name = file_path + '\\' + category + '.tfrec'
    num = len(images)
    #height = images[0].shape[0]
    #width = images[0].shape[1]
    with tf.python_io.TFRecordWriter(file_name) as writer:
        #2 序列化并填充至example  然后写入磁盘
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


'''
description:用以将images和labels转换为tensorflow的专用二进制格式便于后续读取
args:
    images_dict:包含各个类别的images的字典
    labels_dict:包含各个类别的labels的字典
    file_path:生成的文件存放的路径
'''
def convert_to_tfrecords(images_dict,
                         labels_dict,
                         file_path='C:\\Users\\oldhen\\Downloads\\tianchi'):
    print("Start coverting to tfrecord.")
    for key in labels_dict.keys():
        images = images_dict[key]
        labels = labels_dict[key]
        file_name = file_path + '\\' + key + '.tfrec'
        num = len(images)
        height = images[0].shape[0]
        width = images[0].shape[1]
        with tf.python_io.TFRecordWriter(file_name) as writer:
            #2 序列化并填充至example  然后写入磁盘
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


categories=['blouse','skirt','outwear','trousers','dress']
for cat in categories:
    result = data_generate_single(cat)
    convert_to_tfrecords_single(result[0], result[1],cat)
    #for gc
    result=[]
