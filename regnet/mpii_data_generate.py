#mpii数据集的处理以生成tfrecords
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from six.moves import xrange
from PIL import Image
#读取.mat文件
import scipy.io as sio
import numpy as np
#from sklearn.decomposition import PCA
from scipy import stats, ndimage

HEIGHT = 720
WIDTH = 1280
NUM_IMAGES_FOR_TRAIN = 22053  #330093
NUM_IMAGES_FOR_EVAL = 894
NUM_CLASSES = 15

FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string('record_dir','/home/b3336/joints/ICVL/TFRecord/',"Path to store tfrecord.")
#tf.app.flags.DEFINE_string('dataset_dir','/home/b3336/joints/ICVL/',"Path to store dataset.")
#tf.app.flags.DEFINE_boolean('use_pca',True,"""Whether to use pca.""")


def _int64_feature(val):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))


def _bytes_feature(val):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))


def _float_feature(val):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[val]))


def label_and_dpt(dpt_path, path=FLAGS.dataset_dir, file_name='all.txt'):
    label_path = os.path.join(path, file_name)
    if not os.path.exists(lablelpath):
        raise ValueError("Please give a correct label path.")
    print("Reading from files,please wait.")
    with open(labelpath) as f:
        for line in f:
            results = f.split(' ')
            #取得全路径
            file_path = os.path.join(path, '/', results[0])
            if cmp(label_path, file_path) == 0:
                ground_truth = np.asarry(results[i] for i in range(1, 49))
                com = np.asarray([float(results[i]) for i in range(1, 4)])
                com[0] = (com[0] - UX) * com[2] / FX
                com[1] = (com[1] - UY) * com[2] / FY
                dpt = np.asarray(Image.open(dpt_path), np.float32)
                dealed_dpt = _deal_with_dpt(com, dpt)
                print('ground truth:', ground_truth)
                break
            else:
                continue
    return dealed_dpt,


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def annolist_deal(annolist):
    annorect=annolist['annorect']

    if annorect.shape[0] != 0:

    else:
        return np.array([])


#生成对应的图像和标签
def data_generate(image_path='D:\迅雷下载\mpii_human_pose_v1\images',label_path='D:\迅雷下载\mpii_human_pose_v1_u12_2\mpii_human_pose_v1_u12_1.mat'):
    data = sio.loadmat(file_name)['RELEASE']
    #mat文件中就是多包了这两层
    is_train = data['img_train'][0,0]
    anno_list = data['annolist'][0,0]
    #print(is_train.shape)
    #拿来训练的(人有关节点标注)
    train=[i for i in range(is_train.shape[1]) if is_train[0,i]==1]
    #拿来测试的(人无关节点标注)
    test=[i for i in range(is_train.shape[1]) if not is_train[0,i]]
    image_train=[]
    image_test=[]
    label_train=[]
    deprecated=[]
    #训练
    for i in train:
        #label = annolist_deal(anno_list[0,i])
        annorect=anno_list[0,i]['annorect']
        name=image_path+anno_list[0,i]['image']['name'][0,0][0]
        #annorect域缺失
        if annorect.shape[1]==0:
            deprecated.append(i)
            continue
        for j in range(annorect.shape[1]):
            #annorect为1x1但内部为空
            #or 不存在annopoints域
            #or annopoints域为空
            if annorect[0,j] == None or 
               'annopoints' not in list(annorect[0,j].dtype.fields.keys()) or 
                annorect[0,j]['annopoints'].shape[1]==0:
                deprecated.append(i)
                #只要存在为空或#前面的不存在annopoints域后面的肯定也不存在
                #只要图片中存在有人未被标注即舍弃之
                break
            #通过检测的图片 可能会加入多次 因为有些图中存在着多个人
            image_train.append(np.asarray(Image.open(name),np.float32))



    #移去残缺的标注的图 
    for val in deprecated:
        train.remove(val)



def convert_to_tfrecords(images, labels, num, tffile_path):
    assert num == images.shape[0]
    assert num == labels.shape[0]
    height = images.shape[1]
    width = images.shape[2]
    print('before convert:')
    print(labels.shape)
    # print (height)
    #depth=images.shape[3]
    print("Converting to tfrecord.")
    with tf.python_io.TFRecordWriter(tffile_path) as writer:
        #2 序列化并填充至example  然后写入磁盘
        #方向
        #labels.shape 返回元祖  在len以取得维数
        if len(labels.shape) == 1:
            for i in xrange(num):
                image_raw = images[i].tostring()
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'height': _int64_feature(height),
                            'width': _int64_feature(width),
                            'label': _int64_feature(int(labels[i])),
                            'image_raw': _bytes_feature(image_raw)
                        }))
                writer.write(example.SerializeToString())
    #姿势
        elif len(labels.shape) > 1:
            for i in xrange(num):
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
    print("Covert done.")


def _read_from_tfrecords(file_queue):
    reader = tf.TFRecordReader()
    key, val = reader.read(file_queue)
    #返回的feature是一个字典
    if not FLAGS.direction:
        feature = tf.parse_single_example(
            val,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label_raw': tf.FixedLenFeature([], tf.string)
            })
        image = tf.decode_raw(feature['image_raw'], tf.float32)
        reshaped_image = tf.reshape(image, [SIZE, SIZE, CHANNELS])
        label = tf.decode_raw(feature['label_raw'], tf.float32)
        if FLAGS.train and FLAGS.use_pca:
            reshaped_label = tf.reshape(label, [30])
        else:
            reshaped_label = tf.reshape(label, [48])
    else:
        feature = tf.parse_single_example(
            val,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            })
        image = tf.decode_raw(feature['image_raw'], tf.float32)
        reshaped_image = tf.reshape(image, [SIZE, SIZE, CHANNELS])
        label = tf.cast(feature['label'], tf.int64)
        reshaped_label = tf.reshape(label, [1])

    return reshaped_image, reshaped_label


def input(batch_size, file_path):
    print(file_path)
    if not os.path.exists(file_path):
        raise ValueError("Please give an correct path.")
#生成输入文件队列   注意参数应该是string tensor  所以加个[]
    if FLAGS.train:
        file_queue = tf.train.string_input_producer([file_path])
    else:
        file_queue = tf.train.string_input_producer([file_path], 2)

        #测试
#从tfrecords文件中读取并decode
# print (file_queue)
    read_result = _read_from_tfrecords(file_queue)
    #print (len(read_result))
    #print(NUM_IMAGES_FOR_TRAIN)
    if FLAGS.train:
        min_num = int(NUM_IMAGES_FOR_TRAIN * 0.5)
    else:
        min_num = int(NUM_IMAGES_FOR_EVAL * 0.36)
#返回
    if FLAGS.train:
        images_batch, labels_batch = tf.train.shuffle_batch(
            [read_result[0], read_result[1]],
            batch_size=batch_size,
            num_threads=16,
            min_after_dequeue=min_num,
            capacity=min_num + batch_size * 10)
    else:
        images_batch, labels_batch = tf.train.batch(
            [read_result[0], read_result[1]],
            batch_size=batch_size,
            num_threads=8,
            capacity=10000)
    tf.summary.image('images', images_batch)
    #return images_batch,tf.reshape(labels_batch,[batch_size])
    return images_batch, labels_batch


#if __name__=='__main__':
#    images,labels=deal_with_pca()
#    convert_to_tfrecords(images,labels,images.shape[0],'tf_pca.tf')
