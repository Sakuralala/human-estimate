#使用np及cv2而不是tf来进行数据的预处理和生成
import numpy as np
import csv
import cv2
from params_clothes import *
from parallel_gen import GeneratorEnqueuer
from utils_new import *
import sys


class clothes_batch_generater():
    def __init__(
            self,
            resized_height=256,
            resized_width=256,
            #is_resize=True,#必须要resize 因为无法保证每张图片的大小是一致的
    ):
        '''
        desc:输入数据batch生成器
        args:
            resized_height:裁剪后的图像高
            resized_width:裁剪后的图像宽
            sigma:二维高斯核的方差
        '''
        self.resized_size = np.stack([resized_height, resized_width])

    def generate_batch(self,
                       category,
                       file_list,
                       slavers=10,
                       max_queue_size=16,
                       sigma=1.0,
                       batch_size=12,
                       max_epoch=200):
        '''
        desc：生成batch
        args:
            input_size:总的输入数量
            slavers:线程数量
            batch_size:。。。
            sigma:高斯分布的标准差
        '''
        try:
            di = DataIterator(
                category,
                file_list,
                batch_size=batch_size,
                max_epoch=max_epoch)
            gen_enq = GeneratorEnqueuer(di)
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


#数据的预处理及batch生成
class DataIterator():
    def __init__(self,
                 category,
                 file_list,
                 batch_size=12,
                 sigma=1.0,
                 max_epoch=200,
                 is_valid=False):
        '''
        desc:因为win下generator不能被pickle所以只能改成使用iterator了
        args:
            file_list:从csv文件中读取的信息
            is_valid:是否为验证集
        '''
        #self.input_size=input_size
        self.batch_size = batch_size
        self.resized_size = np.stack(
            [input_para['resized_height'], input_para['resized_width']])
        self.file_list = file_list
        self.input_size = len(self.file_list)
        #为了让每个子进程得到不同的perm顺序，先把cur_index置为最大值好在next更新perm
        self.cur_index = self.input_size
        self.perm = None
        self.category = category
        self.is_start = True
        self.is_valid = is_valid
        self.epoch = -1
        self.max_epoch = max_epoch

    def _coor_read(self, cat, coor):
        '''
        description:用以将标签中的字符串三元组坐标转换为np.ndarray并返回之
        args:
            cat:类别
            coor:读取csv文件中的关节点部分
        '''
        ret = []
        cate_index = []
        for i, xyv in enumerate(coor):
            #忽略对应类别没有的关节点
            if xyv[0] == '-':
                if all_kpt_list[i] in categories_dict[cat]:
                    #对于相同类别下可能不存在的关节点直接置全零
                    ret.append(np.asarray([0, 0, 0], np.int32))
                    cate_index.append(0)
            else:
                #[3]
                xyv = np.asarray([int(i) for i in xyv.split('_')], np.int)
                ret.append(xyv)
                cate_index.append(1)
        #[k,3]
        return np.asarray(ret), np.asarray(cate_index)

    def next(self):
        '''
        desc:处理好image和label(train)并返回
        '''
        image_batch = []
        cate_index_batch = []
        if train_para['is_train']:
            label_batch = []
        else:
            img_name_batch = []
            img_cat_batch = []
            old_size_batch = []
        if self.cur_index + self.batch_size > self.input_size:
            if self.is_start == True:
                self.is_start = False
            else:
                #测试时跑完了所有图片
                if train_para['is_train'] == False:
                    raise ValueError('Test finished.')
                else:
                    if self.epoch > self.max_epoch:
                        raise ValueError('Train finished.')

            self.epoch += 1
            self.cur_index = 0
            if train_para['is_train']:
                self.perm = np.random.permutation(self.input_size)
            else:  #测试时不需要随机打乱
                self.perm = np.arange(self.input_size)
        #一个epoch中取的索引
        index = self.perm[self.cur_index:self.cur_index + self.batch_size]

        for i in range(self.batch_size):
            #-------------------------图片读取及原始尺寸保存及统一resize到同一尺寸---------------------------------#
            row = self.file_list[index[i]].split(',')
            #cate_index_batch.append(index_dict[row[1]])
            if train_para['is_train']:
                if self.is_valid == False:
                    img_name = dir_para['train_data_dir'] + '/' + row[0]
                else:
                    img_name = dir_para['val_data_dir'] + '/' + row[0]
            else:
                img_name = dir_para['test_data_dir'] + '/' + row[0]
            img = cv2.imread(img_name)
            #转换为rgb
            img = img[..., ::-1].astype(np.uint8)
            old_size = np.stack([img.shape[0], img.shape[1]])
            #dsize默认为先x后y
            img = cv2.resize(img, (self.resized_size[1], self.resized_size[0]))
            if train_para['is_train'] == False:
                img_name_batch.append(row[0])
                img_cat_batch.append(row[1])
                image_batch.append(img)
                old_size_batch.append(old_size)
            else:
                #resize后坐标的转换
                coor_orig, cate_index = self._coor_read(self.category, row[2:])
                cate_index_batch.append(cate_index)
                coor_yx = np.stack([coor_orig[:, 1], coor_orig[:, 0]], 1)
                coor = coor_translate_np(coor_yx, old_size, self.resized_size)
                coor_xy = np.stack([coor[:, 1], coor[:, 0]], 1)
                if self.is_valid == False:
                    #-------------------------图片的处理-----------------------------------------------------------#
                    #随机旋转-30到30度
                    angle = np.random.uniform(-30.0, 30.0)
                    scale = np.random.uniform(0.75, 1.25)
                    #[size,h,w,3]
                    ret = rotate_and_scale(img, angle, scale=scale)
                    #对图片的hue、brightness做一些增强处理
                    img = augment(ret[1])
                    img = img.astype(np.float) / 255.0
                    image_batch.append(img)
                    #-------------------------坐标的处理-----------------------------------------------------------#
                    #仿射变换矩阵 [3,2]
                    M = np.transpose(ret[0], (1, 0))
                    # [24,3]的坐标修改(转换到齐次坐标系) [x,y,1]
                    coor_3d = np.concatenate(
                        (coor_xy, np.expand_dims(
                            np.ones(coor_orig.shape[0]), 1)), 1)
                    coor_3d = coor_3d.astype(np.float)
                    #[24,2]
                    coor_trans = np.matmul(coor_3d, M)
                    coor_trans = np.floor(coor_trans).astype(np.int)
                    #取第0维 即超出图片尺寸范围的索引
                    #np.where(condition),返回的是一个tuple,其中tuple的长度为输入的array的维数，
                    #tuple中的每个elem为一个一维array，其中array中的每个元素代表对应维的索引，结
                    #合各个elem中相同位置的元素则可得到一个完整的索引
                    #此处返回的是超出图片尺寸的第0维的索引
                    out = np.concatenate((np.where(coor_trans > 255)[0],
                                          np.where(coor_trans < 0)[0]))
                    #unique
                    out = np.unique(out)
                    #print(out)
                    coor_trans = np.concatenate(
                        (coor_trans, np.expand_dims(coor_orig[:, 2], 1)), 1)
                    #超出图片范围的全置为0
                    coor_trans[out] = 0
                    if train_para['using_cg'] == False:
                        gt_heatmaps = make_gt_heatmaps(coor_trans[:, 0:2],
                                                       self.resized_size, 1.0,
                                                       coor_trans[:, 2])
                    else:  #将问题看作分类+回归问题
                        gt_heatmaps = make_probmaps_and_offset(
                            coor_trans, self.resized_size,
                            train_para['radius'], coor_trans[:, 2])
                else:  #验证集不需要做数据增强
                    image_batch.append(img)
                    if train_para['using_cg'] == False:
                        gt_heatmaps = make_gt_heatmaps(
                            coor_xy, self.resized_size, 1.0, coor_orig[:, 2])
                if train_para['using_cg'] == False:
                    #缩小到和网络输出相同尺寸 先x后y
                    gt_heatmaps = cv2.resize(
                        gt_heatmaps,
                        (self.resized_size[1] // 4, self.resized_size[0] // 4))
                label_batch.append(gt_heatmaps)

        if train_para['is_train']:
            cate_index_batch = np.reshape(
                cate_index_batch,
                (train_para['batch_size'], 1, 1, len(cate_index_batch[0])))
            batch = [
                np.asarray(image_batch),
                np.asarray(label_batch), cate_index_batch
            ]
        else:
            batch = [
                image_batch, old_size_batch, img_name_batch, img_cat_batch
            ]
        self.cur_index += self.batch_size

        return batch