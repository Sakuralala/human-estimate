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
            sigma=1.0,
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
	        coor_noise:是否在关节点的位置上加上一定噪声
	        crop_center_noise:是否在crop图像的中心加上一定噪声
        '''
        self.sigma = sigma
        self.resized_size = np.stack([resized_height, resized_width])

    def generate_batch(self,
                       category,
                       file_list,
                       slavers=10,
                       max_queue_size=16,
                       batch_size=8):
        '''
        desc：生成batch
        args:
            input_size:总的输入数量
            slavers:线程数量
            batch_size:。。。
        '''
        try:
            di = DataIterator(category, file_list, batch_size=batch_size)
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
        finally:
            pass
            #if gen_enq is not None:
            #gen_enq.stop()


#数据的预处理及batch生成
class DataIterator():
    def __init__(self, category, file_list, batch_size=12, coor_noise=True):
        '''
        desc:因为win下generator不能被pickle所以只能改成使用iterator了
        args:
            file_list:从原始csv文件中读取的信息
        '''
        #self.input_size=input_size
        self.batch_size = batch_size
        self.resized_size = np.stack(
            [input_para['resized_height'], input_para['resized_width']])
        #除去第一行
        self.file_list = file_list
        self.input_size = len(self.file_list)
        #为了让每个子进程得到不同的perm顺序，先把cur_index置为最大值好在next更新perm
        self.cur_index = self.input_size
        self.perm = None
        self.category = category
        self.coor_noise = coor_noise

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
                #if all_kpt_list[i] in categories_dict[cat]:
                #对于相同类别下可能不存在的关节点直接置全零
                ret.append(np.asarray([0, 0, 0], np.int32))
            else:
                #[3]
                xyv = np.asarray([int(i) for i in xyv.split('_')], np.int)
                ret.append(xyv)
        #[k,3]
        return np.asarray(ret)

    def next(self):
        '''
        desc:处理好image和label(train)并返回
        '''
        image_batch = []
        if train_para['is_train']:
            label_batch = []
        else:
            img_name_batch = []
            img_cat_batch = []
            old_size_batch = []
        if self.cur_index + self.batch_size > self.input_size:
            self.cur_index = 0
            if train_para['is_train']:
                self.perm = np.random.permutation(self.input_size)
            else:  #测试时不需要随机打乱
                self.perm = np.arange(self.input_size)
        #一个epoch中
        index = self.perm[self.cur_index:self.cur_index + self.batch_size]

        for i in range(self.batch_size):
            #每一行
            row = self.file_list[index[i]].split(',')
            img_name = dir_para['train_data_dir'] if train_para[
                'is_train'] else dir_para['test_data_dir'] + '/' + row[0]
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
                #[k,3]
                coor_orig = self._coor_read(self.category, row[2:])
                #[k,3]
                coor_yx = np.stack([coor_orig[:, 1], coor_orig[:, 0]], 1)
                coor = coor_translate_np(coor_yx, old_size, self.resized_size)
                coor_xy = np.stack([coor[:, 1], coor[:, 0]], 1)
                #随机旋转-30到30度
                angle = np.random.uniform(-30.0, 30.0)
                scale = np.random.uniform(0.75, 1.25)
                #[size,h,w,3]
                ret = rotate_and_scale(img, angle, scale=scale)
                img = augment(img)
                img = ret[1].astype(np.float) / 255.0
                image_batch.append(img)
                #仿射变换矩阵
                #[3,2]
                M = np.transpose(ret[0], (1, 0))
                # [24,3]的坐标修改(转换到齐次坐标系) [x,y,1]
                coor_3d = np.concatenate(
                    (coor_xy, np.expand_dims(np.ones(coor_orig.shape[0]), 1)),
                    1)
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
                #超出图片范围的
                coor_trans[out] = 0
                gt_heatmaps = make_gt_heatmaps(coor_trans[:, 0:2],
                                               self.resized_size, 1.0,
                                               coor_trans[:, 2])
                #缩小到和网络输出相同尺寸 先x后y
                gt_heatmaps = cv2.resize(
                    gt_heatmaps,
                    (self.resized_size[1] // 4, self.resized_size[0] // 4))
                label_batch.append(gt_heatmaps)

        if train_para['is_train']:
            batch = [np.asarray(image_batch), np.asarray(label_batch)]
        else:
            batch = [
                image_batch, old_size_batch, img_name_batch, img_cat_batch
            ]
        self.cur_index += self.batch_size

        return batch