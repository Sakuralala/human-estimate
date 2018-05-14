#使用np及cv2而不是tf来进行数据的预处理和生成
#encoding=utf-8
from pycocotools.coco import COCO
import numpy as np
import cv2
from parallel_gen import GeneratorEnqueuer
from coco_wrapper import *
from utils_new import *
import sys
import os
from ref import *
#用于测试时的可视化
import imgaug as ia
from imgaug import augmenters as iaa


#batch生成器
def batch(ann_dir,
          img_dir,
          batch_size=12,
          max_epoch=200,
          filter_num=8,
          slavers=10,
          max_queue_size=16):
    '''
    description：生成batch
    parameters:
        input_size: int 
            总的输入数量
        filter_num: int 
            用于除去数据集中标准点太少的人物,表示最少需要被标注的关节点数量
        slavers: int
            线程数量
        max_epoch: int
            最大迭代轮数
    return: generator
        返回从队列中取得预处理batch的generator
    '''
    try:
        cpp = COCOPreProcess(
            ann_dir, img_dir, batch_size, max_epoch, filter_num=filter_num)
        if os.name is 'posix':
            gen = cpp.get_batch_gen()
        #for fuck windows
        elif os.name is 'nt':
            gen = cpp
        else:
            raise ValueError('Unknown system.')
        gen_enq = GeneratorEnqueuer(gen)
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


#------------------------------------------COCO数据预处理类-----------------------------------------------------------
class COCOPreProcess():
    def __init__(self,
                 ann_dir,
                 img_dir,
                 batch_size=12,
                 max_epoch=200,
                 filter_num=8):
        '''
        description:用于coco数据集人的部分的预处理,包括图片、关节点信息读入及augment,用作关键点检测部分.
        parameters:
            ann_dir: str
                json文件路径 
            img_dir: str
                图片路径
            filter_num: int
                过滤掉小于多少个关键点被标注的人
        '''
        self.img_dir = img_dir
        self.cur_index = 0
        self.batch_size = batch_size
        self.filter_num = filter_num
        #当前的轮数
        self.cur_epoch = 0
        self.max_epoch = max_epoch
        self.coco = COCO(ann_dir)
        #因为person_kpt.json里面只含有person类的信息所以这步做不做都一样
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        self.ann_ids = self.coco.getAnnIds(catIds=self.cat_ids)
        self.ann_ids = np.asarray([
            ann_id for ann_id in self.ann_ids
            if self.coco.loadAnns(ann_id)[0]['num_keypoints'] > self.filter_num
        ])
        print('Total person count:', self.ann_ids.shape[0])
        #用于随机产生序列 为了使每个进程得到不同的perm 先置为none
        self.perm = None

    #linux下调这个
    def get_batch_gen(self, resized_size=(256, 192)):
        '''
        description: 得到一个预处理好的数据batch的generator
        parameters:
            batch_size: int
                batch大小
            max_epoch: int 
                最大迭代轮数
            resized_size: tuple  
                统一resize的宽和高(先width后height)
        return: generator
            batch generator
        '''
        try:
            while True:
                img_batch, kpt_batch, kpt_img_batch = self.get_batch(
                    resized_size)
                yield img_batch, kpt_batch, kpt_img_batch
        except Exception as e:
            raise e

    #windows下调这个
    def next(self, resized_size=(256, 192)):
        '''
        description:for fuck windows
        parameters:
            resized_size: tuple
                统一resize到的图片大小
        return: list
            一个预处理好的数据batch
        '''
        return self.get_batch(resized_size)

    def get_batch(self, resized_size=(192, 256)):
        '''
        description:取得一个预处理好的batch
            流程:
            1.load info;
            2.根据ann_info中的image_id找到对应的img_info并读取图片;
            3.根据ann_info读取bbox;
            4.抠出人;
            5.读入并更新人的kpt坐标,具体来说就是减掉对应bbox左上角的坐标;
            6.将人resize到统一大小并进行坐标转换;
            7.进行data augment;
            8.进行对应的坐标转换;
            9.生成batch.
        parameters:
            resized_size: tuple (width,height)
                统一resize的图片大小
        return: list
            一个batch
        '''
        self._check_cur_index()
        #1
        ann_info = self._load_ann_info()
        img_batch = []
        kpt_batch = []
        kpt_img_batch = []
        for ann in ann_info:
            #2 3 4
            bbox = ann['bbox']
            img = self._load_imgs(ann['image_id'], bbox)
            #5
            #17为关键点数量,3为(x,y,vis),其中vis=0表示未标注,vis=1表示标注了但不可见,vis=2表示标注了且可见
            kpt_vis = np.asarray(ann['keypoints']).reshape((17, 3))
            kpt = kpt_vis[:, 0:2].astype(np.float)
            vis = np.expand_dims(kpt_vis[:, 2], -1)
            kpt -= np.asarray(bbox[0:2])
            #6
            img = cv2.resize(img, resized_size)
            kpt = scale_coords_trans(kpt, np.stack([bbox[2], bbox[3]]),
                                     np.stack(
                                         [resized_size[0], resized_size[1]]))
            #7、8 scale、flip、rotate、brightness、blur、hue等.
            img_aug, kpt = augment_both(img, kpt)
            #9
            img_batch.append(img_aug)
            #测试用
            kpt_img_batch.append(
                ia.KeypointsOnImage.from_coords_array(kpt, img_aug.shape))
            kpt_vis = np.concatenate((kpt, vis), -1)
            #找到未标注的点
            indices = np.where(kpt_vis[:, -1] == 0)[0]
            #清零未标注的点的坐标
            kpt_vis[indices] = 0
            kpt_batch.append(kpt_vis)

        return img_batch, kpt_batch, kpt_img_batch

    def _check_cur_index(self):
        '''
        description:检测当前的索引及epoch是否需要重置(一个epoch完了的话)并在到达最大epoch后raise
        parameters:
            none
        return:
            none
        '''
        if self.perm is None:
            self.perm = np.random.permutation(self.ann_ids.shape[0])
        if self.cur_index + self.batch_size > self.ann_ids.shape[0]:
            self.perm = np.random.permutation(self.ann_ids.shape[0])
            self.cur_index = 0
            self.cur_epoch += 1
            if self.cur_epoch == self.max_epoch:
                raise ValueError('Train/Test finished.')

    def _load_ann_info(self):
        '''
        description:用于从json文件中载入一个batch的ann_info,由于一张图里可能有多个人,为了保证batch的一致性还是使用ann_info.
        '''
        #id的索引
        ann_ids_index = self.perm[self.cur_index:
                                  self.cur_index + self.batch_size]
        self.cur_index += self.batch_size
        #注意一个image_id可以对应多个ann的id
        ann_ids = self.ann_ids[ann_ids_index]
        #print(ann_ids_index, ann_ids)
        ann_info = self.coco.loadAnns(ann_ids)

        return ann_info

    def _load_imgs(self, img_ids, bboxs):
        '''
        description: 根据提供的img_ids找到对应的图片并根据bbox将对应的人抠出
        parameters:
            anns:int or list of int 
                一个或一个batch的图片id
            bbox:list or list of list
                一个或一组bbox,注意bbox的顺序要与img_ids一一对应
        return: ndarray  [h,w,3] or [b,h,w,3]
            imgs
        '''
        img_ids = img_ids if isArrayLike(img_ids) else [img_ids]
        bboxs = bboxs if isArrayLike(bboxs[0]) else [bboxs]
        assert len(img_ids) == len(bboxs)

        #注意此处返回的是list
        img_infos = self.coco.loadImgs(img_ids)

        imgs = []
        #5
        for i, bbox in enumerate(bboxs):
            #避免py3中cv2.imread不能读取中文路径的问题
            img = cv2.imdecode(
                np.fromfile(
                    self.img_dir + img_infos[i]['file_name'], dtype=np.uint8),
                cv2.IMREAD_COLOR)[..., ::-1]
            y1 = int(bbox[1])
            y2 = int(bbox[1] + bbox[3])
            x1 = int(bbox[0])
            x2 = int(bbox[0] + bbox[2])
            imgs.append(img[y1:y2, x1:x2, :])

        return imgs if len(imgs) > 1 else imgs[0]
