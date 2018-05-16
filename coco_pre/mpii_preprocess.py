import numpy as np
import cv2
import h5py as h5
#用于测试时的可视化
import imgaug as ia
from imgaug import augmenters as iaa
from utils_new import *
from data_preprocess import DataPreProcess


#------------------------------------------mpii数据预处理类-----------------------------------------------------------
class MPIIPreProcess(DataPreProcess):
    def __init__(self,
                 h5_dir,
                 img_dir,
                 is_train=True,
                 batch_size=12,
                 max_epoch=200,
                 filter_num=8):
        '''
        description:用于mpii数据集的预处理,包括图片、关节点信息读入及augment,用作关键点检测部分.
        parameters:
            h5_dir: str
                h5文件路径 
            img_dir: str
                图片路径
            is_train: bool
                训练/测试
            batch_size: int
                ...
            max_epoch: int 
                最大迭代轮数 对于test来说该参数无效,内部会置为1
            filter_num: int
                过滤掉小于多少个关键点被标注的人
        '''
        super(MPIIPreProcess, self).__init__(img_dir, is_train, batch_size,
                                             max_epoch)
        self.filter_num = filter_num
        #当前的轮数
        self.cur_epoch = 0
        f = h5.File(h5_dir, 'r')
        self.id = f['index'][:]
        self.img_name = f['img_name'][:]
        #需要先decode  因为存的时候先进行了encode
        '''
        for name in self.img_name:
            name = name.decode()
        '''
        self.img_name = np.asarray([name.decode() for name in self.img_name])
        self.bbox = f['bbox'][:]
        if self.is_train:
            self.max_epoch = max_epoch
            #用于随机产生序列 为了使每个进程得到不同的perm 先置为none
            self.perm = None
            self.vis = f['visible'][:]
            self.coords_num = f['kpt_num'][:]
            self.coords = f['coords'][:]
        else:
            self.max_epoch = 1
            self.perm = np.arange(self.img_name.shape[0])

        self.data_size = self.img_name.shape[0]
        print('Total person count:', self.img_name.shape[0])
        f.close()

    def get_batch(self, resized_size=(192, 256)):
        '''
        description:取得一个预处理好的batch
            训练/验证集流程:
            1.根据img_name读取图片;
            2.根据bbox抠出人;
            3.将人resize到统一大小并进行坐标转换;
            4.读入并更新人的coords坐标,具体来说就是减掉对应bbox左上角的坐标;
            5.进行data augment;
            6.进行对应的坐标转换;
            7.生成batch.
            测试集流程：
            1.load img_info;
            2.get img_batch;
        parameters:
            resized_size: tuple (width,height)
                统一resize的图片大小
        return: list
            一个batch
        '''
        index = self._update_cur_index()
        #1 2
        imgs = self._load_imgs(index)
        #3
        imgs = np.asarray([cv2.resize(img, resized_size) for img in imgs])
        if self.is_train:
            #4
            coords, vis = self._load_coordsvis_info(index)
            #[b,16,2]-[b,1,2]
            coords -= np.expand_dims(self.bbox[index][:, 0:2], 1)
            #[b,1,2]
            old_size = np.expand_dims(
                np.stack([
                    self.bbox[index][:, 2] - self.bbox[index][:, 0],
                    self.bbox[index][:, 3] - self.bbox[index][:, 1]
                ], 1), 1)
            coords = scale_coords_trans(
                coords, old_size, np.stack([resized_size[0], resized_size[1]]))
            #5
            imgs_aug, coords = augment_both(imgs, coords)
            #测试用
            coords_img_batch = [
                ia.KeypointsOnImage.from_coords_array(coords[i],
                                                      imgs_aug[i].shape)
                for i in range(self.batch_size)
            ]
            coords_vis = np.concatenate((coords, vis), -1)
            #找到未标注的点
            indices = np.where(coords_vis[:, -1] == 0)[0]
            #清零未标注的点的坐标
            coords_vis[indices] = 0

            return imgs_aug, coords_vis, coords_img_batch
        else:
            return imgs

    def _load_imgs(self, index):
        '''
        description: 载入一个batch的图片并根据bbox将对应的人抠出
        parameters:
            anns: list of int 
                一个或一个batch的图片id
            index:ndarray
                当前的索引 
        return: ndarray [b,h,w,3]
            imgs
        '''
        bboxs = self.bbox[index]
        bboxs = bboxs.astype(np.int)
        imgs = [
            cv2.imdecode(
                np.fromfile(self.img_dir + name, dtype=np.uint8),
                cv2.IMREAD_COLOR)[..., ::-1] for name in self.img_name[index]
        ]
        for i in range(self.batch_size):
            imgs[i] = imgs[i][bboxs[i, 1]:bboxs[i, 3], bboxs[i, 0]:bboxs[i, 2]]
            if imgs[i].shape[0] <= 0 or imgs[i].shape[1] <= 0:
                print('id:', self.id[index][i], 'name:',
                      self.img_name[index][i])
                print('x1 y1 x2 y2:', bboxs[i])
                raise ValueError('Image shape error.')
        return imgs

    def _load_coordsvis_info(self, index):
        '''
        description:获取关键点位置及是否可见的信息
        parameters:
            index:ndarray
                当前的索引 
        return: tuple of ndarrays
            处理好的coords、vis信息
        '''
        coords = self.coords[index]
        vis = np.expand_dims(self.vis[index], -1)

        return coords, vis
