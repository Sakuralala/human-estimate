#这个根据实际情况改
from pycocotools.coco import COCO
import numpy as np
import cv2
import itertools
#用于测试时的可视化
import imgaug as ia
from imgaug import augmenters as iaa
from utils.utils import *
from data_loader.data_preprocess import DataPreProcess


#------------------------------------------COCO数据预处理类-----------------------------------------------------------
class COCOPreProcess(DataPreProcess):
    def __init__(self,
                 img_dir,
                 json_dir,
                 resized_size=(192, 256),
                 is_train=True,
                 is_multi=False,
                 batch_size=12,
                 max_epoch=200,
                 filter_num=8):
        '''
        description:用于coco数据集人的部分的预处理,包括图片、关节点信息读入及augment,用作关键点检测部分.
        parameters:
            json_dir: str
                json文件路径,对于测试文件来说，没有annotation这一项
            img_dir: str
                图片路径
            resized_size: tuple
                宽 高
            is_train: bool
                训练/测试
            is_multi: bool
                top-down or bottom up
            batch_size: int
                ...
            max_epoch: int 
                最大迭代轮数 对于test来说该参数无效
            filter_num: int
                过滤掉小于多少个关键点被标注的人,对test来说该参数无效
        '''
        super(COCOPreProcess, self).__init__(img_dir, resized_size, is_train,
                                             is_multi, batch_size, max_epoch)
        self.filter_num = filter_num
        self.coco = COCO(json_dir)
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        self.img_ids = np.asarray(self.coco.getImgIds(catIds=self.cat_ids))
        if self.is_train:
            self.ann_ids = self.coco.getAnnIds(catIds=self.cat_ids)
            self.ann_ids = np.asarray([
                ann_id for ann_id in self.ann_ids if self.coco.loadAnns(ann_id)
                [0]['num_keypoints'] > self.filter_num
            ])
            if self.is_multi == False:
                self.data_size = self.ann_ids.shape[0]
                print('Total person count:', self.ann_ids.shape[0])
            else:
                self.data_size = self.img_ids.shape[0]
                print('Total image count:', self.img_ids.shape[0])
        #训练bottom-up方法时及测试任意方法时data_size是根据图片数量来的
        else:
            self.data_size = self.img_ids.shape[0]
            print('Total image count:', self.img_ids.shape[0])
        #用于随机产生序列 为了使每个进程得到不同的perm 先置为none
        #对于test来说不需要打乱
        self.perm = None if self.is_train == True else np.arange(
            self.img_ids.shape[0])

    def get_batch(self):
        '''
        description:取得一个预处理好的batch
            训练/验证集流程:
            1.load info;
            2.根据ann_info中的image_id找到对应的img_info并读取图片;
            3.根据ann_info读取bbox;
            4.抠出人;
            5.读入并更新人的kpt坐标,具体来说就是减掉对应bbox左上角的坐标;
            6.将人resize到统一大小并进行坐标转换;
            7.进行data augment;
            8.进行对应的坐标转换;
            9.生成batch.
            测试集流程：
            1.load img_info;
            2.get img_batch;
        parameters:
        return: list
            一个batch
        '''
        index = self._update_cur_index()
        if self.is_train:
            kpt_img_batch = []
            img_batch, coords, vis = self._get_batch_single(
                index) if self.is_multi == False else self._get_batch_multi(
                    index)
            old_size = np.expand_dims(
                np.asarray(
                    [[img.shape[1], img.shape[0]] for img in img_batch]), 1)
            #6
            img_batch = np.asarray(
                [cv2.resize(img, self.resized_size) for img in img_batch])
            if self.is_multi == False:
                coords = scale_coords_trans(
                    coords, old_size,
                    np.stack([self.resized_size[0], self.resized_size[1]]))
            #因为bottom-up时一张图里的人的数量不一定相同，那么coords也不是一个各维度都相同的array
            else:
                resized_size = np.stack(
                    [self.resized_size[0], self.resized_size[1]])
                coords = np.asarray([
                    scale_coords_trans(coords[i], old_size[i], resized_size)
                    for i in range(len(coords))
                ])
            #7、8 scale、flip、rotate、brightness、blur、hue等.
            img_batch, coords_batch = augment_both(img_batch, coords)
            #测试用
            kpt_img_batch = [
                ia.KeypointsOnImage.from_coords_array(coords_batch[i],
                                                      img_batch[i].shape)
                for i in range(self.batch_size)
            ]
            if self.is_multi == False:
                kpt_batch = np.concatenate((coords_batch, vis), -1)
                #需要做这一步 因为坐标转换总共有三处，第一处为抠出人时的转换，这时候未标注的点都变为了负数坐标；第二处为scale变换时的坐标转换，这个时候基本上就是乘个系数，所以负数坐标还是负数，在scale_coords_trans最后会把超出图片范围的坐标重新变为0，所以未标注的坐标又变为了0；第三处为图像增强时做的坐标变换,此时未标注的00点就会变
                #找到未标注的点
                indices = np.where(kpt_batch[..., 2] == 0)
                #print(indices[0].shape)
                #清零未标注的点的坐标
                if len(indices)>0:
                    kpt_batch[indices[0], indices[1]] = 0
            else:
                kpt_batch_origin = [
                    np.concatenate((coords_batch[i], vis[i]), -1)
                    for i in range(self.batch_size)
                ]
                kpt_batch = []
                for kpt in kpt_batch_origin:
                    indices = np.where(kpt[:, 2] == 0)[0]
                    if len(indices)>0:
                        kpt[indices[0]] = 0
                    kpt_batch.append(kpt)

            return img_batch, kpt_batch, kpt_img_batch
        else:
            img_batch = self._load_imgs(self.img_ids[index])
            for img in img_batch:
                img = cv2.resize(img, self.resized_size)

            return img_batch

    def _get_batch_single(self, index):
        '''
        description:top-down类取得一个batch的原始抠出人的图片及转换后关键点信息
        parameters:
            index: ndarray
                索引
        return: tuple
            一个batch的图片信息和关键点信息
        '''
        #1
        ann_info = self._load_ann_info(index)
        #2 3 4 [b,4]
        bboxs = np.asarray([ann['bbox'] for ann in ann_info])
        img_ids = [ann['image_id'] for ann in ann_info]
        img_batch = self._load_imgs(img_ids, bboxs)
        #[b,51]
        kpt_info = [ann['keypoints'] for ann in ann_info]
        #5 [b,17,2] [b,17,1]
        coords, vis = self._load_kptvis_info(kpt_info)
        coords -= np.expand_dims(bboxs[:, 0:2], 1)

        return img_batch, coords, vis

    def _get_batch_multi(self, index):
        img_ids = self.img_ids[index].tolist()
        img_batch = self._load_imgs(img_ids)
        ann_info_list = self._load_ann_info_multi(img_ids)
        #[b,x,3] x不一定相同。
        coords_list = []
        vis_list = []
        for ann_info in ann_info_list:
            kpt_info = [ann['keypoints'] for ann in ann_info]
            #同张图里所有人的所有关键点放在一个list中
            kpt_info = list(itertools.chain.from_iterable(kpt_info))
            kpt_vis = self._load_kptvis_info(kpt_info)
            coords_list.append(kpt_vis[0])
            vis_list.append(kpt_vis[1])

        return img_batch, coords_list, vis_list

    def _load_ann_info(self, index):
        '''
        description:用于从train/val json文件中载入一个batch的ann_info,由于一张图里可能有多个人,为了保证batch的一致性还是使用ann_info.
        parameters: 
            index: ndarray
                一组索引
        return: list
            ann_info的list
        '''
        #注意一个image_id可以对应多个ann的id
        ann_ids = self.ann_ids[index]
        ann_info = self.coco.loadAnns(ann_ids)

        return ann_info

    def _load_ann_info_multi(self, img_ids):
        '''
        description:用于从train/val json文件中载入一个batch中的每张图的ann_info,由于一张图里可能有多个人,故返回的ann_info list中的每个子list的长度不一定相同.
        parameters: 
            img_ids: list
                一组索引
        return: list of list
            ann_info的list,每个元素为一组ann_info
        '''
        ann_ids_list = [self.coco.getAnnIds(img_id) for img_id in img_ids]
        ann_info_list = [
            self.coco.loadAnns(ann_ids) for ann_ids in ann_ids_list
        ]

        return ann_info_list

    def _load_imgs(self, img_ids, bboxs=None):
        '''
        description: 根据提供的img_ids找到对应的图片,若为训练单人的则根据bbox将对应的人抠出,否则直接返回对应的图片
        parameters:
            img_ids: int or list of int 
                一个或一个batch的图片id
            bbox:list or list of list or ndarray
                一个或一组bbox,注意bbox的顺序要与img_ids一一对应,若为None，表示为测试或多人
        return: ndarray  [1,h,w,3] or [b,h,w,3]
            imgs
        '''
        img_ids = img_ids if isArrayLike(img_ids) else [img_ids]
        #注意此处返回的是list
        img_infos = self.coco.loadImgs(img_ids)
        imgs = []
        if self.is_train and self.is_multi == False:
            bboxs = bboxs if isArrayLike(bboxs[0]) else [bboxs]
            #assert len(img_ids) == len(bboxs)
            for i, bbox in enumerate(bboxs):
                #避免py3中cv2.imread不能读取中文路径的问题
                img = cv2.imdecode(
                    np.fromfile(
                        self.img_dir + img_infos[i]['file_name'],
                        dtype=np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
                y1, y2 = int(bbox[1]), int(bbox[1] + bbox[3])
                x1, x2 = int(bbox[0]), int(bbox[0] + bbox[2])
                imgs.append(img[y1:y2, x1:x2, :])
            imgs = np.asarray(imgs)
        else:
            imgs = np.asarray([
                cv2.imdecode(
                    np.fromfile(
                        self.img_dir + img_info['file_name'], dtype=np.uint8),
                    cv2.IMREAD_COLOR)[..., ::-1] for img_info in img_infos
            ])

        return imgs

    def _load_kptvis_info(self, keypoints_info):
        '''
        description:获取关键点位置及是否可见的信息
        parameters:
            keypoints_info: list or list of list
                coords+vis
        return: tuple of ndarrays
            处理好的coords、vis信息
        '''
        #17为关键点数量,3为(x,y,vis),其中vis=0表示未标注,vis=1表示标注了但不可见,vis=2表示标注了且可见
        keypoints_info = np.asarray(keypoints_info)
        if len(keypoints_info.shape) == 1:
            kpt_vis = keypoints_info.reshape((-1, 3))
        elif len(keypoints_info.shape) == 2:
            kpt_vis = keypoints_info.reshape((keypoints_info.shape[0], -1, 3))
        else:
            raise ValueError('Uncorrect shape of keypoints info.')
        kpt = kpt_vis[..., 0:2].astype(np.float)
        vis = np.expand_dims(kpt_vis[..., 2], -1)

        return kpt, vis
