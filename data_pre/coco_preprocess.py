#这个根据实际情况改
from pycocotools.coco import COCO
import numpy as np
import cv2
#用于测试时的可视化
import imgaug as ia
from imgaug import augmenters as iaa
from utils_new import *
from data_preprocess import DataPreProcess


#------------------------------------------COCO数据预处理类-----------------------------------------------------------
class COCOPreProcess(DataPreProcess):
    def __init__(self,
                 img_dir,
                 json_dir,
                 is_train=True,
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
            is_train: bool
                训练/测试
            batch_size: int
                ...
            max_epoch: int 
                最大迭代轮数 对于test来说该参数无效
            filter_num: int
                过滤掉小于多少个关键点被标注的人,对test来说该参数无效
        '''
        super(COCOPreProcess, self).__init__(img_dir, is_train, batch_size,
                                             max_epoch)
        self.filter_num = filter_num
        self.coco = COCO(json_dir)
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        if self.is_train:
            self.ann_ids = self.coco.getAnnIds(catIds=self.cat_ids)
            self.ann_ids = np.asarray([
                ann_id for ann_id in self.ann_ids if self.coco.loadAnns(ann_id)
                [0]['num_keypoints'] > self.filter_num
            ])
            self.data_size = self.ann_ids.shape[0]
            print('Total person count:', self.ann_ids.shape[0])
        else:
            self.img_ids = np.asarray(self.coco.getImgIds(catIds=self.cat_ids))
            self.data_size = self.img_ids.shape[0]
            print('Total image count:', self.img_ids.shape[0])
        #用于随机产生序列 为了使每个进程得到不同的perm 先置为none
        #对于test来说不需要打乱
        self.perm = None if self.is_train == True else np.arange(
            self.img_ids.shape[0])

    def get_batch(self, resized_size=(192, 256)):
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
            resized_size: tuple (width,height)
                统一resize的图片大小
        return: list
            一个batch
        '''
        index = self._update_cur_index()
        #1
        img_batch = []
        if self.is_train:
            ann_info = self._load_ann_info(index)
            kpt_batch = []
            kpt_img_batch = []
            for ann in ann_info:
                #2 3 4
                bbox = ann['bbox']
                img = self._load_imgs(ann['image_id'], bbox)
                #5
                kpt, vis = self._load_kptvis_info(ann['keypoints'])
                kpt -= np.asarray(bbox[0:2])
                #6
                img = cv2.resize(img, resized_size)
                kpt = scale_coords_trans(
                    kpt, np.stack([bbox[2], bbox[3]]),
                    np.stack([resized_size[0], resized_size[1]]))
                #7、8 scale、flip、rotate、brightness、blur、hue等.
                img_aug, kpt = augment_both(img, kpt)
                #9
                img_batch.append(img_aug)
                #测试用
                kpt_img_batch.append(
                    ia.KeypointsOnImage.from_coords_array(
                        kpt[0], img_aug.shape))
                kpt_vis = np.concatenate((kpt[0], vis), -1)
                #找到未标注的点
                indices = np.where(kpt_vis[:, -1] == 0)[0]
                #清零未标注的点的坐标
                kpt_vis[indices] = 0
                kpt_batch.append(kpt_vis)

            return img_batch, kpt_batch, kpt_img_batch
        else:
            img_batch = self._load_imgs(self.img_ids[index])
            for img in img_batch:
                img = cv2.resize(img, resized_size)

            return img_batch

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

    def _load_imgs(self, img_ids, bboxs=None):
        '''
        description: 根据提供的img_ids找到对应的图片,若为训练则根据bbox将对应的人抠出,否则直接返回对应的图片
        parameters:
            img_ids: int or list of int 
                一个或一个batch的图片id
            bbox:list or list of list
                一个或一组bbox,注意bbox的顺序要与img_ids一一对应,若为None，表示为测试
        return: ndarray  [h,w,3] or [b,h,w,3]
            imgs
        '''
        img_ids = img_ids if isArrayLike(img_ids) else [img_ids]
        #注意此处返回的是list
        img_infos = self.coco.loadImgs(img_ids)
        imgs = []
        if self.is_train:
            assert bboxs != None
            bboxs = bboxs if isArrayLike(bboxs[0]) else [bboxs]
            assert len(img_ids) == len(bboxs)
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

        return imgs if imgs.shape[0] > 1 else imgs[0]

    def _load_kptvis_info(self, keypoints_info):
        '''
        description:获取关键点位置及是否可见的信息
        parameters:
            keypoints_info: list
                coords+vis
        return: tuple of ndarrays
            处理好的coords、vis信息
        '''
        #17为关键点数量,3为(x,y,vis),其中vis=0表示未标注,vis=1表示标注了但不可见,vis=2表示标注了且可见
        kpt_vis = np.asarray(keypoints_info).reshape((17, 3))
        kpt = kpt_vis[:, 0:2].astype(np.float)
        vis = np.expand_dims(kpt_vis[:, 2], -1)

        return kpt, vis
