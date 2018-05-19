#encoding=utf-8
#只需要知道图像的大小而不想读入图像
from PIL import Image
import copy
import scipy.io as sio
import numpy as np
import json
from utils.utils import isArrayLike
from pycocotools.coco import COCO


def mpii_read(
        image_path='D:\迅雷下载\mpii_human_pose_v1\images\\',
        label_path='D:\迅雷下载\mpii_human_pose_v1_u12_2\mpii_human_pose_v1_u12_1.mat'
):
    with open('mpii_valid.txt', 'r') as f:
        valid_name = f.readlines()
        valid_name = [name[:-1] for name in valid_name]

    image = {'id': -1, 'file_name': '', 'height': -1, 'width': -1}
    ann = {
        'id': -1,
        'image_id': -1,
        'bbox': None,
        'keypoints': None,
        'num_keypoints': -1
    }
    train_dict = {'images': [], 'annotations': []}
    valid_dict = {'images': [], 'annotations': []}
    test_dict = {'images': [], 'annotations': []}
    data = sio.loadmat(label_path)['RELEASE']
    #data['img_train']、data['annolist']均为np.ndarray类型，shape为[1,1]
    #is_train、anno_list都是np.ndarray类型，shape为[1,24987]
    is_train = data['img_train'][0, 0][0]
    anno_list = data['annolist'][0, 0][0]
    ann_id = 0
    for i in range(anno_list.shape[0]):
        image['id'] = i
        #包括头部rectangle信息、annopoints信息(train)、scale、center
        anno_rect = anno_list[i]['annorect']
        #信息缺失 或存在anno_rect域但内部为空
        if anno_rect.shape[1] == 0 or anno_rect[0, 0] is None:
            continue
        fields = list(anno_rect.dtype.fields.keys())
        if not 'objpos' in fields or not 'scale' in fields:
            continue

        img_name = anno_list[i]['image']['name'][0, 0][0]
        image['file_name'] = img_name
        img = Image.open(image_path + img_name, 'r')
        image['height'] = img.size[0]
        image['width'] = img.size[1]

        ann['image_id'] = i
        valid_img = False
        for j in range(anno_rect.shape[1]):
            #[1,k] [0,j]得到一个ndarray [0,0]除去array包的两层得到一个元祖，长度为2，每一个元素均为一个二维ndarray 里面存值。
            if anno_rect['objpos'][0,
                                   j].shape[1] > 0 and anno_rect['scale'][0,
                                                                          j].shape[1] > 0:
                # TypeError: Object dtype dtype('O') has no native HDF5 equivalent????
                #h5py不支持存储长度不一的array。 how to work around?
                #一个方法(hg的作者的)就是对于每一个人都对于一张图片，即使可能一张图片里有几个人。
                #google了好久也没找到个靠谱的方法，那暂时只能用这个方法了
                #或者更改存储的结构？不过上面这种做法产生的h5文件也不大，暂时可以先用着。
                center = np.asarray([
                    anno_rect['objpos'][0, j][0, 0]['x'][0, 0],
                    anno_rect['objpos'][0, j][0, 0]['y'][0, 0]
                ])
                half_height = anno_rect['scale'][0, j][0, 0] * 100
                #[x1,y1,x2,y2]
                #注意此处需要判断是否超出边界
                #那就需要读入图片以获取尺寸
                #TODO
                left_up = center - half_height
                left_up = np.maximum(left_up, [0.0, 0.0])
                right_bottom = center + half_height
                '''
                img = cv2.imdecode(
                    np.fromfile(image_path + img_name, dtype=np.uint8),
                    cv2.IMREAD_COLOR)[..., ::-1]
                shape = img.shape
                '''
                right_bottom = np.minimum(right_bottom,
                                          [img.size[1], img.size[0]])
                bbox = np.concatenate((left_up, right_bottom)).tolist()
                ann['bbox'] = bbox
            else:
                continue
            #anno_rect为1x1但内部为空 经统计为0(train的)
            if is_train[i]:
                #or 不存在annopoints域 注意此处可能会存在一张图片里有多个人未被标注的情况
                #经统计 存在99个这样的人 95张这样的图片
                #or annopoints域为空
                #经统计存在134个这样的人,127张这样的图片
                if 'annopoints' in list(anno_rect.dtype.fields.keys(
                )) and anno_rect[0, j]['annopoints'].shape[0] > 0:
                    #[0,0][0]得到的np.void类对象  再用索引取得对应数据
                    points = anno_rect[0, j]['annopoints']['point'][0, 0][0]
                    #id x y is_visible
                    if len(points.dtype.fields.keys()) == 4:
                        coords = np.zeros((16, 2))
                        vis = np.zeros((16))
                        kpt_num = 0
                        x, y, part_id, visible = points['x'], points[
                            'y'], points['id'], points['is_visible']
                        for k, index in enumerate(part_id):
                            #坐标点顺序调整为和readme中一样
                            coords[index, 0], coords[index, 1] = x[k], y[k]
                            if visible[k].shape[0] > 0:
                                vis[index] = visible[k]
                                kpt_num += 1
                        coords_vis = np.concatenate(
                            (coords, np.expand_dims(vis, 1)),
                            -1).reshape(48).tolist()
                        ann['keypoints'] = coords_vis
                        ann['num_keypoints'] = kpt_num
                        ann['id'] = ann_id
                        ann_id += 1
                        valid_img = True
                        if img_name not in valid_name:
                            train_dict['annotations'].append(
                                copy.deepcopy(ann))
                        else:
                            valid_dict['annotations'].append(
                                copy.deepcopy(ann))
            else:
                valid_img = True
                ann['id'] = ann_id
                ann_id += 1
                ann_copy = copy.deepcopy(ann)
                if 'keypoints' in ann_copy.keys():
                    ann_copy.pop('keypoints')
                if 'num_keypoints' in ann_copy.keys():
                    ann_copy.pop('num_keypoints')
                test_dict['annotations'].append(ann_copy)
        if valid_img:
            if is_train[i]:
                if img_name not in valid_name:
                    train_dict['images'].append(copy.deepcopy(image))
                else:
                    valid_dict['images'].append(copy.deepcopy(image))
            else:
                test_dict['images'].append(copy.deepcopy(image))

    print('start to write to json............')
    with open('mpii_train.json', 'w') as f:
        json.dump(train_dict, f)

    with open('mpii_test.json', 'w') as f:
        json.dump(test_dict, f)

    with open('mpii_valid.json', 'w') as f:
        json.dump(valid_dict, f)

    print('write to json completed.')


mpii_read()

#mpii = COCO('mpii_train.json')
