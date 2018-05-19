#encoding=utf-8
import cv2
import scipy.io as sio
import numpy as np
import h5py as h5
from utils.utils import isArrayLike


def mpii_read(
        image_path='D:\迅雷下载\mpii_human_pose_v1\images\\',
        label_path='D:\迅雷下载\mpii_human_pose_v1_u12_2\mpii_human_pose_v1_u12_1.mat'
):
    with open('mpii_valid.txt', 'r') as f:
        valid_name = f.readlines()
        valid_name = [name[:-1] for name in valid_name]
    #TODO height 改为height width h==w
    keys_train = [
        'index', 'img_name', 'bbox', 'coords', 'visible', 'kpt_num'
        #'normalize', 'torsoangle'暂时先不用
    ]
    keys_test = ['index', 'img_name', 'bbox']
    train_dict = {k: [] for k in keys_train}
    valid_dict = {k: [] for k in keys_train}
    test_dict = {k: [] for k in keys_test}
    data = sio.loadmat(label_path)['RELEASE']
    #data['img_train']、data['annolist']均为np.ndarray类型，shape为[1,1]
    #is_train、anno_list都是np.ndarray类型，shape为[1,24987]
    is_train = data['img_train'][0, 0][0]
    anno_list = data['annolist'][0, 0][0]
    for i in range(anno_list.shape[0]):
        #包括头部rectangle信息、annopoints信息(train)、scale、center
        anno_rect = anno_list[i]['annorect']
        #信息缺失 或存在anno_rect域但内部为空
        if anno_rect.shape[1] == 0 or anno_rect[0, 0] is None:
            continue
        fields = list(anno_rect.dtype.fields.keys())
        if not 'objpos' in fields or not 'scale' in fields:
            continue
        '''
        #因为一个anno_rect里可能存在好几个标注 即一张图可以对应几个center、scale等
        center_list = []
        height_list = []
        vis_list = []
        coords_list = []
        '''
        #h5py无法直接存储np.str,需要进行encode
        img_name = anno_list[i]['image']['name'][0, 0][0].encode()
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
                #读图片大概把时间变为了原来的100倍。。？
                img = cv2.imdecode(
                    np.fromfile(
                        image_path + img_name.decode(), dtype=np.uint8),
                    cv2.IMREAD_COLOR)[..., ::-1]
                shape = img.shape
                right_bottom = np.minimum(right_bottom, [shape[1], shape[0]])
                bbox = np.concatenate((left_up, right_bottom))
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
                        if img_name.decode() not in valid_name:
                            train_dict['visible'].append(vis)
                            train_dict['coords'].append(coords)
                            train_dict['index'].append(i)
                            train_dict['img_name'].append(img_name)
                            train_dict['bbox'].append(bbox)
                            train_dict['kpt_num'].append(kpt_num)
                        else:
                            valid_dict['visible'].append(vis)
                            valid_dict['coords'].append(coords)
                            valid_dict['index'].append(i)
                            valid_dict['img_name'].append(img_name)
                            valid_dict['bbox'].append(bbox)
                            valid_dict['kpt_num'].append(kpt_num)
            else:
                test_dict['index'].append(i)
                test_dict['img_name'].append(img_name)
                test_dict['bbox'].append(bbox)

    return train_dict, valid_dict, test_dict


'''
ret = mpii_read()
print('start to convert to h5 file..........')
with h5.File('mpii_train.h5', 'w') as f:
    f.attrs['name'] = 'mpii_train'
    for key in ret[0].keys():
        f[key] = ret[0][key]

print('train.h5 writen done.')

with h5.File('mpii_valid.h5', 'w') as f:
    f.attrs['name'] = 'mpii_valid'
    for key in ret[1].keys():
        f[key] = ret[1][key]

print('valid.h5 writen done.')

with h5.File('mpii_test.h5', 'w') as f:
    f.attrs['name'] = 'mpii_test'
    for key in ret[2].keys():
        f[key] = ret[2][key]

print('test.h5 writen done.')
with h5.File('mpii_train.h5', 'r') as f:
    #14679 22245 train  6619 11731 test 2729 6637 valid
    coords=f['coords'][:]
print(coords.shape)
'''