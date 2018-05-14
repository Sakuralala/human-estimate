import scipy.io as sio
import numpy as np
import h5py as h5


def mpii_read(
        image_path='D:\迅雷下载\mpii_human_pose_v1\images\\',
        label_path='D:\迅雷下载\mpii_human_pose_v1_u12_2\mpii_human_pose_v1_u12_1.mat'
):
    keys_train = [
        'index', 'img_name', 'center', 'height', 'coords', 'visible'
        #'normalize', 'torsoangle'暂时先不用
    ]
    keys_test = ['index', 'img_name', 'center', 'height']
    train_dict = {k: [] for k in keys_train}
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
        #print(i)
        fields = list(anno_rect.dtype.fields.keys())
        if not 'objpos' in fields or not 'scale' in fields:
            continue
        #因为一个anno_rect里可能存在好几个标注 即一张图可以对应几个center、scale等
        center_list = []
        height_list = []
        vis_list = []
        coords_list = []
        img_name = anno_list[i]['image']['name'][0, 0][0]  #.encode()
        for j in range(anno_rect.shape[1]):
            #[1,k] [0,j]得到一个ndarray [0,0]除去array包的两层得到一个元祖，长度为2，每一个元素均为一个二维ndarray 里面存值。
            if anno_rect['objpos'][0, j].shape[1] > 0:
                # TypeError: Object dtype dtype('O') has no native HDF5 equivalent????
                center_list.append(
                    np.asarray([
                        anno_rect['objpos'][0, j][0, 0][0][0, 0],
                        anno_rect['objpos'][0, j][0, 0][1][0, 0]
                    ]))
            if anno_rect['scale'][0, j].shape[1] > 0:
                height_list.append(anno_rect['scale'][0, j][0, 0] * 200)
            #anno_rect为1x1但内部为空 经统计为0(train的)
            if is_train[i]:
                #or 不存在annopoints域 注意此处可能会存在一张图片里有多个人未被标注的情况
                #经统计 存在99个这样的人 95张这样的图片
                #or annopoints域为空
                #经统计存在134个这样的人,127张这样的图片
                if 'annopoints' in list(anno_rect.dtype.fields.keys(
                )) and anno_rect[0, j]['annopoints'].shape[0] > 0:
                    coords = np.zeros((16, 2))
                    vis = np.zeros((16))
                    #[0,0][0]得到的np.void类对象  再用索引取得对应数据
                    points = anno_rect[0, j]['annopoints']['point'][0, 0][0]
                    pt_fields = points.dtype.fields.keys()
                    if 'x' not in pt_fields or 'y' not in pt_fields or 'id' not in pt_fields or 'is_visible' not in pt_fields:
                        continue
                    x, y, part_id, visible = points['x'], points['y'], points[
                        'id'], points['is_visible']
                    for k, index in enumerate(part_id):
                        #坐标点顺序调整为和readme中一样
                        coords[index, 0], coords[index, 1] = x[k], y[k]
                        if visible[k].shape[0] > 0:
                            vis[index] = visible[k]
                    coords_list.append(coords)
                    vis_list.append(vis)
        if is_train[i]:
            #没有被标注的图就扔了
            if len(vis_list) > 0:
                train_dict['visible'].append(vis_list)
                train_dict['coords'].append(coords_list)
                train_dict['index'].append(i)
                train_dict['img_name'].append(img_name)
                train_dict['height'].append(height_list)
                train_dict['center'].append(center_list)
        else:
            test_dict['index'].append(i)
            test_dict['img_name'].append(img_name)
            test_dict['height'].append(height_list)
            test_dict['center'].append(center_list)

    return train_dict, test_dict


ret = mpii_read()
print('start to convert to h5 file..........')
with h5.File('mpii_train.h5', 'w') as f:
    f.attrs['name'] = 'mpii_train'
    for key in ret[0].keys():
        print(key)
        #f[key] = ret[0][key]
        if key == 'index':
            f[key] = ret[0][key]
        else:
            if key == 'img_name':
                dt = h5.special_dtype(vlen=str)
            else:
                dt = h5.special_dtype(vlen=np.float)
            f.create_dataset(key, dtype=dt, data=ret[0][key])

print('train.h5 writen done.')

with h5.File('mpii_test.h5', 'w') as f:
    f.attrs['name'] = 'mpii_test'
    for key in ret[1].keys():
        f[key] = ret[1][key]

print('test.h5 writen done.')
