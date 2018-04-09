import tensorflow as tf
import os
import sys
import csv
import cv2
import argparse
from np_data_gen import *
from params_clothes import *
from multiprocessing import freeze_support


def test(category, csv_file):
    '''
    desc:进行测试。
    args:
        category:服饰类别，若为'full',则之后batch中得到的cate是有用的(因为存在各个类别),否则后面的cate是冗余的(和category一致)。
        csv_file:源文件。
    '''
    #----------------------------------------session的一些配置--------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        # 训练好的模型参数
        if not os.path.exists(dir_para['trained_model_dir']):
            raise ValueError('Please give the trained model.')

        #------------------------------------读取源文件并取得batch---------------------------------------------------#
        with open(csv_file, 'r') as f:
            file_list = f.readlines()[1:]
            if category != 'full':
                file_list = [elem for elem in file_list if category in elem]
        gen = clothes_batch_generater(category)
        batch = gen.generate_batch(category, 1, 8, train_para['batch_size'])

        #-------------------------------------读取预训练模型---------------------------------------------------------#
        saver = tf.train.import_meta_graph(
            dir_para['trained_model_dir'] + '/' + category + '-240000.meta')
        saver.restore(
            sess, dir_para['trained_model_dir'] + '/' + category + '-240000')
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('input:0')
        opt = graph.get_tensor_by_name(
            'hourglass_model/hourglass4/Conv_1/BiasAdd:0')

        #--------------------------------------开始测试-------------------------------------------------------------#
        max_idx_dict = {}
        while True:
            try:
                #------------------------------------获取batch并处理-------------------------------------------------#
                img, size, image_name, cate = next(batch)
                predicted_heatmaps_batch = sess.run(opt, feed_dict={x: img})
                #[total,height,width,kpt_number]
                for i in range(train_para['batch_size']):
                    img_name = image_name[i]
                    cat = cate[i]
                    img_name2 = img_name.split('/')[-1]
                    name = dir_para['test_data_dir'] + '//' + img_name

                    #------------------------------将图片恢复到原尺寸并获取最大值的坐标--------------------------------#
                    pred_heatmaps = cv2.resize(predicted_heatmaps_batch[i],
                                               (size[i][1], size[i][0]))
                    height = size[i][0]
                    width = size[i][1]
                    #[h*w,k]
                    pred_flat = np.reshape(
                        pred_heatmaps,
                        (height * width, pred_heatmaps.shape[2]))
                    #[k]
                    #返回一个tuple，tuple中的每个array和np.argmax(coor,0)的形状相同
                    #即[k] tuple的元素数量为2,即height一个，width一个
                    ind = np.unravel_index(
                        np.argmax(pred_flat, 0), (height, width))
                    #[k,2] k个关节点最大值的坐标
                    ind2 = np.stack([ind[0], ind[1]], 1)

                    max_idx_list = []
                    for m in range(len(all_kpt_list)):
                        max_idx_list.append('-1_-1_-1')

                    #--------------------------可视化各个关键点并将其写list-------------------------------------------#
                    #用以可视化的图片
                    tmp = cv2.imread(name)
                    for j in range(ind2.shape[0]):
                        coor_j = pred_heatmaps[:, :, j]
                        if all_kpt_list[j] in categories_dict[cat]:
                            if coor_j[ind2[j][0], ind2[j][1]] >= thred:
                                cv2.circle(tmp, (ind2[j][1], ind2[j][0]), 4,
                                           (255, 0, 0), -2)
                                #被遮挡的点
                                ind_str = '0_0_0'
                            else:
                                ind_str = str(ind[0][i]) + '_' + str(
                                    ind[1][i]) + '_1'
                            #找到关节点在csv文件中的对应位置索引
                            index = all_kpt_list.index(
                                categories_dict[category][i])
                            max_idx_list[index] = ind_str
                    cv2.imwrite('./res/%s' % img_name2, tmp)
                    max_idx_dict[img_name[0]] = max_idx_list

            except Exception as e:
                break

        return max_idx_dict


if __name__ == '__main__':
    mp.freeze_support()
    #--------------------------------------------解析参数-----------------------------------------------------------#
    parser = argparse.ArgumentParser()
    parser.add_argument('category', type=str, choices=categories)
    args = parser.parse_args()
    train_para['is_train'] = False

    test(args.category, dir_para['test_data_dir'] + '/Annotations/test.csv')

    #---------------------------------------------开始写入----------------------------------------------------------#
    with open('./result.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for key in ret.keys():
            res = [key, cat]
            #找到图片对应的坐标list
            for elem in ret[key]:
                res.append(elem)
            print(res)
            writer.writerow(res)
