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
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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
        
        gen = clothes_batch_generater()
        batch = gen.generate_batch(category, file_list,1, 8, batch_size=train_para['batch_size'])

        #-------------------------------------读取预训练模型---------------------------------------------------------#
        saver = tf.train.import_meta_graph(
            dir_para['trained_model_dir'] + '/' + 'full'+ '-400000.meta')
        saver.restore(
            sess, dir_para['trained_model_dir'] + '/' + 'full-400000')
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('input:0')
        if train_para['using_cg']==False:
            opt = graph.get_tensor_by_name(
                'hourglass_model/hourglass4/Conv_1/BiasAdd:0')
        else:
            opt = graph.get_tensor_by_name(
                'hourglass_model/hourglass4/ResizeBilinear:0')

        #--------------------------------------开始测试-------------------------------------------------------------#
        max_idx_dict = {}
        while True:
            try:
                #------------------------------------获取batch并处理-------------------------------------------------#
                img, size, image_name, cate = next(batch)
                predicted_heatmaps_batch = sess.run(opt, feed_dict={x: img})
                if train_para['using_cg']:
                    predicted_heatmaps_batch=get_locmap(predicted_heatmaps_batch)
                for i in range(train_para['batch_size']):
                    img_name = image_name[i]
                    cat = cate[i][:-1]

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
                                            (128, 0, 255), -2)
                                ind_str = str(ind2[j][1]) + '_' + str(
                                    ind2[j][0]) + '_1'
                            else:
                                #被遮挡的点
                                ind_str = str(ind2[j][1]) + '_' + str(
                                    ind2[j][0]) + '_0'
                                cv2.circle(tmp, (ind2[j][1], ind2[j][0]), 4,
                                            (0, 128, 255), -2)
                            max_idx_list[j] = ind_str
                    cv2.imwrite('./res-400000/%s' % img_name2, tmp)
                    max_idx_dict[img_name] = max_idx_list

            except Exception as e:
                import traceback
                traceback.print_exc()
                break

        return max_idx_dict


if __name__ == '__main__':
    freeze_support()
    #--------------------------------------------解析参数-----------------------------------------------------------#
    parser = argparse.ArgumentParser()
    parser.add_argument('category', type=str, choices=categories)
    args = parser.parse_args()
    #???这个局部变量不会传给子进程？
    train_para['is_train'] = False

    ret=test(args.category, dir_para['test_data_dir'] + '/test.csv')

    #---------------------------------------------开始写入----------------------------------------------------------#
    print('Start to write...........')
    with open('./result.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for key in ret.keys():
            cate=key.split('/')[1]
            res = [key, cate]
            #找到图片对应的坐标list
            for elem in ret[key]:
                res.append(elem)
            #print(res)
            writer.writerow(res)