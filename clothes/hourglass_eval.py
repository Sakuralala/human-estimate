#for hourglass test.
import tensorflow as tf
import os
import sys
import csv
#import argparse  #用以解析运行时的参数
from clothes_data_generate import *
from hourglass import HourglassModel
from params_clothes import *


def test(category):
    #非训练
    train_para['is_train'] = False
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # 训练好的模型参数
        if not os.path.exists(dir_para['trained_model_dir']):
            raise ValueError('Please give the trained model.')
        #一张张测试防止维度不同
        gen = clothes_batch_generater(category, 1, train_para['batch_size'])
        # 用以作为输入的二进制文件
        if not os.path.exists(dir_para['tf_dir_test']):
            os.mkdir(dir_para['tf_dir_test'])
            print('Created tfrecords dir for test:', dir_para['tf_dir_test'])
        rec_name = category + '.tfrec'
        for (root, dirs, files) in os.walk(dir_para['tf_dir_test']):
            if rec_name not in files:
                gen.convert_to_tfrecords_single(category,
                                                dir_para['test_data_dir'],
                                                dir_para['tf_dir_test'])
    #生成batch
        batch = gen.dataset_input_new(dir_para['tf_dir_test'] + '/' + rec_name)
        # Start TF
        coord=tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess)
        #输出名：hourglass_model/hourglass8/inter_output/output:0
        #saver
        saver = tf.train.import_meta_graph(category + '-20000.meta')
        saver.restore(sess,
                      tf.train.latest_checkpoint(
                          dir_para['trained_model_dir']))
        graph = tf.get_default_graph()
        #使用place_holder
        x = graph.get_tensor_by_name('input:0')
        opt = graph.get_tensor_by_name(
            'hourglass_model/hourglass8/inter_output/output:0')
        #用以保存图片名及对应坐标
        max_idx_dict={}
        try:
            while not coord.should_stop():
                #最终输出的heatmap 缩小了四倍的
                predicted_heatmaps = sess.run(opt, feed_dict={x: batch[0]})
                img_name=sess.run(batch[1])
                predicted_heatmaps = tf.image.resize_bilinear(
                    predicted_heatmaps,
                    [tf.shape(predicted_heatmaps)[1]*4,
                    tf.shape(predicted_heatmaps)[2]*4])
                predicted_heatmaps=tf.squeeze(predicted_heatmaps)
                coor=sess.run(predicted_heatmaps)
                max_idx_list=[]
                for i in range(len(all_kpt_list)):
                    max_idx_list.append('-1_-1_-1')
                #不需要背景类 故-1
                for i in range(coor.shape[-1]-1):
                    mean=np.mean(coor[:,:,i])
                    if mean<thred:
                        #不存在或被遮挡的点
                        ind_str='0_0_0'
                    else:
                        ind=np.unravel_index(np.argmax(coor[:,:,i]),coor.shape[:2])
                        ind_str=str(ind[1])+'_'+str(ind[0])+'_1'
                    #找到关节点在csv文件中的对应位置索引
                    index=all_kpt_list.index(categories_dict[category][i])
                    max_idx_list[index](ind_str)
                print(max_idx_list)
                max_idx_dict[img_name]=max_idx_list
            coord.join()
        except Exception as e:
            coord.request_stop(e)

    return max_idx_dict

for cat in categories:
    #ret=test(cat)
    ret={'1.jpg':['1_2_3','4_5_5','7_8_0'],'2.jpg':[2,3,4]}
    f=open('result.csv','a',newline='')
    writer=csv.writer(f)
    for key in ret:
        res=[key,cat]
        for elem in ret[key]:
            res.append(elem)
        writer.writerow(res)
