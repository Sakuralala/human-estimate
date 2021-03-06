#encoding=utf-8
import tensorflow as tf
import os
import sys
import csv
#import argparse  #用以解析运行时的参数
from data_generate import *
from params_clothes import *
import cv2

def test(category):
    #非训练
    #train_para['is_train'] = False
    #限制使用一个gpu
    os.environ["CUDA_VISIBLE_DEVICES"]='1'
    gpu_options = tf.GPUOptions(allow_growth=True)
    config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        # 训练好的模型参数
        if not os.path.exists(dir_para['trained_model_dir']):
            raise ValueError('Please give the trained model.')
        #一张张测试防止维度不同
        gen = clothes_batch_generater(category)
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
        tf.train.start_queue_runners(sess=sess)
        #saver
        saver = tf.train.import_meta_graph(dir_para['trained_model_dir']+'/'+category + '-240000.meta')
        saver.restore(sess,dir_para['trained_model_dir']+'/'+category+'-240000')

        graph = tf.get_default_graph()
        '''
        sess.run(tf.global_variables_initializer())
        variable_name = [v.name for v in tf.trainable_variables()]
        print(variable_name)
        '''
        #使用place_holder
        x = graph.get_tensor_by_name('input:0')
        #print(x)
        opt = graph.get_tensor_by_name(
            'hourglass_model/hourglass4/Conv_1/BiasAdd:0')
        #用以保存图片名及对应坐标
        max_idx_dict={}
        i=0
        while True:
            try:
                #最终输出的heatmap 缩小了四倍
                img,image_name,size,cate=sess.run(batch)
                predicted_heatmaps = sess.run(opt, feed_dict={x: img})
                predicted_heatmaps = tf.image.resize_bilinear(
                    predicted_heatmaps,
                    size[0])
                #[total,height,width,kpt_number]
                coor_batch=sess.run(predicted_heatmaps)
                for i in range(train_para['batch_size']):
                    #[8]
                    img_name=image_name[i].decode() 
                    cat=cate[i].decode() 
                    img_name2=img_name.split('/')[-1]
                    name=dir_para['test_data_dir']+'//'+img_name
                    coor=coor_batch[i]
                    height=coor.shape[0]
                    width=coor.shape[1]
                    #[h*w,k]
                    coor_flat=np.reshape(coor,(height*width,coor.shape[2]))
                    #[k]
                    #返回一个tuple，tuple中的每个array和np.argmax(coor,0)的形状相同
                    #即[k] tuple的元素数量为2,即height一个，width一个
                    ind=np.unravel_index(np.argmax(coor_flat,0),(height,width))
                    #[k,2] k个关节点最大值的坐标
                    ind2=np.stack([ind[0],ind[1]],1)
                    #print(ind2)
                    #[k] True 表示该最大值有效 False表明该关节点不存在或被遮挡
                    #coor_cond=np.greater(coor_maxval,thred)
                    #[k,2]并且将
                    #ind=np.transpose(ind*np.cast(coor_cond,np.int32),(1,0))
                    max_idx_list=[]
                    for m in range(len(all_kpt_list)):
                        max_idx_list.append('-1_-1_-1')

                    tmp=cv2.imread(name)
                    for j in range(ind2.shape[0]):
                        coor_j=coor[:,:,j]
                        if all_kpt_list[j] in categories_dict[cat] and coor_j[ind2[j][0],ind2[j][1]]>=thred:
                            cv2.circle(tmp,(ind2[j][1],ind2[j][0]),6,(255,0,0),-3)
                    cv2.imwrite('./res/%s'%img_name2,tmp)
                    '''
                        if coor_i[ind[0][i],ind[1][i]]<thred:
                            #不存在或被遮挡的点
                            ind_str='0_0_0'
                        else:
                            ind_str=str(ind[0][i])+'_'+str(ind[1][i])+'_1'
                        #找到关节点在csv文件中的对应位置索引
                        index=all_kpt_list.index(categories_dict[category][i])
                        #print(index)
                        max_idx_list[index]=ind_str
                        #print(max_idx_list)
                    print(max_idx_list)
                    max_idx_dict[img_name[0]]=max_idx_list
                    '''
            #数据读取迭代器已至末尾
            except tf.errors.OutOfRangeError:
                print('End of %s.'%category)
                break
        #return max_idx_dict
#for cat in categories:
#ret=test(cat)
train_para['is_train']=False
#train_para['batch_size']=1
test('full')
#ret={'1.jpg':['1_2_3','4_5_5','7_8_0'],'2.jpg':[2,3,4]}
'''
f=open('./result.csv','a',newline='')
writer=csv.writer(f)
for key in ret.keys():
    res=[key,cat]
#找到图片对应的坐标list
    for elem in ret[key]:
        res.append(elem)
    print(res)
    writer.writerow(res)
'''
