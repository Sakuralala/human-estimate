#for hourglass test.
import tensorflow as tf
import os
import sys
import csv
#import argparse  #用以解析运行时的参数
from data_generate import *
from hourglass import HourglassModel
from params_clothes import *
from utils import kpt_coor_translate
from PIL import ImageDraw,Image
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
        #输出名：hourglass_model/hourglass8/inter_output/output:0
        #gen = clothes_batch_generater(category)
        # 用以作为输入的二进制文件
        #imgs,img_names=gen.data_generate_single(category,dir_para['test_data_dir'])
        #print(type(imgs))
        #saver
        saver = tf.train.import_meta_graph(dir_para['trained_model_dir']+'/'+category + '-340000.meta')
        saver.restore(sess,dir_para['trained_model_dir']+'/'+category+'-340000')

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
            'hourglass_model/hourglass4/inter_output/BiasAdd/BiasAdd:0')
        #用以保存图片名及对应坐标
        max_idx_dict={}
        i=0
        while True:
            try:
                #最终输出的heatmap 缩小了四倍的
                img,img_name,size,cat=sess.run(batch)
                #[8]
                #print(img_name.shape)
                img_name=img_name[0].decode()
                cat=cat[0].decode()
                img_name2=img_name.split('/')[-1]
                name=dir_para['test_data_dir']+'//'+img_name
                predicted_heatmaps = sess.run(opt, feed_dict={x: img})
                #print(predicted_heatmaps.shape)
                for i in range(predicted_heatmaps.shape[-1]-1):
                    if all_kpt_list[i] in categories_dict[cat]:
                        scipy.misc.imsave('origin%s.png'%img_name2,predicted_heatmaps[0,:,:,i])

                '''
                predicted_heatmaps = tf.image.resize_bilinear(
                    predicted_heatmaps,
                    size[0])
                predicted_heatmaps=tf.squeeze(predicted_heatmaps)
    #[total,height,width,kpt_number+1]
                coor=sess.run(predicted_heatmaps)
                '''
                #predicted_heatmaps=np.transpose(predicted_heatmaps,())
                #print(coor.shape)
    #for j in range(coor.shape[0]):
                #去除背景 [h,w,k]
                coor=predicted_heatmaps[0,:,:,:predicted_heatmaps.shape[-1]-1]
                #coor=coor[:,:,:coor.shape[2]-1]
                height=coor.shape[0]
                width=coor.shape[1]
                #print('h:%s,w:%s'%(height,width))
                #[h*w,k]
                coor_flat=np.reshape(coor,(height*width,coor.shape[2]))
                #[k]
                #返回一个tuple，tuple中的每个array和np.argmax(coor,0)的形状相同
                #即[k] tuple的元素数量为2,即height一个，width一个
                #print(coor_flat.shape)
                #print(height,width)
                print(np.argmax(coor_flat,0))
                ind=np.unravel_index(np.argmax(coor_flat,0),(height,width))
                #print(ind[0])
                #print(ind[1])
                #[k,2] k个关节点最大值的坐标
                ind2=np.stack([ind[0],ind[1]],1)
                scale=np.asarray([size[0,0]/coor.shape[0],size[0,1]/coor.shape[1]],np.float)
                old_center=np.asarray([coor.shape[0]/2-1,coor.shape[1]/2-1],np.float)
                new_center=size[0]/2-1
                #[k,2]
                ind2=sess.run(kpt_coor_translate(ind2,scale,old_center,new_center))
                print(ind2)
                #print(ind2)
                #[k] True 表示该最大值有效 False表明该关节点不存在或被遮挡
                #coor_cond=np.greater(coor_maxval,thred)
                #[k,2]并且将
                #ind=np.transpose(ind*np.cast(coor_cond,np.int32),(1,0))
                max_idx_list=[]
                for i in range(len(all_kpt_list)):
                    max_idx_list.append('-1_-1_-1')
                #print(ind[0].shape[0])  
                #print(img_name[0].decode())
                tmp=cv2.imread(name)
                #tmp=Image.open(name,'r')
                #draw=ImageDraw.Draw(tmp)
                for i in range(ind2.shape[0]):
                    #使用mean好像效果不太好。。有负值。。。
                    #mean=np.mean(coor[:,:,i])
                    #print(img_name[0])
                    coor_i=coor[:,:,i]
                    #scipy.misc.imsave('pre%s.png'%img_name2,coor_i)
                    print(ind2[i,0],ind2[i,1])
                    if all_kpt_list[i] in categories_dict[cat]:
                        #draw.point([ind[0][i],ind[1][i]],fill=(0,255,0))
                        cv2.circle(tmp,(ind2[i][0],ind2[i][1]),6,(255,0,0),-3)
                #tmp.save('./res/%s'%img_name2)
                cv2.imwrite('./res/%s'%img_name2,tmp)
                    #if i%10:
                    #    scipy.misc.imsave('pre%d.png'%i,coor_i)
                    #print(coor_i[ind[0][i],ind[1][i]])
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
