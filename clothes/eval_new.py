import tensorflow as tf
import os
import sys
import csv
from np_data_gen import *
from params_clothes import *
import cv2

def test(category,csv_file):
    #限制使用一个gpu
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    gpu_options = tf.GPUOptions(allow_growth=True)
    config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        # 训练好的模型参数
        if not os.path.exists(dir_para['trained_model_dir']):
            raise ValueError('Please give the trained model.')

        #源csv文件
        with open(csv_file,'r') as f:
            file_list=f.readlines()[1:]
            if category!='full':
                file_list=[elem for elem in file_list if category in elem]

        gen = clothes_batch_generater(category)
        batch = gen.generate_batch(category,1,8,train_para['batch_size'])
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
                img,size,image_name,cate=next(batch)
                predicted_heatmaps_batch = sess.run(opt, feed_dict={x: img})
                #[total,height,width,kpt_number]
                for i in range(train_para['batch_size']):
                    img_name=image_name[i]
                    cat=cate[i]
                    img_name2=img_name.split('/')[-1]
                    name=dir_para['test_data_dir']+'//'+img_name
                    pred_heatmaps=cv2.resize(predicted_heatmaps_batch[i],(size[i][1],size[i][0]))
                    height=size[i][0]
                    width=size[i][1]
                    #[h*w,k]
                    pred_flat=np.reshape(pred_heatmaps,(height*width,pred_heatmaps.shape[2]))
                    #[k]
                    #返回一个tuple，tuple中的每个array和np.argmax(coor,0)的形状相同
                    #即[k] tuple的元素数量为2,即height一个，width一个
                    ind=np.unravel_index(np.argmax(pred_flat,0),(height,width))
                    #[k,2] k个关节点最大值的坐标
                    ind2=np.stack([ind[0],ind[1]],1)

                    max_idx_list=[]
                    for m in range(len(all_kpt_list)):
                        max_idx_list.append('-1_-1_-1')

                    #用以可视化的图片
                    tmp=cv2.imread(name)
                    for j in range(ind2.shape[0]):
                        coor_j=pred_heatmaps[:,:,j]
                        if all_kpt_list[j] in categories_dict[cat] and coor_j[ind2[j][0],ind2[j][1]]>=thred:
                            cv2.circle(tmp,(ind2[j][1],ind2[j][0]),4,(255,0,0),-2)
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
            except Exception as e:
                break
        #return max_idx_dict
#for cat in categories:
#ret=test(cat)
train_para['is_train']=False
#train_para['batch_size']=1
test('full',dir_para['test_data_dir']+'/Annotations/test.csv')
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
