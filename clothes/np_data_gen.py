#使用np而不是tf来进行数据的预处理和生成
import numpy as np
import os
import sys
import csv
import cv2
from params_clothes import *
from parallel_gen import GeneratorEnqueuer
from utils_new import *


class clothes_batch_generater():
    def __init__(self,
                 sigma=1.0,
                 resized_height=256,
                 resized_width=256,
                 #is_resize=True,#必须要resize 因为无法保证每张图片的大小是一致的
                 coor_noise=True,
                 crop_center_noise=True):
        '''
        args:
            resized_height:裁剪后的图像高
            resized_width:裁剪后的图像宽
            sigma:二维高斯核的方差
	        coor_noise:是否在关节点的位置上加上一定噪声
	        crop_center_noise:是否在crop图像的中心加上一定噪声
        '''
        self.sigma = sigma
        self.resized_size = np.stack([resized_height,resized_width])
        self.coor_noise = coor_noise
        self.crop_center_noise = crop_center_noise


    def generate_batch(self,category,file_list,slavers=10,max_queue_size=16,batch_size=8):
        '''
        desc：生成batch
        args:
            input_size:总的输入数量
            slavers:线程数量
            batch_size:。。。
        '''
        try:
            di=DataIterator(category,file_list,batch_size=batch_size)
            gen_enq=GeneratorEnqueuer(di)
            #产生了多个slaver并开始工作
            gen_enq.start(slaver=slavers,max_queue_size=max_queue_size)
            while True:
                #is_running/empty在get中已经判断了
                batch_output=gen_enq.get() 
                #返回的也是一个生成器 所以需要next
                batch_output=next(batch_output)
                yield batch_output
                batch_output=None
        finally:
            pass
            #if gen_enq is not None:
                #gen_enq.stop()



class DataIterator():
    def __init__(self,category,file_list,batch_size=8):
        '''
        desc:因为win下generator不能被pickle所以只能改成使用iterator了
        args:
            fd:文件描述符
        '''
        #self.input_size=input_size 
        self.batch_size=batch_size 
        self.resized_size=np.stack([input_para['resized_height'],input_para['resized_width']])
        #除去第一行
        self.file_list=file_list
        self.input_size=len(self.file_list)
        self.cur_index=self.input_size
        self.perm=None
        self.category=category

    def _coor_read(self, cat, coor):
        '''
        description:用以将标签中的字符串三元组坐标转换为np.ndarray并返回之
        args:
            cat:类别
            coor:读取csv文件中的关节点部分
        '''
        ret = []
        for i, xyv in enumerate(coor):
            #忽略对应类别没有的关节点
            if xyv[0] == '-':
                if all_kpt_list[i] in categories_dict[cat]:
                    #对于相同类别下可能不存在的关节点直接置全零
                    ret.append(np.asarray([0, 0, 0], np.int32))
            else:
                #[3]
                xyv = np.asarray([int(i) for i in xyv.split('_')], np.int)
                ret.append(xyv)
        #[k,3]
        return np.asarray(ret)    
        
    def next(self):
        '''
        desc:处理好image和label并返回
        '''
        #多进程如何随机且均匀地读取？
        #各个子进程之间不共享局部变量和全局变量
        #返回一个随机序列，大小为input_size
        #轮完一个epoch
        if self.cur_index + self.batch_size > self.input_size:
            self.cur_index = 0
            self.perm=np.random.permutation(self.input_size)
        #未轮完一个epoch
        random_index=self.perm[self.cur_index:self.cur_index+self.batch_size]
        image_batch=[]
        label_batch=[]
        #data augment
        for i in range(self.batch_size):
            angle=np.random.uniform(-30.0,30.0)
            scale=np.random.uniform(0.75,1.25)
            row=self.file_list[random_index[i]].split(',')
            img_name = dir_para['train_data_dir']+ '/'+row[0] 
            img = cv2.imread(img_name)
            #转换为rgb
            img=img[...,::-1].astype(np.uint8)
            old_size=np.stack([img.shape[0],img.shape[1]])
            #resize到统一大小  默认使用双线性插值
            #dsize默认为先x后y
            img=cv2.resize(img,(self.resized_size[1],self.resized_size[0]))
            if train_para['is_train']:
                #[k,3]
                coor_orig= self._coor_read(self.category, row[2:])
                #[k,3] 
                #TODO  传入的应该是先y后x的坐标
                coor_yx=np.stack([coor_orig[:,1],coor_orig[:,0]],1)
                coor=coor_translate_np(coor_yx,old_size,self.resized_size)
                coor_xy=np.stack([coor[:,1],coor[:,0]],1)
                #coor=np.concatenate((coor,np.expand_dims(coor_orig[:,2],1)),1)
            else:
                pass
            #[size,h,w,3]
            ret=rotate_and_scale(img,angle,scale=scale)        
            image_batch.append(ret[1])
            #仿射变换矩阵
            #[3,2]
            M=np.transpose(ret[0],(1,0))
            # [24,3]的坐标修改 [x,y,1]
            coor_3d=np.concatenate((coor_xy,
            np.expand_dims(np.ones(coor_orig.shape[0]),1)),1)
            coor_3d=coor_3d.astype(np.float)
            #[24,2]
            coor_trans=np.matmul(coor_3d,M)
            coor_trans=np.floor(coor_trans).astype(np.int)
            #取第0维 即超出图片尺寸范围的索引
            out=np.concatenate((np.where(coor_trans>255)[0],np.where(coor_trans<0)[0]))
            #unique
            out=np.unique(out)
            #print(out)
            coor_trans=np.concatenate((coor_trans,np.expand_dims(coor_orig[:,2],1)),1)
            #超出图片范围的
            coor_trans[out]=0
            gt_heatmaps=make_gt_heatmaps(coor_trans[:,0:2],self.resized_size,1.0,coor_trans[:,2])
            #缩小到和网络输出相同尺寸 先x后y
            gt_heatmaps=cv2.resize(gt_heatmaps,(self.resized_size[1]//4,self.resized_size[0]//4))
            label_batch.append(gt_heatmaps)
        #
        batch=[image_batch,label_batch]
        self.cur_index+=self.batch_size
        return batch