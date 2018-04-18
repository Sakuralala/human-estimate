import cv2
import numpy as np
from utils_new import *
from np_data_gen import clothes_batch_generater
from params_clothes import *
from multiprocessing import freeze_support
import multiprocessing as mp

class A():
    def __init__(self):
        self.l=[]
        self.mg=None
    
    def start(self,num):
        self.mg=mp.Manager()
        for i in range(num):
            #if I used self.mg=mp.Manager(),
            # and the target=self.fuck,then an error occured:OSError: handle is closed
            #but if the target=fuck,then it works fine.
            #if I don't use self.mg=mp.Manager(),then both work fine.
            p1=mp.Process(target=fuck)
            p1.daemon=True
            p1.start()

    def fuck(self):
        b=9

def fuck():
    b=10

if __name__=='__main__':
    freeze_support()
    '''
    a=A()
    a.start(5)
    '''
    cl_gen=clothes_batch_generater()
    with open(dir_para['train_data_dir']+'/Annotations/train.csv','r') as f:
        file_list=f.readlines()[1:]
        #file_list=[elem for elem in file_list if 'skirt' in elem]
    #print(len(file_list))
    #file_list=[1]
    gen=cl_gen.generate_batch('full',file_list)
    while True:
        batch=next(gen)
        #print(batch[0][0].shape)
        print(batch[0][0].shape)
        #img_ex=cv2.cvtColor(batch[0][0],cv2.COLOR_BGR2RGB)
        img_ex=batch[0][0]
        cv2.imshow('image',img_ex)

        lb_ex=batch[1][0][:,:,0]*255
        for i in range(1,batch[1][0].shape[-1]):
            lb_ex=cv2.addWeighted(lb_ex,1,batch[1][0][:,:,i]*255,1,0)
        #print(np.max(img_ex),np.max(lb_ex))
        cv2.imshow('label',lb_ex)
        print(img_ex.shape,lb_ex.shape)

        add_2=cv2.addWeighted(img_ex[:,:,0],0.4,lb_ex,0.6,0)
        cv2.imshow('add',add_2)
        cv2.waitKey(0)
