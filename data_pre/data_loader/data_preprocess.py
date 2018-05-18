import numpy as np


#--------------------------------------------数据预处理父类-----------------------------------------------------------
class DataPreProcess():
    def __init__(self,
                 img_dir,
                 resized_size=(192, 256),
                 is_train=True,
                 is_multi=False,
                 batch_size=12,
                 max_epoch=200):
        '''
        description:数据预处理的父类，写这个的目的当前暂时是减少冗余代码，提高代码重用率。
        parameters:
            img_dir: str
                图片路径
            resized_size: tuple
                宽 高
            is_train: bool
                模式
            is_multi: bool
                top-down or bottom-up
            batch_size: int 
                批大小
            max_epoch: int
                最大轮数
        '''
        self.img_dir = img_dir
        self.resized_size = resized_size
        self.is_train = is_train
        self.is_multi = is_multi
        self.batch_size = batch_size
        self.cur_index = 0
        self.cur_epoch = 0
        #下列两个参数继承的子类需要根据实际情况更改
        self.data_size = 10000
        self.perm = None
        if self.is_train:
            self.max_epoch = max_epoch
        else:
            self.max_epoch = 1

    #linux下调这个
    def get_batch_gen(self):
        '''
        description: 得到一个预处理好的数据batch的generator
        parameters:
            batch_size: int
                batch大小
            max_epoch: int 
                最大迭代轮数
        return: generator
            batch generator
        '''
        try:
            while True:
                ret = self.get_batch()
                yield ret
        except Exception as e:
            raise e

    #windows下调这个
    def next(self):
        '''
        description:for fuck windows
        parameters:
        return: list
            一个预处理好的数据batch
        '''
        return self.get_batch()

    def get_batch(self):
        '''
        description:具体数据的处理，继承这个类的子类必须override这个方法
        '''
        pass

    def _update_cur_index(self):
        '''
        description:检测当前的索引及epoch是否需要重置(一个epoch完了的话)并在到达最大epoch后raise
        parameters:
            none
        return: ndarray
            返回一组index
        '''
        if self.perm is None:
            self.perm = np.random.permutation(self.data_size)
        if self.cur_index + self.batch_size > self.data_size:
            self.perm = np.random.permutation(self.data_size)
            self.cur_index = 0
            self.cur_epoch += 1
            if self.cur_epoch == self.max_epoch:
                if self.is_train:
                    raise ValueError('Train finished.')
                else:
                    raise ValueError('Test finished.')
        index = self.perm[self.cur_index:self.cur_index + self.batch_size]
        self.cur_index += self.batch_size

        return index
