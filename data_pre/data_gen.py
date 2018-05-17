#使用np及cv2而不是tf来进行数据的预处理和生成
#encoding=utf-8
from parallel_gen import GeneratorEnqueuer
import os
from ref import *
from coco_preprocess import COCOPreProcess


#batch生成器
def batch_generator(data_preprocess, slavers=10, max_queue_size=16):
    '''
    description：生成batch
    parameters:
        data_preprocess: object
            数据集预处理类，必须有get_batch_gen方法(linux下使用)和next方法(windows下使用)
        slavers: int
            线程数量
        max_epoch: int
            最大迭代轮数
    return: generator
        返回从队列中取得预处理batch的generator
    '''
    try:
        if os.name is 'posix':
            gen = data_preprocess.get_batch_gen()
        #for fuck windows
        elif os.name is 'nt':
            gen = data_preprocess
        else:
            raise ValueError('Unknown system.')
        gen_enq = GeneratorEnqueuer(gen)
        #产生了多个slaver并开始工作
        gen_enq.start(slaver=slavers, max_queue_size=max_queue_size)
        while True:
            #is_running/empty在get中已经判断了
            batch_output = gen_enq.get()
            #返回的也是一个生成器 所以需要next
            batch_output = next(batch_output)
            #能用生成器 表明父进程的这个类实例未被传给子进程？
            yield batch_output
            batch_output = None
    except Exception as e:
        raise e
        #if gen_enq is not None:
        #gen_enq.stop()
