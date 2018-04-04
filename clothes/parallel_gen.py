#多进程/线程生成批量数据
import time
import numpy as np
#用于多进程 默认使用这个
import multiprocessing as mp
#下面两个用以多线程
import threading
import queue as q

#参考 from https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py
class GeneratorEnqueuer():
    '''
    desc:多进程生成器处理队列
    args:
        generator:具体使用的生成器,预处理好的数据在该函数中生成
        use_mp:是否使用多进程，默认为True
        wait_time:用于在不断从队列获取数据时队列为空的情况，此时master进程/线程应该睡眠的时间
        seed:随机数种子
    '''

    def __init__(self, generator, use_mp=True, wait_time=0.05, seed=None):
        #这些东西是否也会被复制给子进程呢？
        #Manager、Queue、Event返回的是一个proxy对象，可以理解为真正对象的引用。
        self.generator = generator
        self.use_mp = use_mp
        self.wait_time = wait_time
        self.seed = seed
        #用于同步
        self.stop_event = None
        #多线程
        self.queue = None
        self.threads = []
        #多进程
        self.manager = None

    def _task_(self):
        #slaver进程/线程一直等待master进程/线程发送终止信号
        while not self.stop_event.is_set():
            if self.use_mp == True:
                try:
                    #generated_output = next(self.generator)
                    #Q:此处generator是否会被独立复制多次？
                    #A:spawn方法  子进程仅继承父进程的必要资源(但是，什么是必要资源呢)
                    generated_output=self.generator.next()
                    #put是否为原子性操作？
                    #是原子操作  详见官方文档:Queue->Queue->deque
                    #创建的queue是共享的
                    self.queue.put(generated_output)
                #迭代结束
                except StopIteration:
                    break
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    #在队列中加入异常
                    self.queue.put(e)
                    self.stop_event.set()
                    break
            else:
                #线程版(TODO,对计算密集型效率极低懒得写了)
                pass

    def start(self, slaver=6, max_queue_size=10):
        '''
        desc:开始生产
        args:
            slaver:进程/线程数量
            max_queue_size:最大队列数
        '''
        #note:win下不要用多进程，会报错，因为generator不可被pickle。
        #或者把generator改为iterator
        try:
            if self.use_mp:
                #self.manager = mp.Manager()
                #self.queue = self.manager.Queue(max_queue_size)
                self.queue=mp.Queue(max_queue_size)
                self.stop_event = mp.Event()
            else:  #多线程
                pass
            for i in range(slaver):
                if self.use_mp:
                    np.random.seed(self.seed)
                    #子进程
                    process = mp.Process(target=self._task_)
                    process.daemon = True  #转换为后台进程
                    if self.seed != None:
                        self.seed += 1
                    #类似句柄 以便后续终止进程
                    self.threads.append(process)
                else:
                    pass
                process.start()
        except Exception:
            self.stop()
            raise

    def is_running(self):
        '''
        desc:判断当前是否在生产
        '''
        return self.stop_event is not None and not self.stop_event.is_set()

    def stop(self, max_wait_time=None):
        '''
        desc:停止生产
        args:
            max_wait_time:最大等待join时间
        '''
        if self.is_running():
            #首先master设置flag
            self.stop_event.set()

        for proc in self.threads:
            if self.use_mp:
                if proc.is_alive():
                    proc.terminate()
            #proc.join()

        if self.manager:
            self.manager.shutdown()
        self.queue = None
        self.threads = []
        self.stop_event = None

    def get(self):
        '''
        desc:从队列中获取已经处理好的数据
        '''
        while self.is_running():
            if not self.queue.empty():
                #是原子操作  详见官方文档:Queue->Queue->deque
                val=self.queue.get()
                if val is not None:
                    yield val
            else:
                time.sleep(self.wait_time)