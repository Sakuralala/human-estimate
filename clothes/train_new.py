#新的train结构，使用多进程
import multiprocessing as mp 
from np_data_gen import *
import tensorflow as tf
import numpy as np
import os
import sys
#import argparse  #用以解析运行时的参数
#from clothes_data_generate import *
from hourglass_orgin import HourglassModel
from params_clothes import *


def train(category,csv_file):
    #限制使用一个gpu
    os.environ["CUDA_VISIBLE_DEVICES"]='1'
    gpu_options = tf.GPUOptions(allow_growth=True)
    run_metadata=tf.RunMetadata()
    run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True,gpu_options=gpu_options)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # 训练好的模型参数
        if not os.path.exists(dir_para['trained_model_dir']):
            os.mkdir(dir_para['trained_model_dir'])
            print('Created trained_model dir:', dir_para['trained_model_dir'])
        #源csv文件
        with open(csv_file,'r') as f:
            file_list=f.readlines()[1:]
        gen=clothes_batch_generater()
        #生成generator
        batch=gen.generate_batch(category,file_list,4,8,train_para['batch_size'])

        #为了之后test能够喂进不同的数据
        #image
        x=tf.placeholder(tf.float32,shape=[train_para['batch_size'],256,256,3],name='input')
        #label
        y=tf.placeholder(tf.float32,
        shape=[train_para['batch_size'],64,64,len(categories_dict[category])],name='gt_heatmaps')
        #Model
        model = HourglassModel(len(categories_dict[category]))
        #构建网络
        output=model.build_model(x)
        #loss计算
        loss=model.loss_calculate(y)
        #全局步数
        global_step = tf.Variable(0, trainable=False, name="global_step")
        # Solver
        lr = tf.train.exponential_decay(
            train_para['init_lr'],
            global_step,
            decay_steps=train_para['lr_decay_steps'],
            decay_rate=train_para['lr_decay_rate'])
        opt = tf.train.AdamOptimizer(lr)
        train_op = opt.minimize(loss, global_step=global_step)
        merged_op = tf.summary.merge_all()

        #writer
        writer = tf.summary.FileWriter(dir_para['log_dir'], sess.graph)
        #saver
        saver = tf.train.Saver(
            max_to_keep=0, keep_checkpoint_every_n_hours=4.0)
        ckpt = tf.train.get_checkpoint_state(dir_para['trained_model_dir'])
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('reload model.')
        else:
            print('No checkpoint file found.')
            # init weights
            sess.run(tf.global_variables_initializer())
        #sess.run(tf.global_variables_initializer())
        step = sess.run(global_step)
        # Training loop
        while True:
            try:
                #产生对应的batch
                image_batch,label_batch=next(batch)
                _, loss_v, summ, step = sess.run(
                    [train_op, loss, merged_op, global_step],
                    feed_dict={x:image_batch,y:label_batch})#,options=run_options,run_metadata=run_metadata)
                writer.add_summary(summ, step)
                #统计内存显存等使用信息
                #writer.add_run_metadata(run_metadata,'step %d'%step)
                if (step % train_para['show_loss_freq']) == 0:
                    print('Iteration %d\t Loss %.3e' % (step, loss_v))
                    '''
                    cv2.imwrite('.//outimg//image0.png',image_batch[0])
                    gt_heatmaps=label_batch[0][:,:,0]
                    cv2.imwrite('.//outimg//pre0.png',output_v3[0,:,:,0])
                    print(output_v3.shape,np.max(output_v3[0]))
                    for i in range(1,output_v3.shape[-1]):
                        #叠加的gt_heatmaps
                        gt_heatmaps=cv2.addWeighted(gt_heatmaps,0.5,label_batch[0][:,:,i],0.5,0) 
                        cv2.imwrite('.//outimg//pre%s.png'%i,output_v3[0,:,:,i])
                    cv2.imwrite('.//outimg//gt.png',gt_heatmaps)
                    '''
                if (step % train_para['show_lr_freq']) == 0:
                    print('Current lr:', sess.run(lr))
                    sys.stdout.flush()

                if (step % train_para['save_freq']) == 0:
                    saver.save(
                        sess,
                        "%s/%s" % (dir_para['trained_model_dir'], category),
                        global_step=step)
                    print('Saved a trained_model.')
                    sys.stdout.flush()
            except tf.errors.OutOfRangeError:
                break

        print('Training %s finished. Saving final trained_model.'%category)
        saver.save(
            sess,
            "%s/%s" % (dir_para['trained_model_dir'], category),
            global_step=train_para['max_iter'])


def main():
    #parser = argparse.ArgumentParser()
    #后面必须跟一个服装类别的参数
    #parser.add_argument('category', type=str, choices=categories)
    #返回Namespace对象
    #args = parser.parse_args()
    #print(args.category)
    #一次只能训练一个类别
    #train(args.category)

    
    '''
    for cat in categories:
        print('Start to train %s.'%cat)
        train(cat)
    '''
    
    train('full',dir_para['train_data_dir']+'/Annotations/train.csv')
    
    #train('blouse')
    #train('full')


if __name__ == '__main__':
    mp.freeze_support()
    main()
