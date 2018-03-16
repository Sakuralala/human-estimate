#使用hourglass结构来进行关节点预测
import tensorflow as tf
import scipy
import numpy as np
import os
import sys
import argparse  #用以解析运行时的参数
from clothes_data_generate import *
from hourglass import HourglassModel
from params_clothes import *


def train(category):
    #限制使用一个gpu
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    gpu_options = tf.GPUOptions(allow_growth=True)
    config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # 训练好的模型参数
        if not os.path.exists(dir_para['trained_model_dir']):
            os.mkdir(dir_para['trained_model_dir'])
            print('Created trained_model dir:', dir_para['trained_model_dir'])
        gen=clothes_batch_generater(category,train_para['batch_size'])
	# 用以作为输入的二进制文件
        if not os.path.exists(dir_para['tf_dir']):
            os.mkdir(dir_para['tf_dir'])
            print('Created tfrecords dir:', dir_para['tf_dir'])
        rec_name=category+'.tfrec'
        for (root,dirs,files) in os.walk(dir_para['tf_dir']):
            if rec_name not in files:
                gen.convert_to_tfrecords_single(category,dir_para['train_data_dir'],dir_para['tf_dir'])
	#生成batch
        batch=gen.dataset_input_new(dir_para['tf_dir']+'/'+rec_name)
	# Start TF
        tf.train.start_queue_runners(sess=sess)
        #为了之后test能够喂进不同的数据
        x=tf.placeholder(tf.float32,shape=[None,None,None,3],name='input')
        #Model
        model = HourglassModel(1+len(categories_dict[category]))
        #构建网络
        output=model.build_model(x)
        #loss计算
        loss=model.loss_calculate(batch[1])
        #全局步数
        global_step = tf.Variable(0, trainable=False, name="global_step")
        # Solver
        lr = tf.train.exponential_decay(
            train_para['init_lr'],
            global_step,
            decay_steps=train_para['lr_decay_steps'],
            decay_rate=train_para['lr_decay_rate'])
        #lr=train_para['init_lr']
        opt = tf.train.AdamOptimizer(lr)
        #opt = tf.train.RMSPropOptimizer(lr,decay=0.99,epsilon=1e-8)
        train_op = opt.minimize(loss, global_step=global_step)
        #summar
        merged_op = tf.summary.merge_all()

        #writer
        writer = tf.summary.FileWriter(dir_para['log_dir'], sess.graph)
        #saver
        saver = tf.train.Saver(
            max_to_keep=0, keep_checkpoint_every_n_hours=4.0)
        '''
        ckpt = tf.train.get_checkpoint_state(dir_para['trained_model_dir'])
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('reload model.')
        else:
            print('No checkpoint file found.')
            # init weights
            sess.run(tf.global_variables_initializer())
        '''
        sess.run(tf.global_variables_initializer())
        step = sess.run(global_step)
        # Training loop
        while step < train_para['max_iter']:
            image_batch,gt_heatmap_batch=sess.run([batch[0],batch[1]])
            _, loss_v,output_v3, summ, step = sess.run(
                [train_op, loss, output[3],merged_op, global_step],feed_dict={x:image_batch})
            writer.add_summary(summ, step)
            if (step % train_para['show_loss_freq']) == 0:
                print('Iteration %d\t Loss %.3e' % (step, loss_v))
                scipy.misc.imsave('image0.png',image_batch[0,:,:,:])
                for i in range(output_v3.shape[-1]-1):
                      #  scipy.misc.imsave('pre_4%d.png'%i,output_v4[0,:,:,i])
                        scipy.misc.imsave('pre_7%d.png'%i,output_v3[0,:,:,i])
                        scipy.misc.imsave('gt%d.png'%i,gt_heatmap_batch[0,:,:,i])
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

        print('Training %s finished. Saving final trained_model.'%category)
        saver.save(
            sess,
            "%s/%s" % (dir_para['trained_model_dir'], category),
            global_step=train_para['max_iter'])
        #清除图中节点
    #tf.reset_default_graph()


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
    train('skirt')
    #train('full')


if __name__ == '__main__':
    main()
