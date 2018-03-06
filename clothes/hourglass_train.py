#使用hourglass结构来进行关节点预测
from __future__ import print_function, unicode_literals

import tensorflow as tf
import os
import sys
import argparse  #用以解析运行时的参数
from clothes_data_generate import *
from hourglass import HourglassModel
from params_clothes import *


def train(category):
    # 训练好的模型参数
    if not os.path.exists(dir_para['trained_model_dir']):
        os.mkdir(dir_para['trained_model_dir'])
        print('Created trained_model dir:', dir_para['trained_model_dir'])
    gen=clothes_batch_generater(train_para['batch_size'])
    # 用以作为输入的二进制文件
    if not os.path.exists(dir_para['tf_dir']):
        os.mkdir(dir_para['tf_dir'])
        print('Created tfrecords dir:', dir_para['tf_dir'])
    rec_name=category+'.tfrec'
    for (root,dirs,files) in os.walk(dir_para['tf_dir']):
        if rec_name not in files:
            gen.convert_to_tfrecords_single(dir_para['train_data_dir'],dir_para['tf_dir'])
    #生成batch
    batch=gen.batch_generate(dir_para['tf_dir']+'/'+rec_name)
    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.train.start_queue_runners(sess=sess)

        #Model
        model = HourglassModel(len(categories_dict[category]))
        model.build_model(batch[0], batch[1])

        loss = sum(model.loss)
        #全局步数
        global_step = tf.Variable(0, trainable=False, name="global_step")
        # Solver
        lr = tf.train.exponential_decay(
            train_para['init_lr'],
            global_step,
            decay_steps=train_para['lr_decay_steps'],
            decay_rate=train_para['lr_decay_rate'])
        #opt = tf.train.AdamOptimizer(lr, 0.9, 0.999)
        opt = tf.train.RMSPropOptimizer(lr, epsilon=1e-8)
        #opt=tf.train.AdadeltaOptimizer(lr)
        train_op = opt.minimize(loss, global_step=global_step)
        #summary
        merged_op = tf.summary.merge_all()

        #writer
        writer = tf.summary.FileWriter(dir_para['log_dir'], sess.graph)
        #saver
        saver = tf.train.Saver(
            max_to_keep=1, keep_checkpoint_every_n_hours=4.0)
        ckpt = tf.train.get_checkpoint_state(dir_para['trained_model_dir'])
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('reload model.')
        else:
            print('No checkpoint file found.')
            # init weights
            sess.run(tf.global_variables_initializer())
        step = sess.run(global_step)
        # Training loop
        while step < train_para['max_iter']:
            _, loss_v, summ, step = sess.run(
                [train_op, loss, merged_op, global_step])
            writer.add_summary(summ, step)
            #if (i % train_para['show_loss_freq']) == 0:
            print('Iteration %d\t Loss %.3e' % (step, loss_v))
            if (step % train_para['show_lr_freq']) == 0:
                print('Current lr:', sess.run(lr))
            sys.stdout.flush()

            if (step % train_para['trained_model_freq']) == 0:
                saver.save(
                    sess,
                    "%s/%s" % (train_para['trained_model_dir'], category),
                    global_step=step)
                print('Saved a trained_model.')
                sys.stdout.flush()

        print('Training finished. Saving final trained_model.')
        saver.save(
            sess,
            "%s/%s" % (dir_para['trained_model_dir'], category),
            global_step=train_para['max_iter'])


def main():
    parser = argparse.ArgumentParser()
    #后面必须跟一个服装类别的参数
    parser.add_argument('category', type=str, choices=categories)
    #返回Namespace对象
    args = parser.parse_args()
    print(args.category)
    #一次只能训练一个类别
    #train(args.category)


if __name__ == '__main__':
    main()