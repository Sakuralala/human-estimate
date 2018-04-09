#新的train结构，使用多进程
import multiprocessing as mp
from np_data_gen import *
import tensorflow as tf
import numpy as np
import os
import sys
import scipy
import argparse  #用以解析运行时的参数
from hourglass_orgin import HourglassModel
from params_clothes import *


def train(category, csv_file):
    '''
    desc:用以进行训练。
    args:
        category:训练的服装类别。
        csv_file:源csv文件。
    '''
    #----------------------------------------session的一些配置--------------------------------------------------------
    #限制使用一个gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    gpu_options = tf.GPUOptions(allow_growth=True)
    run_metadata = tf.RunMetadata()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    config = tf.ConfigProto(
        log_device_placement=True,
        allow_soft_placement=True,
        gpu_options=gpu_options)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if not os.path.exists(dir_para['trained_model_dir']):
            os.mkdir(dir_para['trained_model_dir'])
            print('Created trained_model dir:', dir_para['trained_model_dir'])
        #------------------------------------读取源文件并预处理及取得batch--------------------------------------------#
        with open(csv_file, 'r') as f:
            file_list = f.readlines()[1:]
            if category != 'full':
                file_list = [elem for elem in file_list if category in elem]
        gen = clothes_batch_generater()
        batch = gen.generate_batch(category, file_list, 1, 8,
                                   train_para['batch_size'])

        #---------------------------------构建模型以及一些参数的设置--------------------------------------------------#
        #image
        x = tf.placeholder(
            tf.float32,
            shape=[train_para['batch_size'], 256, 256, 3],
            name='input')
        #label
        y = tf.placeholder(
            tf.float32,
            shape=[
                train_para['batch_size'], 64, 64,
                len(categories_dict['full'])
            ],
            name='gt_heatmaps')
        #Model
        model = HourglassModel(len(categories_dict['full']))
        #构建网络
        output = model.build_model(x)
        #loss计算
        loss = model.loss_calculate(y)
        #全局步数
        global_step = tf.Variable(0, trainable=False, name="global_step")
        #学习率
        lr = tf.train.exponential_decay(
            train_para['init_lr'],
            global_step,
            decay_steps=train_para['lr_decay_steps'],
            decay_rate=train_para['lr_decay_rate'],
            staircase=True)
        opt = tf.train.AdamOptimizer(lr)
        train_op = opt.minimize(loss, global_step=global_step)

        #---------------------------------可视化输出和输入-----------------------------------------------------------#
        with tf.variable_scope('input_image'):
            tf.summary.image('input_0', tf.expand_dims(x[0], 0), max_outputs=1)
        with tf.variable_scope('gt_and_pred'):
            for i in range(output[-1].get_shape()[-1]):
                #gt
                gt_heatmap0 = tf.expand_dims(y[0, :, :, i], -1)
                gt_heatmap0 = tf.expand_dims(gt_heatmap0, 0)
                #pred
                pred_heatmap0 = tf.expand_dims(output[-1][0, :, :, i], -1)
                pred_heatmap0 = tf.expand_dims(pred_heatmap0, 0)

                tf.summary.image(
                    'gt_heatmaps%s' % i, gt_heatmap0, max_outputs=1)
                tf.summary.image(
                    'pre_heatmaps%s' % i, pred_heatmap0, max_outputs=1)

        merged_op = tf.summary.merge_all()

        #writer
        writer = tf.summary.FileWriter(dir_para['log_dir'], sess.graph)
        #saver
        saver = tf.train.Saver(
            max_to_keep=0, keep_checkpoint_every_n_hours=4.0)
        ckpt = tf.train.get_checkpoint_state(dir_para['trained_model_dir'])
        #------------------------------------------读取预训练的模型--------------------------------------------------#
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('reload model.')
        else:
            print('No checkpoint file found.')
            # init weights
            sess.run(tf.global_variables_initializer())

        step = sess.run(global_step)
        #-----------------------------------------------训练的循环--------------------------------------------------#
        while True:
            try:
                #产生对应的batch
                image_batch, label_batch = next(batch)
                #print(image_batch,label_batch)
                _, output_final, loss_v, summ, step = sess.run(
                    [train_op, output[-1], loss, merged_op, global_step],
                    feed_dict={
                        x: image_batch,
                        y: label_batch
                    })  #,options=run_options,run_metadata=run_metadata)
                writer.add_summary(summ, step)
                #统计内存显存等使用信息
                #writer.add_run_metadata(run_metadata,'step %d'%step)
                if (step % train_para['show_loss_freq']) == 0:
                    print('Iteration %d\t Loss %.3e' % (step, loss_v))
                    '''
                    scipy.misc.imsave('.//outimg//image0.png', image_batch[0])
                    for i in range(output_final.shape[-1]):
                        scipy.misc.imsave('.//outimg//pre_4%d.png' % i,
                                          output_final[0, :, :, i])
                        scipy.misc.imsave('.//outimg//gt%d.png' % i,
                                          label_batch[0, :, :, i])
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

            except Exception as e:
                break

        print('Training %s finished. Saving final trained_model.' % category)
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

    train(args.category, dir_para['train_data_dir'] + '/Annotations/train.csv')


if __name__ == '__main__':
    mp.freeze_support()
    main()
