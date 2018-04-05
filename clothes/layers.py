#often used layers

import tensorflow as tf
import numpy as np
import os
from params_clothes import *


def _res18_block(input, num_outputs, size=3, stride=1, name='res18_block'):
    with tf.variable_scope(name):
        bn1 = tf.contrib.layers.batch_norm(
            input,
            epsilon=1e-5,
            decay=layer_para['bn_decay'],
            activation_fn=tf.nn.relu)
        conv1 = tf.contrib.layers.conv2d(
            bn1, num_outputs, size, stride, activation_fn=None)
        bn2 = tf.contrib.layers.batch_norm(
            conv1,
            epsilon=1e-5,
            decay=layer_para['bn_decay'],
            activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            bn2, num_outputs, size, 1, activation_fn=None)
        if tf.shape(input)[-1] != num_outputs:
            trans = tf.contrib.layers.conv2d(
                input, num_outputs, 1, stride, activation_fn=None)
            output = trans + conv2
        else:
            output = conv2 + input
        return output


#stn中的location net
def res18_loc(input, name='localization_net'):
    with tf.variable_scope(name):
        bn1 = tf.contrib.layers.batch_norm(
            input,
            epsilon=1e-5,
            decay=layer_para['bn_decay'],
            activation_fn=tf.nn.relu)
        #input num_outputs size stride
        conv1 = tf.contrib.layers.conv2d(bn1, 64, 7, 2, activation_fn=None)
        #input size stride padding
        pool = tf.contrib.layers.max_pool2d(conv1, 3, 2, 'SAME')
        conv2 = _res18_block(pool, 64, name='res1')
        conv3 = _res18_block(conv2, 128, stride=2, name='res2')
        conv4 = _res18_block(conv3, 256, stride=2, name='res3')
        conv5 = _res18_block(conv4, 512, stride=2, name='res4')
        #[8,?,?,512]
        #[8,x,x,512]
        # shape=tf.shape(conv5)
        conv5_flatten = tf.reshape(conv5, [train_para['batch_size'], -1])
        dim = conv5_flatten.get_shape()[-1]
        #print(dim)
        #print(shape[0])
        #print(conv5.get_shape())
        w1 = _variable_with_weight_decay('w1', [dim, 1024])
        local1 = tf.matmul(conv5_flatten, w1)
        #print(local1)
        local1 = tf.contrib.layers.bias_add(local1, activation_fn=tf.nn.relu)
        w2 = _variable_with_weight_decay('w2', [1024, 6])
        output = tf.contrib.layers.bias_add(tf.matmul(local1, w2))
        #print(output.shape)
        #输出转换矩阵参数 [batch,6]
        #output=tf.layers.dense(local1,6,name='theta')
        #为了使用tf.contrib.layers.transform,加两个
        #[batch,2]
        tmp = tf.tile(
            tf.expand_dims(tf.stack([0.0, 0.0]), 0),
            [train_para['batch_size'], 1])
        #[b,2]
        #print(tmp.shape)
        #[batch,8]
        output = tf.concat([output, tmp], -1)
        #[b,8]
        #print(output.shape)
        return output


#hourglass中的残差模块的identity mapping 的升级优化版
#m为一超参数，c为每个prm中的level数
def PRMA_block(input,
               out_channels,
               use_conv=False,
               m=1,
               c=4,
               name='PRMA_block'):
    with tf.variable_scope(name):
        f_list = []
        height = tf.shape(input)[1]
        width = tf.shape(input)[2]
        #f0
        for i in range(0, c + 1):
            bn1 = tf.contrib.layers.batch_norm(
                input,
                epsilon=1e-5,
                decay=layer_para['bn_decay'],
                activation_fn=tf.nn.relu)
            conv1 = tf.contrib.layers.conv2d(
                bn1, out_channels // 2, 1, activation_fn=None)
            ratio = 2.0**(m * i / c)
            if ratio > 1.0:
                conv1 = tf.nn.fractional_max_pool(conv1,
                                                  [1.0, ratio, ratio, 1.0])
            bn2 = tf.contrib.layers.batch_norm(
                conv1,
                epsilon=1e-5,
                decay=layer_para['bn_decay'],
                activation_fn=tf.nn.relu)
            conv2 = tf.contrib.layers.conv2d(
                bn2, out_channels // 2, 3, activation_fn=None)
            if ratio > 1.0:
                conv2 = tf.image.resize_bilinear(conv2, [height, width])
            f_list.append(conv2)

        bn3 = tf.contrib.layers.batch_norm(
            f_list[0],
            epsilon=1e-5,
            decay=layer_para['bn_decay'],
            activation_fn=tf.nn.relu)
        f0 = tf.contrib.layers.conv2d(bn3, out_channels, 1, activation_fn=None)
        fi = tf.add_n([f_list[i] for i in range(1, c + 1)], 'fi_total')
        fi = tf.contrib.layers.batch_norm(
            fi,
            epsilon=1e-5,
            decay=layer_para['bn_decay'],
            activation_fn=tf.nn.relu)
        fi = tf.contrib.layers.conv2d(fi, out_channels, 1, activation_fn=None)
        fi += f0
        #identify mapping
        if use_conv or tf.shape(input)[-1] != out_channels:
            trans = tf.contrib.layers.conv2d(input, out_channels, 1)
            output = tf.add(fi, trans, 'output')
        else:
            output = tf.add(fi, input, 'output')

        return output


#use_conv  用以控制残差模块的输出方差
def PRMB_block(input,
               out_channels,
               use_conv=False,
               name='PRMB_block',
               m=1,
               c=4):
    with tf.variable_scope(name):
        f_list = []
        height = tf.shape(input)[1]
        width = tf.shape(input)[2]
        #f0 权重不和fc共享
        f0 = tf.contrib.layers.batch_norm(
            input,
            epsilon=1e-5,
            decay=layer_para['bn_decay'],
            activation_fn=tf.nn.relu)
        f0 = tf.contrib.layers.conv2d(
            f0, out_channels // 2, 1, activation_fn=None)
        f0 = tf.contrib.layers.batch_norm(
            f0,
            epsilon=1e-5,
            decay=layer_para['bn_decay'],
            activation_fn=tf.nn.relu)
        f0 = tf.contrib.layers.conv2d(
            f0, out_channels // 2, 3, activation_fn=None)
        f0 = tf.contrib.layers.batch_norm(
            f0,
            epsilon=1e-5,
            decay=layer_para['bn_decay'],
            activation_fn=tf.nn.relu)
        f0 = tf.contrib.layers.conv2d(f0, out_channels, 3, activation_fn=None)
        #fi
        bn1 = tf.contrib.layers.batch_norm(
            input,
            epsilon=1e-5,
            decay=layer_para['bn_decay'],
            activation_fn=tf.nn.relu)
        conv1 = tf.contrib.layers.conv2d(
            bn1, out_channels // 2, 1, activation_fn=None)
        for i in range(1, c + 1):
            ratio = float(2.0**(m * i / c))
            #print(ratio)
            with tf.variable_scope('sharing_weights', reuse=tf.AUTO_REUSE):
                pool, row_pooling_seq, col_pooling_seq = tf.nn.fractional_max_pool(
                    conv1, [1.0, ratio, ratio, 1.0])
                bn2 = tf.contrib.layers.batch_norm(
                    pool,
                    epsilon=1e-5,
                    decay=layer_para['bn_decay'],
                    activation_fn=tf.nn.relu)

                conv2 = conv_block(
                    bn2,
                    3,
                    1,
                    out_channels // 2,
                    do_normalization=False,
                    do_RELU=False)
            resized = tf.image.resize_bilinear(conv2, [height, width])
            f_list.append(resized)

        fi = tf.add_n(f_list, 'fi_total')
        fi = tf.contrib.layers.batch_norm(
            fi,
            epsilon=1e-5,
            decay=layer_para['bn_decay'],
            activation_fn=tf.nn.relu)
        fi = tf.contrib.layers.conv2d(fi, out_channels, 1, activation_fn=None)
        fi += f0
        #identify mapping
        if use_conv or tf.shape(input)[-1] != out_channels:
            trans = tf.contrib.layers.conv2d(input, out_channels, 1)
            output = tf.add(fi, trans, 'output')
        else:
            output = tf.add(fi, input, 'output')

        return output


def residual_block(input, out_channels, use_conv=False, name='residual_block'):
    with tf.variable_scope(name):
        bn1 = tf.contrib.layers.batch_norm(
            input,
            epsilon=1e-5,
            decay=layer_para['bn_decay'],
            scale=True,
            activation_fn=tf.nn.relu)
        conv1 = tf.contrib.layers.conv2d(
            bn1, out_channels // 2, 1, activation_fn=None)
        bn2 = tf.contrib.layers.batch_norm(
            conv1,
            epsilon=1e-5,
            decay=layer_para['bn_decay'],
            scale=True,
            activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            bn2, out_channels // 2, 3, activation_fn=None)
        bn3 = tf.contrib.layers.batch_norm(
            conv2,
            epsilon=1e-5,
            decay=layer_para['bn_decay'],
            scale=True,
            activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            bn3, out_channels, 1, activation_fn=None)

        #identify mapping
        if use_conv or tf.shape(input)[-1] != out_channels:
            #tensorflow/model 若要进行1x1卷积操作，则输入应该为经过pre-activation的
            trans = tf.contrib.layers.conv2d(bn1, out_channels, 1)
            output = tf.add(conv3, trans, 'output')
        else:
            output = tf.add(conv3, input, 'output')

        return output


def conv_block_new(input,
                   output_num,
                   kernel_size,
                   stride,
                   name='conv',
                   padding='SAME',
                   do_normalization=True,
                   do_relu=True):
    norm_fn = tf.contrib.layers.batch_norm if do_normalization == True else None
    norm_paras = {
        'decay': 0.99,
        'center': True,
        'scale': True,
        'epsilon': 1e-3
    }
    activation_fn = tf.nn.relu if do_relu == True else None
    output = tf.contrib.layers.conv2d(
        input,
        output_num,
        kernel_size,
        stride,
        activation_fn=activation_fn,
        normalizer_fn=norm_fn,
        normalizer_params=norm_paras
    )

    return output


# 简单封装的卷积层 padding模式默认为same
def conv_block(input,
               kernel_size,
               stride,
               kernel_number,
               name='conv',
               padding='SAME',
               do_normalization=True,
               do_RELU=True):
    with tf.variable_scope(name):
        weights = _variable_with_weight_decay(
            name,
            [kernel_size, kernel_size,
             input.get_shape()[-1], kernel_number])
        conv = tf.nn.conv2d(
            input,
            weights, [1, stride, stride, 1],
            padding=padding,
            name='conv')
        #name BiasAdd:0 'hourglass_model/hourglass4/inter_output/inter_output:0', 'hourglass_model/hourglass4/inter_output/BiasAdd/biases:0'
        output = tf.contrib.layers.bias_add(conv)
        if do_normalization:
            output = tf.contrib.layers.batch_norm(
                output, epsilon=1e-5, decay=layer_para['bn_decay'])
        if do_RELU:
            output = tf.nn.relu(output, name='activated_output')
        return output


#deprecated
# 归一化  在激活前做
def batch_normalization(input,
                        decay,
                        epsilon,
                        name='batch_normalization',
                        is_train=True):
    mean, variance = tf.nn.moments(input, [0, 1, 2])
    scale = tf.Variable(tf.ones([1]))
    offset = tf.Variable(tf.zeros([1]))

    #训练时更新mean和variance
    ema = tf.train.ExponentialMovingAverage(decay)

    def mean_var_update():
        ema_apply_op = ema.apply([mean, variance])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(mean), tf.identity(variance)

    #训练时更新，测试时使用平均值
    mean, variance = tf.cond(
        tf.cast(is_train, tf.bool), mean_var_update,
        lambda: (ema.average(mean), ema.average(variance)))
    normalized = tf.nn.batch_normalization(input, mean, variance, offset,
                                           scale, epsilon, name)
    return normalized


# tf.get_variable的封装
def _variable_with_weight_decay(name, shape, uniform=False):
    with tf.device('/cpu:0'):
        return tf.get_variable(
            name,
            shape,
            #TODO  多branch的初始化
            initializer=tf.contrib.layers.xavier_initializer(uniform))


def max_pool(input, kernel_size=3, stride=2, name='max_pool'):
    return tf.nn.max_pool(
        input,
        ksize=[1, kernel_size, kernel_size, 1],
        strides=[1, stride, stride, 1],
        padding='SAME',
        name=name)
