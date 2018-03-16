#often used layers

import tensorflow as tf
import numpy as np
import os
from params_clothes import *

#hourglass中的残差模块的identity mapping 的升级优化版
#m为一超参数，c为每个prm中的level数
def PRMA_block(input,out_channels,use_conv=False,m=1,c=4,name='PRMA_block'):
    with tf.variable_scope(name):
        f_list=[]
        height=tf.shape(input)[1]
        width=tf.shape(input)[2]
        #f0
        for i in range(0,c+1):
            bn1=tf.contrib.layers.batch_norm(input,epsilon=1e-5,decay=layer_para['bn_decay'],activation_fn=tf.nn.relu)
            conv1=tf.contrib.layers.conv2d(bn1,out_channels//2,1,activation_fn=None)
            ratio=2.0**(m*i/c)
            if ratio>1.0:
                conv1=tf.nn.fractional_max_pool(conv1,[1.0,ratio,ratio,1.0])
            bn2=tf.contrib.layers.batch_norm(conv1,epsilon=1e-5,decay=layer_para['bn_decay'],activation_fn=tf.nn.relu)
            conv2=tf.contrib.layers.conv2d(bn2,out_channels//2,3,activation_fn=None)
            if ratio>1.0:
                conv2=tf.image.resize_bilinear(conv2,[height,width])
            f_list.append(conv2)

        bn3=tf.contrib.layers.batch_norm(f_list[0],epsilon=1e-5,decay=layer_para['bn_decay'],activation_fn=tf.nn.relu)
        f0=tf.contrib.layers.conv2d(bn3,out_channels,1,activation_fn=None)
        fi=tf.add_n([f_list[i] for i in range(1,c+1)],'fi_total')     
        fi=tf.contrib.layers.batch_norm(fi,epsilon=1e-5,decay=layer_para['bn_decay'],activation_fn=tf.nn.relu)
        fi=tf.contrib.layers.conv2d(fi,out_channels,1,activation_fn=None)
        fi+=f0
        #identify mapping  
        if use_conv or tf.shape(input)[-1]!=out_channels:
            trans=tf.contrib.layers.conv2d(input,out_channels,1)
            output=tf.add(fi,trans,'output')
        else:
            output=tf.add(fi,input,'output')

        return output


#use_conv  用以控制残差模块的输出方差
def PRMB_block(input,out_channels,use_conv=False,name='PRMB_block',m=1,c=4):
    with tf.variable_scope(name):
        f_list=[]
        height=tf.shape(input)[1]
        width=tf.shape(input)[2]
        #f0 权重不和fc共享
        f0=tf.contrib.layers.batch_norm(input,epsilon=1e-5,decay=layer_para['bn_decay'],activation_fn=tf.nn.relu)
        f0=tf.contrib.layers.conv2d(f0,out_channels//2,1,activation_fn=None)
        f0=tf.contrib.layers.batch_norm(f0,epsilon=1e-5,decay=layer_para['bn_decay'],activation_fn=tf.nn.relu)
        f0=tf.contrib.layers.conv2d(f0,out_channels//2,3,activation_fn=None)
        f0=tf.contrib.layers.batch_norm(f0,epsilon=1e-5,decay=layer_para['bn_decay'],activation_fn=tf.nn.relu)
        f0=tf.contrib.layers.conv2d(f0,out_channels,3,activation_fn=None)
        #fi
        bn1=tf.contrib.layers.batch_norm(input,epsilon=1e-5,decay=layer_para['bn_decay'],activation_fn=tf.nn.relu)
        conv1=tf.contrib.layers.conv2d(bn1,out_channels//2,1,activation_fn=None)
        for i in range(1,c+1):
            ratio=float(2.0**(m*i/c))
            #print(ratio)
            pool,row_pooling_seq,col_pooling_seq=tf.nn.fractional_max_pool(conv1,[1.0,ratio,ratio,1.0])
            bn2=tf.contrib.layers.batch_norm(pool,epsilon=1e-5,decay=layer_para['bn_decay'],activation_fn=tf.nn.relu)
            with tf.variable_scope('sharing_weights',reuse=tf.AUTO_REUSE):
               #conv2=tf.contrib.layers.conv2d(bn2,out_channels//2,3,activation_fn=None)
               conv2=conv_block(bn2,3,1,out_channels//2,do_normalization=False,do_RELU=False)
            resized=tf.image.resize_bilinear(conv2,[height,width])
            f_list.append(resized)

        fi=tf.add_n(f_list,'fi_total')     
        fi=tf.contrib.layers.batch_norm(fi,epsilon=1e-5,decay=layer_para['bn_decay'],activation_fn=tf.nn.relu)
        fi=tf.contrib.layers.conv2d(fi,out_channels,1,activation_fn=None)
        fi+=f0
        #identify mapping  
        if use_conv or tf.shape(input)[-1]!=out_channels:
            trans=tf.contrib.layers.conv2d(input,out_channels,1)
            output=tf.add(fi,trans,'output')
        else:
            output=tf.add(fi,input,'output')

        return output

def residual_block2(input,out_channels,name='residual_block'):
    with tf.variable_scope(name):
        bn1=tf.contrib.layers.batch_norm(input,epsilon=1e-5,decay=layer_para['bn_decay'],activation_fn=tf.nn.relu)
        conv1=tf.contrib.layers.conv2d(bn1,out_channels//2,1,activation_fn=None)
        #conv1=conv_block(bn1,1,1,out_channels//2,'conv1',do_normalization=False,do_RELU=False)
        bn2=tf.contrib.layers.batch_norm(conv1,epsilon=1e-5,decay=layer_para['bn_decay'],activation_fn=tf.nn.relu)
        conv2=tf.contrib.layers.conv2d(bn2,out_channels//2,3,activation_fn=None)
        #conv2=conv_block(bn2,1,3,out_channels//2,'conv2',do_normalization=False,do_RELU=False)
        bn3=tf.contrib.layers.batch_norm(conv2,epsilon=1e-5,decay=layer_para['bn_decay'],activation_fn=tf.nn.relu)
        #conv3=conv_block(conv2,1,1,out_channels,'conv3',do_normalization=False,do_RELU=False)
        conv3=tf.contrib.layers.conv2d(bn3,out_channels,1,activation_fn=None)
        if tf.shape(input)[-1]==out_channels:
            output=tf.add(conv3,input,'output')
        else:
            trans=tf.contrib.layers.conv2d(input,out_channels,1)
            output=tf.add(conv3,trans,'output')
        return output




def residual_block(input,
                   stride,
                   kernel_number1,
                   kernel_number2,
                   name='residual_block'):
    with tf.variable_scope(name):
        conv1 = conv_block(input, 1, 1
                           if input.get_shape().as_list()[-1] == kernel_number2
                           else stride, kernel_number1, 'left_conv1')
        conv2 = conv_block(conv1, 3, 1, kernel_number1, 'left_conv2')
        conv3 = conv_block(
            conv2, 1, 1, kernel_number2, 'left_conv3', do_RELU=False)
        if input.get_shape().as_list()[-1] != kernel_number2:
            conv4 = conv_block(
                input, 1, stride, kernel_number2, 'right_conv', do_RELU=False)
            output = tf.add(conv3, conv4, 'output')
        else:
            output = tf.add(conv3, input, 'output')
        return tf.nn.relu(output, 'activated_output')


# 简单封装的卷积层 padding模式为same
def conv_block(input,
               kernel_size,
               stride,
               kernel_number,
               name='conv',
               padding='SAME',
               do_normalization=True,
               do_RELU=True):
    with tf.variable_scope(name):
        #print(tf.shape(input))
        weights = _variable_with_weight_decay(name, [
            kernel_size, kernel_size,
            input.get_shape()[-1], kernel_number
        ])
        conv = tf.nn.conv2d(
            input,
            weights, [1, stride, stride, 1],
            padding=padding,
            name='conv')
        #bias = tf.Variable(tf.zeros([kernel_number]), name='bias')
        #output = tf.nn.bias_add(conv, bias, name='output')
        output=tf.contrib.layers.bias_add(conv)
        if do_normalization:
            #output = batch_normalization(output,layer_para['bn_decay'] ,layer_para['bn_epsilon'],name='batch_normalized_output')
            output=tf.contrib.layers.batch_norm(output,epsilon=1e-5,decay=layer_para['bn_decay'])
        if do_RELU:
            output = tf.nn.relu(output, name='activated_output')
        return output


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
    return tf.get_variable(
        name,
        shape,
        #改成这个初始化方式
        initializer=tf.contrib.layers.xavier_initializer(uniform))
        #TODO  正则化的问题
        #regularizer=tf.contrib.layers.l1_regularizer(wd))


def max_pool(input, kernel_size=3, stride=2, name='max_pool'):
    return tf.nn.max_pool(
        input,
        ksize=[1, kernel_size, kernel_size, 1],
        strides=[1, stride, stride, 1],
        padding='SAME',
        name=name)
