#often used layers

import tensorflow as tf
import numpy as np
import os

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
NUM_OF_HEATMAPS=22

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
        return tf.nn.relu(output, 'activated output')


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
        weights = _variable_with_weight_decay(name, [
            kernel_size, kernel_size,
            input.get_shape().as_list()[-1], kernel_number
        ], CONV_WEIGHT_STDDEV, CONV_WEIGHT_DECAY)
        conv = tf.nn.conv2d(
            input,
            weights, [1, stride, stride, 1],
            padding=padding,
            name='conv')
        bias = tf.Variable(tf.zeros([kernel_number]), name='bias')
        output = tf.nn.bias_add(conv, bias, name='output')
        if do_normalization:
            output = batch_normalization(output, 'batch_normalized_output')
        if do_RELU:
            output = tf.nn.relu(output, name='activated_output')
        return output


# 归一化  在激活前做
def batch_normalization(input,
                        decay=0.9,
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
                                           scale, 10e-5, name)
    return normalized


# tf.get_variable的封装
def _variable_with_weight_decay(name, shape, stddev, wd):
    return tf.get_variable(
        name,
        shape,
        initializer=tf.truncated_normal_initializer(stddev=stddev),
        regularizer=tf.contrib.layers.l1_regularizer(wd))


def max_pool(input, kernel_size=3, stride=2, name='max_pool'):
    return tf.nn.max_pool(
        input,
        ksize=[1, kernel_size, kernel_size, 1],
        strides=[1, stride, stride, 1],
        padding='SAME',
        name=name)
