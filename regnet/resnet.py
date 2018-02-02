import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import datetime
import numpy as np
import os
import time

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'regnet_variables'
UPDATE_OPS_COLLECTION = 'regnet_update_ops'  # must be grouped with training op
BATCH_SIZE = 64
JOR_INPUT_SIZE = 128
# 21个关节加一个背景
NUM_OF_HEATMAPS = 22
#关节数量
NUM_OF_JOINTS = 21

tf.app.flags.DEFINE_integer('input_size', 224, "input image size")


#HALNet和JORNet相同的部分 同时返回三个子网络和一个最终网络(按先后顺序)
def main_net(input, name='main net'):
    with tf.variable_scope(name):
        conv1 = conv_block(input, 7, 1, 64, 'conv1')
        pool1 = max_pool(conv1, name='max pool1')
        _, residual_layer1 = residual_layer_of_3blocks(pool1, 1, 64, 256,
                                                       'residual layer1')
        residual_layer2_block1, residual_layer2 = residual_layer_of_3blocks(
            residual_layer1, 2, 128, 512, 'residual layer2')
        residual_layer3_block1, residual_layer3 = residual_layer_of_3blocks(
            residual_layer2, 2, 256, 1024, 'residual layer3')
        conv2 = conv_block(residual_layer3, 3, 1, 512, 'conv2')
        conv3 = conv_block(conv2, 3, 1, 256, 'conv3')
        return residual_layer2_block1, residual_layer3_block1, conv2, conv3


#residual_block的封装
def residual_layer_of_3blocks(input,
                              stride,
                              kernel_number1,
                              kernel_number2,
                              name='residual layer'):
    with tf.variable_scope(name):
        residual_block1 = residual_block(input, stride, kernel_number1,
                                         kernel_number2, 'residual block1')
        residual_block2 = residual_block(residual_block1, 1, kernel_number1,
                                         kernel_number2, 'residual block2')
        residual_block3 = residual_block(residual_block2, 1, kernel_number1,
                                         kernel_number2, 'residual block3')
        #因为要做intermidiate supervision 所以还要返回中间的结果
        return residual_block1,residual_block3


def residual_block(input,
                   stride,
                   kernel_number1,
                   kernel_number2,
                   name='residual_block'):
    with tf.variable_scope(name):
        conv1 = conv_block(input, 1, 1
                           if input.get_shape().as_list()[-1] == kernel_number2 else
                           stride, kernel_number1, 'left_conv1')
        conv2 = conv_block(conv1, 3, 1, kernel_number1, 'left_conv2')
        conv3 = conv_block(
            conv2, 1, 1, kernel_number2, 'left_conv3', do_RELU=False)
        if input.get_shape().as_list()[-1] != kernel_number2:
            conv4 = conv_block(input, 1, stride, kernel_number2, 'right_conv',do_RELU=False)
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
               do_normalization=True,
               do_RELU=True):
    with tf.variable_scope(name):
        weights = get_variable(
            shape=[
                kernel_size, kernel_size,
                input.get_shape().as_list()[-1], kernel_number
            ],
            dtype='float',
            name='weights',
            initializer=tf.truncated_normal_initializer(
                stddev=CONV_WEIGHT_STDDEV),
            weight_decay=CONV_WEIGHT_DECAY)
        conv = tf.nn.conv2d(
            input,
            weights, [1, stride, stride, 1],
            padding='SAME',
            name='conv')
        bias = tf.Variable(tf.zeros([kernel_number]), name='bias')
        output = tf.nn.bias_add(conv, bias, name='output')
        if do_normalization:
            output = batch_normalization(output, 'batch normalized output')
        if do_RELU:
            output = tf.nn.relu(output, name='activated output')
        return output


# 归一化  在激活前做
def batch_normalization(input, decay=0.9,name='batch normalization',is_train=True):
    mean, variance = tf.nn.moments(input, [0, 1, 2])
    scale = tf.Variable(tf.ones([1]))
    offset = tf.Variable(tf.zeros([1]))

    #训练时更新mean和variance
    ema=tf.train.ExponentialMovingAverage(decay)
    def mean_var_update():
        ema_apply_op=ema.apply([mean,variance])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(mean),tf.identity(variance)
    #训练时更新，测试时使用平均值
    mean,variance=tf.cond(tf.reshape(is_train,[]),mean_var_update,lambda:(ema.average(mean),ema.average(variance)))
    normalized=tf.nn.batch_normalization(input, mean, variance, offset, scale,
                                     10e-5, name)
    return normalized


# tf.get_variable的封装
def get_variable(shape,
                 initializer,
                 name,
                 weight_decay=0.0,
                 dtype='float',
                 trainable=True):
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(
        name,
        shape=shape,
        initializer=initializer,
        dtype=dtype,
        regularizer=regularizer,
        collections=collections,
        trainable=trainable)


# 全连接层
def full_connect(input, output_number=NUM_OF_JOINTS*3,name='full connect layer'):
    with tf.variable_scope(name):
        input_reshaped=tf.reshape(input,[BATCH_SIZE,-1],'reshaped input')
        dim1=input_reshaped.get_shape().as_list()[1]
        weights=get_variable([dim1,output_number],tf.truncated_normal_initializer(),'weights')
        bias=tf.Variable(tf.zeros([output_number]),name='bias')
        output=tf.nn.bias_add(tf.matmul(input,weights),bias,name='output')
        return output


# 用以将heatmap恢复至指定大小
def transpose_conv2d(input,
                     kernel_size=4,
                     stride=2,#需要放大的倍数(对应padding为SAME而言)
                     num_of_heatmaps=NUM_OF_HEATMAPS,
                     name='transpose_conv',
                     is_weight_fixed=False):
    with tf.variable_scope(name):
        height=input.get_shape().as_list()[1]
        width=input.get_shape().as_list()[2]
        kernel_number1=input.get_shape().as_list()[-1]
        if stride is None:
            stride=2
        if is_weight_fixed==False:
            weights = get_variable(
                [
                    kernel_size, kernel_size, num_of_heatmaps,
                    input.get_shape().as_list()[-1]
                ],
                tf.truncated_normal_initializer(stddev=CONV_WEIGHT_DECAY),
                'weights')
        else:
            weights=bilinear_upsampling(stride,kernel_number1,num_of_heatmaps)
        conv_transposed= tf.nn.conv2d_transpose(input, weights, [BATCH_SIZE,height*stride,width*stride,num_of_heatmaps],
                                        [1, stride, stride, 1])
        bias=tf.get_variable('bias',[num_of_heatmaps],initializer=tf.zeros_initializer())
        output=tf.nn.bias_add(conv_transposed,bias,name='output')
        return output

#双线性插值初始化权重
def bilinear_upsampling(factor,kernel_number1,kernel_number2=NUM_OF_HEATMAPS):
    kernel_size=get_kernel_size(factor)
    weight=np.zeros((kernel_size,kernel_size,kernel_number2,kernel_number1))
    value=upsample_filter(kernel_size)
    weight[:,:,range(kernel_number2),range(kernel_number1)]=value
    return tf.Variable(weight,name='fixed_weights')


#bilinear 具体的计算
def upsample_filter(size):
    factor = (size + 1) // 2
    if size % 2:
        center = factor - 1
    else:
        center = factor - 0.5
    #返回一组可用于广播计算的数组
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


#计算需要的卷积核大小 factor代表要放大的倍数
def get_kernel_size(factor):
    return factor * 2 - factor % 2


def max_pool(input, kernel_size=3, stride=2, name='max pool'):
    return tf.nn.max_pool(
        input,
        ksize=[1, kernel_size, kernel_size, 1],
        strides=[1, stride, stride, 1],
        padding='SAME',
        name=name)


def total_loss_of_HALNet(input, ground_truth_heatmaps, name='HALNet loss'):
    inter_output1, inter_output2, inter_output3, final_output = main_net(
        input, 'HALNet')
    with tf.variable_scope(name):
        inter_loss1=inter_heatmap_loss(inter_output1,ground_truth_heatmaps,'inter heatmap loss 1')
        inter_loss2=inter_heatmap_loss(inter_output2,ground_truth_heatmaps,'inter heatmap loss 2')
        inter_loss3=inter_heatmap_loss(inter_output3,ground_truth_heatmaps,'inter heatmap loss 3')
        main_loss=main_loss_of_HALNet(final_output,ground_truth_heatmaps)
        total_loss=tf.add(inter_loss1+inter_loss2+inter_output3,main_loss,'total loss')
    return total_loss




def main_loss_of_HALNet(input, ground_truth_heatmaps,name='main loss'):
    input_channels=input.get_shape().as_list()[-1]
    size=input.get_shape().as_list()[1]
    heatmap_size=ground_truth_heatmaps.get_shape().as_list()[1]
    #filter参数，不要直接传入一个tensor！！！
    weights=get_variable([3,3,input_channels,NUM_OF_HEATMAPS],tf.truncated_normal_initializer(),name='weights')
    conv=tf.nn.conv2d(input,weights,[1,1,1,1],'SAME',name='conv')
    conv_transposed=transpose_conv2d(conv,4,heatmap_size/size,NUM_OF_HEATMAPS,'transposed conv',is_weight_fixed=True)
    loss=tf.nn.l2_loss(conv_transposed-ground_truth_heatmaps,name=name)
    return loss


def total_loss_of_JORNet(input,ground_truth_heatmaps,ground_truth_positions,name='JORNet loss'):
    inter_output1,inter_output2,inter_output3,final_output=main_net(input,'JORNet')
    with tf.variable_scope(name):
        inter_loss1=inter_loss_of_JORNet(inter_output1,ground_truth_heatmaps,ground_truth_positions,'inter loss1')
        inter_loss2=inter_loss_of_JORNet(inter_output2,ground_truth_heatmaps,ground_truth_positions,'inter loss2')
        inter_loss3=inter_loss_of_JORNet(inter_output3,ground_truth_heatmaps,ground_truth_positions,'inter loss3')
        main_loss=main_loss_of_JORNet(final_output,ground_truth_heatmaps,ground_truth_positions)
        total_loss=tf.add(inter_loss1+inter_loss2+inter_loss3,main_loss,'total loss')
        return total_loss


def main_loss_of_JORNet(input,ground_truth_heatmaps,ground_truth_positions,name='main loss'):
    input_channels=input.get_shape().as_list()[-1]
    size=input.get_shape().as_list()[1]
    weights=get_variable([3,3,input_channels,64],tf.truncated_normal_initializer(),'weights')
    conv=tf.nn.conv2d(input,weights,[1,1,1,1],'SAME',name='conv')
    conv_transposed=transpose_conv2d(conv,4,2,NUM_OF_HEATMAPS,'transposed conv1')
    conv_transposed2=transpose_conv2d(conv_transposed,4,2,NUM_OF_HEATMAPS,'transposed conv2')
    conv_transposed3=transpose_conv2d(conv_transposed2,4,2,NUM_OF_HEATMAPS,'transposed conv3',is_weight_fixed=True)
    loss1=tf.nn.l2_loss(conv_transposed2-ground_truth_heatmaps,name='heatmaps loss')

    full_connect1=full_connect(input,200,'full connect layer1')
    full_connect2=full_connect(full_connect1,NUM_OF_JOINTS*3,'full connect layer2')
    loss2=tf.nn.l2_loss(full_connect2-ground_truth_positions,name='regression loss')
    main_loss=tf.add(loss1,loss2,name)
    return main_loss


def inter_loss_of_JORNet(input,ground_truth_heatmaps,ground_truth_positions,name='JORNet inter loss'):
    with tf.variable_scope(name):
        loss1=inter_heatmap_loss(input,ground_truth_heatmaps)
        loss2=inter_regression_loss_of_JORNet(input,ground_truth_positions)
        loss=tf.add(loss1,loss2,name)
        return loss



def inter_heatmap_loss(input, ground_truth_heatmaps,name='inter heatmap loss'):
    current_size = input.get_shape().as_list()[1]
    heatmap_size=ground_truth_heatmaps.get_shape().as_list()[1]
    predict_heatmaps = transpose_conv2d(
        input,
        4,
        heatmap_size/ current_size,
        NUM_OF_HEATMAPS,
        name=name,
        is_weight_fixed=True
    )
    loss = tf.nn.l2_loss(
        predict_heatmaps - ground_truth_heatmaps,
        name=name)
    #tf.summary.scalar('inter_heatmap_loss', loss)
    return loss


def inter_regression_loss_of_JORNet(input,ground_truth_positions,name='inter regression loss'):
    predicted_positions=full_connect(input,NUM_OF_JOINTS*3)
    loss=tf.nn.l2_loss(predicted_positions-ground_truth_positions,name=name)
    return loss
