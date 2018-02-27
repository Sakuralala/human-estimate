import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import datetime
import numpy as np
import os
import time

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
BATCH_SIZE = 8
JOR_INPUT_SIZE = 128
# 21个关节加一个背景
NUM_OF_HEATMAPS = 22
#关节数量
NUM_OF_JOINTS = 21
HAL_LOSS_WEIGHT=1.0
INTER_LOSS_WEIGHT=0.5


#HALNet和JORNet相同的部分 同时返回三个子网络和一个最终网络(按先后顺序)
def main_net(input, name='main_net'):
    with tf.variable_scope(name):
        conv1 = conv_block(input, 7, 1, 64, 'conv1')
        pool1 = max_pool(conv1, name='max_pool1')
        _, residual_layer1 = residual_layer(pool1, 1, 64, 256,3,
                                                       'residual_layer1')
        residual_layer2_block1, residual_layer2 = residual_layer(
            residual_layer1, 2, 128, 512, 3,'residual_layer2')
        residual_layer3_block1, residual_layer3 = residual_layer(
            residual_layer2, 2, 256, 1024, 4,'residual_layer3')

        conv2 = conv_block(residual_layer3, 3, 1, 512, 'conv2')
        conv3 = conv_block(conv2, 3, 1, 256, 'conv3')
        return residual_layer2_block1, residual_layer3_block1, conv2, conv3


#residual_block的封装
def residual_layer(input,
                   stride,
                   kernel_number1,
                   kernel_number2,
                   block_number=3,
                   name='residual_layer'):
    residual_blocks=[]
    with tf.variable_scope(name):
        for i in range(block_number):
            if len(residual_blocks)==0:
                residual_blockx = residual_block(input, stride, kernel_number1,
                                         kernel_number2, 'residual_block1')
            else:
                residual_blockx=residual_block(residual_blocks[-1],stride,kernel_number1,
		                         kernel_number2,'residual_block%d'%(i+1))
            residual_blocks.append(residual_blockx)
		
        #因为要做intermidiate supervision 所以还要返回中间的结果
        return residual_blocks[0],residual_blocks[-1]


def residual_block(input,
                   stride,
                   kernel_number1,
                   kernel_number2,
                   name='residual_block'):
    with tf.variable_scope(name):
        conv1 = conv_block(input, 1, 1 if input.get_shape().as_list()[-1] == kernel_number2 else stride,
	 kernel_number1, 'left_conv1')
        conv2 = conv_block(conv1, 3, 1, kernel_number1, 'left_conv2')
        conv3 = conv_block(
            conv2, 1, 1, kernel_number2, 'left_conv3', do_RELU=False)
        if input.get_shape().as_list()[-1] != kernel_number2:
            conv4 = conv_block(input, 1, stride, kernel_number2, 'right_conv',do_RELU=False)
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
               do_normalization=True,
               do_RELU=True):
    with tf.variable_scope(name):
        weights = get_variable(
            shape=[
                kernel_size, kernel_size,
                input.get_shape().as_list()[-1], kernel_number
            ],
            dtype=tf.float32,
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
            output = batch_normalization(output, BN_DECAY,'batch_normalized_output')
        if do_RELU:
            output = tf.nn.relu(output, name='activated_output')
        return output


# 归一化  在激活前做
def batch_normalization(input, decay=0.9,name='batch_normalization',is_train=True):
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
    mean,variance=tf.cond(tf.cast(is_train,tf.bool),mean_var_update,lambda:(ema.average(mean),ema.average(variance)))
    normalized=tf.nn.batch_normalization(input, mean, variance, offset, scale,
                                     10e-3, name)
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
    collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    return tf.get_variable(
        name,
        shape=shape,
        initializer=initializer,
        dtype=dtype,
        regularizer=regularizer,
        collections=collections,
        trainable=trainable)


# 全连接层
def full_connect(input, output_number=NUM_OF_JOINTS*3,name='full_connect_layer'):
    with tf.variable_scope(name):
        input_reshaped=tf.reshape(input,[BATCH_SIZE,-1],'reshaped_input')
        dim1=input_reshaped.get_shape().as_list()[1]
        weights=get_variable([dim1,output_number],tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV),'weights',weight_decay=FC_WEIGHT_DECAY)
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
                    kernel_number1 
                ],
                tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV),
                'weights',weight_decay=CONV_WEIGHT_DECAY)
        else:
            weights_val=bilinear_upsampling(stride,kernel_number1,num_of_heatmaps)
            weights=tf.Variable(weights_val,name='fixed_weights',dtype=tf.float32)
        conv_transposed= tf.nn.conv2d_transpose(input, weights, [BATCH_SIZE,int(height*stride),int(width*stride),num_of_heatmaps],
                                        [1, stride, stride, 1])
        bias=tf.get_variable('bias',[num_of_heatmaps],initializer=tf.zeros_initializer())
        output=tf.nn.bias_add(conv_transposed,bias,name='output')
        return output

#双线性插值初始化权重
def bilinear_upsampling(factor,kernel_number1,kernel_number2=NUM_OF_HEATMAPS):
    kernel_size=get_kernel_size(factor)
    weight=np.zeros((kernel_size,kernel_size,kernel_number2,kernel_number1))
    #weight=np.zeros((kernel_number2,kernel_number1,kernel_size,kernel_size),dtype=np.float32)
    value=upsample_filter(kernel_size)
    '''
    weight=tf.Variable(value,name='fixed_weights')
    weight=tf.expand_dims(tf.expand_dims(weight,-1),-1)
    #[kernel_size,kernel_size,kernel_number2,kernel_number1]
    weight=tf.cast(tf.tile(weight,[1,1,kernel_number2,kernel_number1]),tf.float32)
    '''
    for i in range(NUM_OF_HEATMAPS):
        weight[:,:,i,i]=value
    #weight[:,:,range(kernel_number2),range(kernel_number1)]=value
    return weight


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
    return int(factor * 2 - factor % 2)


def max_pool(input, kernel_size=3, stride=2, name='max_pool'):
    return tf.nn.max_pool(
        input,
        ksize=[1, kernel_size, kernel_size, 1],
        strides=[1, stride, stride, 1],
        padding='SAME',
        name=name)


def total_loss_of_HALNet(input, ground_truth_heatmaps, name='HALNet_loss'):
    inter_output1, inter_output2, inter_output3, final_output = main_net(
        input, 'HALNet')
    with tf.variable_scope(name):
        inter_loss1=inter_heatmap_loss(inter_output1,ground_truth_heatmaps,'inter_heatmap_loss_1')
        inter_loss2=inter_heatmap_loss(inter_output2,ground_truth_heatmaps,'inter_heatmap_loss_2')
        inter_loss3=inter_heatmap_loss(inter_output3,ground_truth_heatmaps,'inter_heatmap_loss_3')
        main_loss=main_loss_of_HALNet(final_output,ground_truth_heatmaps)
        #0-D
        #4???  [8,40,40,512]
        total_loss=tf.add_n([INTER_LOSS_WEIGHT*inter_loss1,INTER_LOSS_WEIGHT*inter_loss2,INTER_LOSS_WEIGHT*inter_loss3,HAL_LOSS_WEIGHT*main_loss],'total_loss')
        #print((total_loss.get_shape()))
    return total_loss




def main_loss_of_HALNet(input, ground_truth_heatmaps,name='main_loss'):
    input_channels=input.get_shape().as_list()[-1]
    size=input.get_shape().as_list()[1]
    height=ground_truth_heatmaps.get_shape().as_list()[1]
    width=ground_truth_heatmaps.get_shape().as_list()[2]
    
    weight=get_variable([3,3,input_channels,NUM_OF_HEATMAPS],\
            tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV),'weights',weight_decay=CONV_WEIGHT_DECAY)
    conv=tf.nn.conv2d(input,weight,[1,1,1,1],'SAME',name='conv')
    bias=tf.Variable(tf.zeros([NUM_OF_HEATMAPS]),name='bias')
    conv=tf.nn.bias_add(conv,bias) 
    #conv_transposed=transpose_conv2d(conv,4,heatmap_size/size,NUM_OF_HEATMAPS,'transposed_conv',is_weight_fixed=True)
    predict_heatmaps=tf.image.resize_bilinear(conv,[height,width])
    loss=tf.nn.l2_loss(predict_heatmaps-ground_truth_heatmaps,name=name)/BATCH_SIZE
    #0-D
    return loss


def total_loss_of_JORNet(input,ground_truth_heatmaps,ground_truth_positions,name='JORNet_loss'):
    inter_output1,inter_output2,inter_output3,final_output=main_net(input,'JORNet')
    with tf.variable_scope(name):
        inter_loss1=inter_loss_of_JORNet(inter_output1,ground_truth_heatmaps,ground_truth_positions,'inter_loss1')
        inter_loss2=inter_loss_of_JORNet(inter_output2,ground_truth_heatmaps,ground_truth_positions,'inter_loss2')
        inter_loss3=inter_loss_of_JORNet(inter_output3,ground_truth_heatmaps,ground_truth_positions,'inter_loss3')
        main_loss=main_loss_of_JORNet(final_output,ground_truth_heatmaps,ground_truth_positions)
        total_loss=tf.add(inter_loss1+inter_loss2+inter_loss3,main_loss,'total_loss')
        return total_loss


def main_loss_of_JORNet(input,ground_truth_heatmaps,ground_truth_positions,name='main_loss'):
    input_channels=input.get_shape().as_list()[-1]
    size=input.get_shape().as_list()[1]
    weight=get_variable([3,3,input_channels,64],tf.truncated_normal_initializer(),'weight')
    conv=tf.nn.conv2d(input,weight,[1,1,1,1],'SAME',name='conv')
    conv_transposed=transpose_conv2d(conv,4,2,NUM_OF_HEATMAPS,'transposed_conv1')
    conv_transposed2=transpose_conv2d(conv_transposed,4,2,NUM_OF_HEATMAPS,'transposed_conv2')
    conv_transposed3=transpose_conv2d(conv_transposed2,4,2,NUM_OF_HEATMAPS,'transposed_conv3',is_weight_fixed=True)
    loss1=tf.nn.l2_loss(conv_transposed2-ground_truth_heatmaps,name='heatmaps_loss')

    full_connect1=full_connect(input,200,'full_connect_layer1')
    full_connect2=full_connect(full_connect1,NUM_OF_JOINTS*3,'full_connect_layer2')
    loss2=tf.nn.l2_loss(full_connect2-ground_truth_positions,name='regression_loss')
    main_loss=tf.add(loss1,loss2,name)
    return main_loss


def inter_loss_of_JORNet(input,ground_truth_heatmaps,ground_truth_positions,name='JORNet_inter_loss'):
    with tf.variable_scope(name):
        loss1=inter_heatmap_loss(input,ground_truth_heatmaps)
        loss2=inter_regression_loss_of_JORNet(input,ground_truth_positions)
        loss=tf.add(loss1,loss2,name)
        return loss



def inter_heatmap_loss(input, ground_truth_heatmaps,name='inter_heatmap_loss'):
  s=input.get_shape().as_list()
  height=ground_truth_heatmaps.get_shape().as_list()[1]
  width=ground_truth_heatmaps.get_shape().as_list()[2]
  '''
    #TODO:Slice input。根据网络结构推测 应该是拆分channel维 那么拆分后的多维数组怎么处理呢？
    #难道是对应元素分别相加？ 另外 不整除怎么办？丢掉剩下的？
    channels_fixed=channels-channels%NUM_OF_HEATMAPS
    num_of_split=int(channels_fixed/NUM_OF_HEATMAPS)
    #print(channels_fixed,num_of_split)
    input_splited=tf.split(input[:,:,:,0:channels_fixed],num_of_split,-1)
    #[B,H,W,21]
    input_average=tf.add_n(input_splited)/len(input_splited)
  '''
    #直接卷积一个了不管了
  with tf.variable_scope(name):
    weight=get_variable([3,3,s[3],NUM_OF_HEATMAPS],
            dtype=tf.float32,
            name='inter_weights',
            initializer=tf.truncated_normal_initializer(
                stddev=CONV_WEIGHT_STDDEV),
            weight_decay=CONV_WEIGHT_DECAY)
    conv=tf.nn.conv2d(input,weight,[1,1,1,1],'SAME')
    bias=tf.Variable(tf.zeros([NUM_OF_HEATMAPS]))
    conv=tf.nn.bias_add(conv,bias)
    '''
    predict_heatmaps = transpose_conv2d(
        conv,
        4,#TODO:不知道多少合适。。。
        heatmap_size/ s[1],
        NUM_OF_HEATMAPS,
        name='transposed_conv2d',
        is_weight_fixed=True
    )
    '''
    #直接resize了
    predict_heatmaps=tf.image.resize_bilinear(conv,[height,width])
	#官方实现没有除batch_size
    loss = tf.nn.l2_loss(
        predict_heatmaps - ground_truth_heatmaps,
        name=name)/BATCH_SIZE
    #tf.summary.scalar('inter_heatmap_loss', loss)
    #0-D
    #print(loss.get_shape())
    return loss


def inter_regression_loss_of_JORNet(input,ground_truth_positions,name='inter_regression_loss'):
    predicted_positions=full_connect(input,NUM_OF_JOINTS*3)
    loss=tf.nn.l2_loss(predicted_positions-ground_truth_positions,name=name)
    return loss
