#hourglass model

from layers import *
import tensorflow as tf


class HourglassModel():
    def __init__(self,
                 number_classes,
                 stack_number=4,
                 stage=4,
                 output_features=256,
                 name='hourglass_model'):
        '''
        desc:hourglass网络结构
        args:
            number_classes:关键点数量
            stack_number:hourglass堆叠的数量
            stage:单个hourglass的阶数
            output_features:residule module输出的特征图数量
        '''

        self.stack_number = stack_number
        self.stage = stage
        self.output_features = output_features
        self.name = name
        self.number_classes = number_classes
        self.output = []
        self.loss = []

    def build_model(self, input, use_loc=False):
        with tf.variable_scope('hourglass_model'):
            #使用stn
            if use_loc:
                theta = res18_loc(input)
                input = tf.contrib.image.transform(input, theta, 'BILINEAR')

            conv1 = conv_block_new(
                input,
                self.output_features // 4,
                7,
                2,
                'conv1')
            #tf.summary.histogram('conv1',conv1)
            res1 = residual_block(
                conv1, self.output_features // 2, name='res1')
            #tf.summary.histogram('res1',res1)
            pool = max_pool(res1, 2, 2)
            res2 = residual_block(pool, self.output_features // 2, name='res2')
            #tf.summary.histogram('res2',res2)
            inter_total = residual_block(
                res2, self.output_features, name='res3')
            #tf.summary.histogram('res3',inter_total)

            for i in range(self.stack_number):
                with tf.variable_scope('hourglass' + str(i + 1)):
                    hourglass = self.hourglass(inter_total, self.stage)
                    tf.summary.histogram('hourglass%s' % (i + 1), hourglass)
                    hourglass = residual_block(
                        hourglass, self.output_features, name='res4')
                    #tf.summary.histogram('res4',hourglass)
                    conv2 = conv_block_new(hourglass, self.output_features, 1,
                                           1, 'conv1')
                    #tf.summary.histogram('conv1',conv2)
                    inter_output = conv_block_new(
                        conv2,
                        self.number_classes,
                        1,
                        1,
                        'inter_output',
                        do_normalization=False,
                        do_relu=False)
                    tf.summary.histogram('inter_output', inter_output)
                    for i in range(inter_output.get_shape()[-1]):
                        pred_heatmap0 = tf.expand_dims(
                            inter_output[0, :, :, i], -1)
                        pred_heatmap0 = tf.expand_dims(pred_heatmap0, 0)
                        tf.summary.image(
                            'pre_heatmaps0', pred_heatmap0, max_outputs=24)

                    self.output.append(inter_output)
                    if i != self.stack_number - 1:
                        conv3 = conv_block_new(
                            conv2,
                            self.output_features,
                            1,
                            1,
                            'conv2',
                            do_normalization=False,
                            do_relu=False)
                        #tf.summary.histogram('conv2',conv3)
                        conv4 = conv_block_new(
                            inter_output,
                            self.output_features,
                            1,
                            1,
                            'conv3',
                            do_normalization=False,
                            do_relu=False)
                        #tf.summary.histogram('conv3',conv4)
                        inter_total = tf.add_n([inter_total, conv3, conv4])
                        #tf.summary.histogram('inter_total',inter_total)

        return self.output

    #计算损失
    def loss_calculate(self, gt_heatmaps):
        with tf.variable_scope('Loss'):
            #真实heatmaps
            for i in range(gt_heatmaps.get_shape()[-1]):
                gt_heatmap0 = tf.expand_dims(gt_heatmaps[0, :, :, i], -1)
                gt_heatmap0 = tf.expand_dims(gt_heatmap0, 0)
                tf.summary.image('gt_heatmaps0', gt_heatmap0, max_outputs=24)
            #for i,output in enumerate(self.output):
            for i in range(len(self.output)):
                loss = tf.losses.mean_squared_error(gt_heatmaps,
                                                    self.output[i])
                #loss = tf.nn.l2_loss(self.output[i] - gt_heatmaps,
                #                      'inter_loss' + str(i + 1)) /train_para['batch_size']
                #loss=tf.reduce_sum(tf.square(self.output[i],gt_heatmaps))/train_para['batch_size']
                self.loss.append(loss)
                tf.summary.scalar(loss.op.name, loss)
            total_loss = tf.add_n(self.loss, 'Total_loss')
            tf.summary.scalar(total_loss.op.name, total_loss)

            return total_loss

    def hourglass(self, input, stage):
        if stage < 1:
            raise ValueError('Stage must >=1!')
        with tf.variable_scope('stage' + str(stage)):
            up = residual_block(input, self.output_features, name='up')
            down = max_pool(input, 2, 2, 'maxpooling')
            down = residual_block(down, self.output_features, name='down')
            if stage > 1:
                ret = self.hourglass(down, stage - 1)
            else:
                ret = residual_block(
                    down, self.output_features, name='inest_res')
            ret = residual_block(ret, self.output_features, True, name='res')
            height = tf.shape(ret)[1]
            width = tf.shape(ret)[2]
            #坑爹 这里*2回去可能会比up尺寸大  比如up是13那么maxpool再resize就变成了14.。。草
            #所以图片尺寸需要是2的指数次最好
            ret = tf.image.resize_bilinear(ret, [height * 2, width * 2])
            total = ret + up

            return total
