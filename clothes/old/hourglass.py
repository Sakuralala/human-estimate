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
        #self.input = None

    def build_model(self, input):
        with tf.variable_scope('hourglass_model'):
            conv1 = conv_block(input, 7, 2, 64, 'conv1',do_normalization=False,do_RELU=False)
            res1 = PRMB_block(conv1,self.output_features//2, name='res1')
            pool = max_pool(res1, 2, 2)
            res2 = PRMB_block(pool,self.output_features // 2,name='res2')
            inter_total = PRMB_block(res2, 
                                         self.output_features, name='res3')

            for i in range(self.stack_number):
                with tf.variable_scope('hourglass' + str(i + 1)):
                    hourglass = self.hourglass2(inter_total, self.stage)
                    hourglass = PRMB_block(hourglass,
                                               self.output_features, name='res4')
                    conv2 = conv_block(hourglass, 1, 1, self.output_features,
                                       'conv1')
                    inter_output = conv_block(
                        conv2,
                        1,
                        1,
                        self.number_classes,
                        'inter_output',
                        do_normalization=False,
                        do_RELU=False)
                    #height = inter_output.get_shape().as_list()[1]
                    #width = inter_output.get_shape().as_list()[2]
                    #height=tf.shape(inter_output)[1]
                    #width=tf.shape(inter_output)[2]
                    #resized_inter_output=tf.image.resize_bilinear(inter_output,[height*4,width*4])
                    #self.output.append(resized_inter_output)
                    self.output.append(inter_output)
                    if i != self.stack_number - 1:
                        conv3 = conv_block(
                            conv2,
                            1,
                            1,
                            self.output_features,
                            'conv2',
                            do_normalization=False,
                            do_RELU=False
                        )
                        conv4 = conv_block(inter_output, 1, 1,
                                           self.output_features, 'conv3',do_normalization=False,do_RELU=False)
                        inter_total = tf.add_n([inter_total, conv3, conv4])

        return self.output
    #计算损失
    def loss_calculate(self, gt_heatmaps):
        with tf.variable_scope('Loss'):
            #for i,output in enumerate(self.output):
            for i in range(len(self.output)):
                #loss = tf.nn.l2_loss(self.output[i] - gt_heatmaps,
                #                     'inter_loss' + str(i + 1)) /train_para['batch_size'] 
                #loss=tf.reduce_sum(tf.square(self.output[i],gt_heatmaps))/train_para['batch_size']
                loss=tf.losses.mean_squared_error(gt_heatmaps,self.output[i],reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
                self.loss.append(loss)
                tf.summary.scalar(loss.op.name, loss)
            total_loss = tf.add_n(self.loss, 'Total_loss')
            tf.summary.scalar(total_loss.op.name, total_loss)
            return total_loss

    def hourglass(self, input, stage):
        with tf.variable_scope('stage' + str(stage)):
            input = residual_block(
                input,
                1,
                self.output_features // 2,
                self.output_features,
                name='res1')
            up = residual_block(input, 1, self.output_features // 2,
                                self.output_features, 'res2')
            down = max_pool(up, 2, 2, 'maxpooling')
            #down=residual_block(down,1,int(self.output_features/2),self.output_features)
            stage -= 1
            if stage > 0:
                ret = self.hourglass(down, stage)
            else:
                ret = residual_block(down, 1, self.output_features // 2,
                                     self.output_features, 'inest_res1')
                ret = residual_block(ret, 1, self.output_features // 2,
                                     self.output_features, 'inest_res2')
                ret = residual_block(ret, 1, self.output_features // 2,
                                     self.output_features, 'inest_res3')
            height = ret.get_shape().as_list()[1]
            wigth = ret.get_shape().as_list()[2]
            up2 = tf.image.resize_bilinear(ret, [height * 2, wigth * 2])
            total = tf.add(up, up2)
            total = residual_block(total, 1, self.output_features // 2,
                                   self.output_features, 'res3')
            return total

    def hourglass2(self, input, stage):
        if stage < 1:
            raise ValueError('Stage must >=1!')
        with tf.variable_scope('stage' + str(stage)):
            up = PRMB_block(input,
                                self.output_features, name='up')
            down = max_pool(input, 2, 2, 'maxpooling')
            down = PRMB_block(down,
                                  self.output_features,name='down')
            if stage > 1:
                ret = self.hourglass2(down, stage - 1)
            else:
                ret = PRMB_block(down,
                                     self.output_features, name='inest_res')
            ret = PRMB_block(ret,
                                 self.output_features, use_conv=True,name='res')
            height = tf.shape(ret)[1]
            width = tf.shape(ret)[2]
            #坑爹 这里*2回去可能会比up尺寸大  比如up是13那么maxpool再resize就变成了14.。。草
            #所以图片尺寸需要是2的倍数最好
            ret = tf.image.resize_bilinear(ret, [height * 2, width * 2])
            total = ret + up
            #total= batch_normalization(total,layer_para['bn_decay'] ,
            #            layer_para['bn_epsilon'],name='batch_normalized_total')
            #total=tf.nn.relu(total,'activated_total')

            return total
