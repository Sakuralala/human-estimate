#hourglass model


from layers import *
import tensorflow as tf 



class HourglassModel():
    def __init__(self,number_classes,stack_number=8,stage=4,output_features=256,name='hourglass_model'):
        '''
        stack_number:hourglass堆叠的数量
        stage:单个hourglass的阶数
        output_features:residule module输出的特征图数量
        '''

        self.stack_number=stack_number
        self.stage=stage
        self.output_features=output_features
        self.name=name
        self.number_classes=number_classes
        self.loss=[]
    
    def build_model(self,input,gt_heatmaps):
        with tf.variable_scope('hourglass_model'):
            conv1=conv_block(input,7,2,64,'conv1')
            res1=residual_block(conv1,1,int(self.output_features/2),self.output_features,'res1')
            pool=max_pool(res1,2,2)
            res2=residual_block(pool,1,int(self.output_features/2),self.output_features,'res2')
            inter_total=residual_block(res2,1,int(self.output_features/2),self.output_features,'res3')
            '''
            with tf.variable_scope('hourglass1'):
                #res4=residual_block(res3,1,int(self.output_features/2),self.output_features,'res4')
                hourglass1=self.hourglass(res3,self.stage)
                #res5=residual_block(res4,1,int(self.output_features/2),self.output_features,'res5')
                conv2=conv_block(hourglass1,1,1,self.output_features,'conv1')
                conv3=conv_block(conv2,1,1,256,'conv2',do_normalization=False,do_RELU=False)
                inter_output=conv_block(conv2,1,1,self.number_classes,'inter_output',do_normalization=False,do_RELU=False)
                conv4=conv_block(inter_output,1,1,self.output_features,'conv3')
                inter_total=tf.add(res3,conv4)
            '''
            for i in range(self.stack_number):
                with tf.variable_scope('hourglass'+(i+1)):
                    hourglass=self.hourglass(inter_total,self.stage)
                    conv2=conv_block(hourglass,1,1,self.output_features,'conv1')
                    inter_output=conv_block(conv2,1,1,self.number_classes,'inter_output',do_normalization=False,do_RELU=False)
                    #loss的计算
                    batch_size=inter_output.get_shape().as_list()[0]
                    height=inter_output.get_shape().as_list()[1]
                    width=inter_output.get_shape().as_list()[2]
                    pred_heatmaps=tf.image.resize_bilinear(inter_output,[height*4,width*4],'predicted_heatmaps')
                    loss=tf.nn.l2_loss(pred_heatmaps-gt_heatmaps,'inter_loss'+(i+1))/batch_size
                    self.loss.append(loss)
                    if i!=self.stack_number-1:
                        conv3=conv_block(conv2,1,1,256,'conv2',do_normalization=False,do_RELU=False)
                        conv4=conv_block(inter_output,1,1,self.output_features,'conv3')
                        inter_total=tf.add_n([inter_total,conv3,conv4])


    def hourglass(self,input,stage):
        with tf.variable_scope('stage'+stage):
            input=residual_block(input,1,int(self.output_features/2),self.output_features)
            up=residual_block(input,1,int(self.output_features/2),self.output_features)
            down=max_pool(up,2,2)
            #down=residual_block(down,1,int(self.output_features/2),self.output_features)
            stage-=1
            if stage>0:
                ret=self.hourglass(stage,down)
            else:
                ret=residual_block(down,1,int(self.output_features/2),self.output_features)
                ret=residual_block(ret,1,int(self.output_features/2),self.output_features) 
                ret=residual_block(ret,1,int(self.output_features/2),self.output_features) 
            height=ret.get_shape().as_list()[1]
            wigth=ret.get_shape().as_list()[2]
            up2=tf.image.resize_bilinear(ret,[height*2,wigth*2])
            total=tf.add(up,up2)
            total=residual_block(total,1,int(self.output_features/2),self.output_features)
            return total
