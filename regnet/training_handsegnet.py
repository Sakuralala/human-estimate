#使用halnet来进行关节点预测
from __future__ import print_function, unicode_literals

import tensorflow as tf
import os
import sys
from params import *

from BinaryDbReader import BinaryDbReader
#from utils import LearningRateScheduler
from regnet import total_loss_of_HALNet
# training parameters


dataset = BinaryDbReader(mode='training',
                         batch_size=8, shuffle=True,
                         hue_aug=True,hand_crop=False)

# build network graph
data = dataset.get()

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.train.start_queue_runners(sess=sess)

#Loss [B,320,320]  [B,320,320]
loss=total_loss_of_HALNet(data['image'],data['gaussian_map'])

#Variables
trainable_variables=[var for var in tf.trainable_variables() if 'fixed_weights' not in var.name]
#138 142
#print(len(trainable_variables),len(tf.trainable_variables()))
#for var in trainable_variables:
#	print(var.name)
# Solver
global_step = tf.Variable(0, trainable=False, name="global_step")
#lr_scheduler = LearningRateScheduler(values=train_para['lr'], steps=train_para['lr_iter'])
#lr = lr_scheduler.get_lr(global_step)
initial_learning_rate=0.1
lr=tf.train.exponential_decay(initial_learning_rate,global_step,decay_steps=10000,decay_rate=0.5)
opt = tf.train.AdamOptimizer(lr,0.9,0.999)
#opt=tf.train.AdadeltaOptimizer(lr)
train_op = opt.minimize(loss,var_list=trainable_variables,global_step=global_step)

# init weights
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=4.0)

# snapshot dir
if not os.path.exists(train_para['snapshot_dir']):
    os.mkdir(train_para['snapshot_dir'])
    print('Created snapshot dir:', train_para['snapshot_dir'])

print(sess.run(data['gaussian_map']))

ckpt = tf.train.get_checkpoint_state(train_para['snapshot_dir'])
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('reload model.')
               # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
else:
    print('No checkpoint file found')
# Training loop
for i in range(train_para['max_iter']):
    _, loss_v = sess.run([train_op, loss])

    #if (i % train_para['show_loss_freq']) == 0:
    print('Iteration %d\t Loss %.1e' % (i,loss_v))
    if (i % train_para['show_lr_freq']) == 0:
	    print('Current lr:',sess.run(lr))
    sys.stdout.flush()

    if (i % train_para['snapshot_freq']) == 0:
        saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=i)
        print('Saved a snapshot.')
        sys.stdout.flush()


print('Training finished. Saving final snapshot.')
saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=train_para['max_iter'])
