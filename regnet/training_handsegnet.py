from __future__ import print_function, unicode_literals

import tensorflow as tf
import os
import sys

from BinaryDbReader import BinaryDbReader
from utils import LearningRateScheduler
from resnet import total_loss_of_HALNet
# training parameters
train_para = {'lr': [1e-5, 1e-6, 1e-7],
              'lr_iter': [20000, 30000],
              'max_iter': 40000,
              'show_loss_freq': 1000,
              'snapshot_freq': 5000,
              'snapshot_dir': 'snapshots_handsegnet'}

# get dataset
dataset = BinaryDbReader(mode='training',
                         batch_size=8, shuffle=True,
                         hue_aug=True,hand_crop=False)

# build network graph
data = dataset.get()

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.train.start_queue_runners(sess=sess)

#Loss [B,320,320]  [B,320,320]
loss=total_loss_of_HALNet(data['image'],data['gaussian_map'])

# Solver
global_step = tf.Variable(0, trainable=False, name="global_step")
lr_scheduler = LearningRateScheduler(values=train_para['lr'], steps=train_para['lr_iter'])
lr = lr_scheduler.get_lr(global_step)
opt = tf.train.AdamOptimizer(lr)
train_op = opt.minimize(loss)

# init weights
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=4.0)

#rename_dict = {'CPM/PersonNet': 'HandSegNet',
#               '_CPM': ''}
#load_weights_from_snapshot(sess, './weights/cpm-model-mpii', ['PoseNet', 'Mconv', 'conv6'], rename_dict)

# snapshot dir
if not os.path.exists(train_para['snapshot_dir']):
    os.mkdir(train_para['snapshot_dir'])
    print('Created snapshot dir:', train_para['snapshot_dir'])

# Training loop
for i in range(train_para['max_iter']):
    _, loss_v = sess.run([train_op, loss])

    if (i % train_para['show_loss_freq']) == 0:
        print('Iteration %d\t Loss %.1e' % (i, loss_v))
        sys.stdout.flush()

    if (i % train_para['snapshot_freq']) == 0:
        saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=i)
        print('Saved a snapshot.')
        sys.stdout.flush()


print('Training finished. Saving final snapshot.')
saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=train_para['max_iter'])
