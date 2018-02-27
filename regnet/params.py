#一些参数

#others
MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
NUM_OF_HEATMAPS=22

#train
train_para = {
    'max_iter': 200000,
    'show_lr_freq': 1000,
    'show_loss_freq': 500,
    'snapshot_freq': 5000,
    'snapshot_dir': 'snapshots_handsegnet'
}
initial_learning_rate = 0.1