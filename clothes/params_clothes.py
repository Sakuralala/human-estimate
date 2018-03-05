#clothes 的参数

#data generate smthg
#类别名和对应关节点数量的映射
categories_dict = {
    'blouse': 13,
    'skirt': 4,
    'outwear': 14,
    'trousers': 7,
    'dress': 15
}
#input
input_para = {
    'height': 512,
    'width': 512,
    'channels': 3,
    'train_num': 31631,
    'test_num': 9996
}

#layers
layer_para = {
    'weight_decay': 0.00004,
    #'weight_decay_rate': 0.1,
    'weight_stddev': 0.1,
    'bn_decay':0.99,
    'bn_epsilon':10e-4,

}

#train
train_para = {
    'batch_size':8,
    'max_iter': 200000,
    'show_lr_freq': 500,
    'show_loss_freq': 50,
    'snapshot_freq': 5000,
    'lr_decay_steps': 30000,
    'lr_decay_rate': 0.1,
    'init_lr': 2.5e-4,
    'is_train': True,
}

#dir
dir_para={
    #TODO
    'trained_model_dir': '',
    'log_dir': '',
    'tf_dir':'',
    'train_data_dir':'',
    'test_data_dir':''
}
