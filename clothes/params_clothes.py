#clothes 的参数

#data generate smthg
all_kpt_list = [
    'neckline_left', 'neckline_right', 'center_front', 'shoulder_left',
    'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left',
    'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in',
    'cuff_right_out', 'top_hem_left', 'top_hem_right', 'waistband_left',
    'waistband_right', 'hemline_left', 'hemline_right', 'crotch',
    'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out'
]

blouse_kpt_list = [
    'neckline_left', 'neckline_right', 'center_front', 'shoulder_left',
    'shoulder_right', 'armpit_left', 'armpit_right', 'cuff_left_in',
    'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left',
    'top_hem_right'
]

skirt_kpt_list = [
    'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right'
]

outwear_kpt_list = [
    'neckline_left', 'neckline_right', 'shoulder_left', 'shoulder_right',
    'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right',
    'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out',
    'top_hem_left', 'top_hem_right'
]

trousers_kpt_list = [
    'waistband_left', 'waistband_right', 'crotch', 'bottom_left_in',
    'bottom_left_out', 'bottom_right_in', 'bottom_right_out'
]

dress_kpt_list = [
    'neckline_left', 'neckline_right', 'center_front', 'shoulder_left',
    'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left',
    'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in',
    'cuff_right_out', 'hemline_left', 'hemline_right'
]

categories_dict = {
    'blouse':blouse_kpt_list,
    'skirt': skirt_kpt_list,
    'outwear': outwear_kpt_list,
    'trousers': trousers_kpt_list,
    'dress':dress_kpt_list 
}

categories = {'blouse', 'skirt', 'outwear', 'trousers', 'dress'}
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
    'bn_decay': 0.99,
    'bn_epsilon': 10e-4,
}

#train
train_para = {
    'batch_size': 8,
    'max_iter': 100000,
    'save_freq':5000,
    'show_lr_freq': 500,
    'show_loss_freq': 50,
    'snapshot_freq': 5000,
    'lr_decay_steps': 30000,
    'lr_decay_rate': 0.1,
    'init_lr': 2.5e-4,
    'is_train': True,
}

#dir
dir_para = {
    #TODO
    'trained_model_dir':
    '/home/b3336/singlehandpose/clothes/trained_model/',
    'log_dir':
    '/home/b3336/singlehandpose/clothes/log/',
    'tf_dir':
    '/home/b3336/singlehandpose/clothes/tfrec/',
    'tf_dir_test':
    '/home/b3336/singlehandpose/clothes/tfrec_test/',
    'train_data_dir':
    '/home/b3336/singlehandpose/clothes/train/',
    'test_data_dir':
    ''
}

thred=10e-12