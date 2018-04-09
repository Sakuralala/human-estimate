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
    'dress':dress_kpt_list,
    'full':all_kpt_list
}

categories = ['skirt','blouse','outwear','trousers','dress','full']
#input
input_para = {
    'height': 512,
    'width': 512,
    'channels': 3,
    'train_num': 31631,
    'test_num': 9996,
    'resized_height':256,
    'resized_width':256
}

#layers
layer_para = {
    'weight_decay': 0.0,
    'weight_stddev': 1.0,
    'bn_decay': 0.99,
    'bn_epsilon': 1e-4,
}

#train
train_para = {
    'batch_size': 12,
    'max_iter': 400000,
    'save_freq':20000,
    'show_lr_freq': 500,
    'show_loss_freq': 50,
    'lr_decay_steps': 500000,
    'lr_decay_rate': 0.1,
    'init_lr': 2e-4,
    'is_train': True,
    'epoch':100
}

#dir
dir_para = {
    #TODO
    'trained_model_dir':
    './trained_model',
    'log_dir':
    './logsin/',
    'tf_dir':
    './tfrec/',
    'tf_dir_test':
    './tfrec_test/',
    'train_data_dir':
    'D:\\clothes\\tianchi\\fashionAI_key_points_train_20180227\\train',
    'test_data_dir':
    'D:\\clothes\\tianchi\\fashionAI_key_points_test_a_20180227\\test'
}

thred=1e-2
