# -*- coding: utf-8 -*-
# @Time    : 2019/1/3 17:40
# @Author  : zhoujun

# train config
gpu_id = '0'
workers = 12
start_epoch = 0
epochs = 2000

train_batch_size = 1

# learning_rate
lr = 5 * 1e-5
end_lr = 1e-8
lr_gamma = 0.1
lr_decay_step = [300, 1000]
# warn_up
weight_decay = 5e-4
warm_up_epoch = 6
warm_up_lr = lr * lr_gamma

display_interval = 1
pretrained = True
restart_training = True
checkpoint = ''

# net config
# random seed
seed = 2

# data config
trainroot = '/home/dong/Downloads/receipt/ctpntc/data/new_train/'
testroot = '/home/dong/Downloads/receipt/ctpntc/data/new_train/'
output_dir = f'output/ctpn_{len(gpu_id.split(","))}_gpu1111'
MAX_LEN = 1200
MIN_LEN = 600
# bgr can find from  here: https://github.com/fchollet/deep-learning-models/blob/master/imagenet_utils.py
IMAGE_MEAN = [123.68, 116.779, 103.939]


anchor_scale = 16
IOU_NEGATIVE = 0.5
IOU_POSITIVE = 0.9
IOU_SELECT = 0.9

RPN_POSITIVE_NUM = 128
RPN_TOTAL_NUM = RPN_POSITIVE_NUM * 2

def print():
    from pprint import pformat
    tem_d = {}
    for k, v in globals().items():
        if not k.startswith('_') and not callable(v):
            tem_d[k] = v
    return pformat(tem_d)
