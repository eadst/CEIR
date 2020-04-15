# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
'''
Stage 2: detection
Last time for updating: 04/15/2020
'''


# @Time    : 2018/6/11 15:54
# @Original Author  : zhoujun
import torch
import shutil
import numpy as np
import config
import os
import cv2
from tqdm import tqdm
from model import CTPN_Model
from predict import Pytorch_model
from cal_recall.script import cal_recall_precison_f1
from utils import draw_bbox

torch.backends.cudnn.benchmark = True


def main(model_path, path, save_path, gpu_id):
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_img_folder = os.path.join(save_path, 'img')
    if not os.path.exists(save_img_folder):
        os.makedirs(save_img_folder)
    save_txt_folder = os.path.join(save_path, 'result')
    if not os.path.exists(save_txt_folder):
        os.makedirs(save_txt_folder)
    img_paths = [os.path.join(path, x) for x in os.listdir(path)]
    net = CTPN_Model(pretrained=False)
    model = Pytorch_model(model_path, net=net, gpu_id=gpu_id)
    total_frame = 0.0
    total_time = 0.0
    for img_path in tqdm(img_paths):
        img_name = str(os.path.basename(img_path).split('.')[0])
        save_name = os.path.join(save_txt_folder, img_name + '.txt')
        boxes_list, t = model.predict(img_path)
        total_frame += 1
        total_time += t
        boxes_list = np.array([b[0] for b in boxes_list])
        # print('boxes_list: ', boxes_list)
        img = draw_bbox(img_path, boxes_list, color=(0, 0, 255))
        cv2.imwrite(os.path.join(save_img_folder, '{}.jpg'.format(img_name)), img)
        np.savetxt(save_name, boxes_list.reshape(-1, 8), delimiter=',', fmt='%d')
    print('fps:{}'.format(total_frame / total_time))
    return save_txt_folder


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str('1')
    model_path = 'output/ctpn_1_gpu1111/best_loss0.000151.pth'
    data_path = 'data/train/image'
    gt_path = 'data/train/label'
    save_path = './result'
    gpu_id = '0'
    # print('model_path:{}'.format(model_path))
    save_path = main(model_path, data_path, save_path, gpu_id=gpu_id)
    result = cal_recall_precison_f1(gt_path=gt_path, result_path=save_path)
    print(result)
    # print(cal_recall_precison_f1('/data2/dataset/ICD151/test/gt', '/data1/zj/tensorflow_PSENet/tmp/'))
