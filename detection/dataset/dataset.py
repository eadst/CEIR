# -*- coding:utf-8 -*-
# '''
# Created on 18-12-27 上午10:34
#
# @Author: Greg Gao(laygin)
# '''

import os
import glob
import random
import pathlib
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
from config import IMAGE_MEAN
from utils.bbox_utils import cal_rpn
from utils.split_polys import split_polys


# from ctpn_utils import cal_rpn


def readxml(path):
    gtboxes = []
    imgfile = ''
    xml = ET.parse(path)
    for elem in xml.iter():
        if 'filename' in elem.tag:
            imgfile = elem.text
        if 'object' in elem.tag:
            for attr in list(elem):
                if 'bndbox' in attr.tag:
                    xmin = int(round(float(attr.find('xmin').text)))
                    ymin = int(round(float(attr.find('ymin').text)))
                    xmax = int(round(float(attr.find('xmax').text)))
                    ymax = int(round(float(attr.find('ymax').text)))

                    gtboxes.append((xmin, ymin, xmax, ymax))

    return np.array(gtboxes), imgfile


def read_txt(label_path: str) -> tuple:
    boxes = []
    text_tags = []
    with open(label_path, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
            try:
                label = params[8]
                if label == '*' or label == '###':
                    text_tags.append(True)
                else:
                    text_tags.append(False)
                # if label == '*' or label == '###':
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, params[:8]))
                boxes.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            except:
                print('load label failed on {}'.format(label_path))
    return np.array(boxes, dtype=np.float32), np.array(text_tags, dtype=np.bool)


def load_data(data_dir: str) -> list:
    data_list = []
    for x in glob.glob(data_dir + '/image/*.jpg', recursive=True):
        d = pathlib.Path(x)
        label_path = os.path.join(data_dir, 'label', (str(d.stem) + '.txt'))
        polys, text = read_txt(label_path)
        if len(polys) > 0:
            data_list.append((x, polys, text))
        else:
            print('there is no suit bbox on {}'.format(label_path))
    return data_list


def horizontal_flip(im: np.ndarray, text_polys: np.ndarray) -> tuple:
    """
    对图片和文本框进行水平翻转
    :param im: 图片
    :param text_polys: 文本框
    :return: 水平翻转之后的图片和文本框
    """
    flip_text_polys = text_polys.copy()
    flip_im = cv2.flip(im, 1)
    h, w, _ = flip_im.shape
    flip_text_polys[:, :, 0] = w - flip_text_polys[:, :, 0]
    return flip_im, flip_text_polys


def vertical_flip(im: np.ndarray, text_polys: np.ndarray) -> tuple:
    """
     对图片和文本框进行竖直翻转
    :param im: 图片
    :param text_polys: 文本框
    :return: 竖直翻转之后的图片和文本框
    """
    flip_text_polys = text_polys.copy()
    flip_im = cv2.flip(im, 0)
    h, w, _ = flip_im.shape
    flip_text_polys[:, :, 1] = h - flip_text_polys[:, :, 1]
    return flip_im, flip_text_polys


def resize(im: np.ndarray, text_polys: np.ndarray, min_len: int, max_len: int) -> tuple:
    img_size = im.shape

    # 图片缩放
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])
    # 短边缩放到600 并且保证长边不超过1200
    im_scale = float(min_len) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_len:
        im_scale = float(max_len) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)
    # 保证边长能被16整除
    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    re_size = re_im.shape
    text_polys[:, :, 0] = text_polys[:, :, 0] / img_size[1] * re_size[1]
    text_polys[:, :, 1] = text_polys[:, :, 1] / img_size[0] * re_size[0]
    return re_im, text_polys


def augmentation(im: np.ndarray, text_polys: np.ndarray, min_len: int, max_len: int) -> tuple:
    if random.random() < 0.5:
        im, text_polys = horizontal_flip(im, text_polys)
    im, text_polys = resize(im, text_polys, min_len, max_len)
    return im, text_polys


# for ctpn text detection
class MyDataset(Dataset):
    def __init__(self, data_dir, min_len, max_len, transform=None, target_transform=None):
        '''

        :param txtfile: image name list text file
        :param datadir: image's directory
        '''
        if not os.path.isdir(data_dir):
            raise Exception(f'[ERROR] {data_dir} is not a directory')
        self.data_list = load_data(data_dir)
        self.min_len = min_len
        self.max_len = max_len
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, polys, _ = self.data_list[idx]
        img = cv2.imread(img_path)
        img, polys = augmentation(img, polys, self.min_len, self.max_len)
        polys, _ = split_polys(polys)
        # 进行anchor和 polys的匹配
        h, w, c = img.shape
        [cls, regr], _ = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, polys)
        # 减去均值
        # img = img - IMAGE_MEAN

        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])
        cls = np.expand_dims(cls, axis=0)

        cls = torch.from_numpy(cls).float()
        regr = torch.from_numpy(regr).float()
        if self.transform:
            img = self.transform(img)
        return img, cls, regr


if __name__ == '__main__':
    import torch
    import config
    from utils.utils import show_img, draw_bbox
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms

    train_data = MyDataset(config.trainroot, config.MIN_LEN, config.MAX_LEN, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False, num_workers=0)

    pbar = tqdm(total=len(train_loader))
    for i, (img, cls, regr) in enumerate(train_loader):
        print(img.shape)
        break
    pbar.close()
