# -*- coding: utf-8 -*-
'''
Stage 1: preprocessing for training
Last time for updating: 04/15/2020
'''

# @Time    : 1/4/19 11:18 AM
# @Author  : zhoujun
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt


def show_img(imgs: np.ndarray, color=True):
    if (len(imgs.shape) == 3 and color) or (len(imgs.shape) == 2 and not color):
        imgs = np.expand_dims(imgs, axis=0)
    for img in imgs:
        plt.figure()
        plt.imshow(img, cmap=None if color else 'gray')


def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2, mode=0):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        if mode == 0:
            point = point.astype(int)
        else:
            score = point[-1]
            point = point[0].astype(int)
            # cv2.putText(img_path, f'{score:.2f}', tuple(point[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.line(img_path, tuple(point[0]), tuple(point[1]), color, thickness)
        cv2.line(img_path, tuple(point[1]), tuple(point[2]), color, thickness)
        cv2.line(img_path, tuple(point[2]), tuple(point[3]), color, thickness)
        cv2.line(img_path, tuple(point[3]), tuple(point[0]), color, thickness)
    return img_path


def draw_anchor(img_path, anchor, color=(255, 0, 0), thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for an in anchor:
        x1, y1, x2, y2 = an
        cv2.line(img_path, (x1, y1), (x2, y1), color, thickness)
        cv2.line(img_path, (x2, y1), (x2, y2), color, thickness)
        cv2.line(img_path, (x2, y2), (x1, y2), color, thickness)
        cv2.line(img_path, (x1, y2), (x1, y1), color, thickness)
    return img_path


def setup_logger(log_file_path: str = None):
    import logging
    from colorlog import ColoredFormatter
    logging.basicConfig(filename=log_file_path, format='%(asctime)s %(levelname)-8s %(filename)s: %(message)s',
                        # 定义输出log的格式
                        datefmt='%Y-%m-%d %H:%M:%S', )
    """Return a logger with a default ColoredFormatter."""
    formatter = ColoredFormatter("%(asctime)s %(log_color)s%(levelname)-8s %(reset)s %(filename)s: %(message)s",
                                 datefmt='%Y-%m-%d %H:%M:%S',
                                 reset=True,
                                 log_colors={
                                     'DEBUG': 'blue',
                                     'INFO': 'green',
                                     'WARNING': 'yellow',
                                     'ERROR': 'red',
                                     'CRITICAL': 'red',
                                 })

    logger = logging.getLogger('ctpn.pytorch')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info('logger init finished')
    return logger


def save_checkpoint(checkpoint_path, model, optimizer, epoch, logger):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch}
    torch.save(state, checkpoint_path)
    logger.info('model saved to %s' % checkpoint_path)


def load_checkpoint(checkpoint_path, model, logger, device, optimizer=None):
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']
    logger.info('model loaded from %s' % checkpoint_path)
    return start_epoch


# --exeTime
def exe_time(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        print("{} cost {:.3f}s".format(func.__name__, time.time() - t0))
        return back

    return newFunc
