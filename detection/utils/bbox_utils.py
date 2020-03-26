#-*- coding:utf-8 -*-
#'''
# Created on 18-12-11 上午10:05
#
# @Author: Greg Gao(laygin)
#'''
import numpy as np
from config import *


def gen_anchor(featuresize, scale):
    """
    生成默认锚框 gen base anchor from feature map [HXW][9][4]
        reshape  [HXW][9][4] to [HXWX9][4]
    :param featuresize: 特征图的大小
    :param scale: 特征图相对于原图的比例
    :return:
    """
    # 高度不固定
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    # 宽度固定
    widths = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]

    # gen k=9 anchor size (h,w)
    heights = np.array(heights).reshape(len(heights), 1)
    widths = np.array(widths).reshape(len(widths), 1)

    # x1,y1,x2,y3
    base_anchor = np.array([0, 0, 15, 15])
    # 锚框中心点
    xt = (base_anchor[0] + base_anchor[2]) * 0.5
    yt = (base_anchor[1] + base_anchor[3]) * 0.5

    # 锚框左上角和右下角,x1 y1 x2 y2
    x1 = xt - widths * 0.5
    y1 = yt - heights * 0.5
    x2 = xt + widths * 0.5
    y2 = yt + heights * 0.5
    base_anchor = np.hstack((x1, y1, x2, y2))

    # 开始在每一个点处生成锚框
    h, w = featuresize
    shift_x = np.arange(0, w) * scale
    shift_y = np.arange(0, h) * scale
    # apply shift
    anchor = []
    for i in shift_y:
        for j in shift_x:
            anchor.append(base_anchor + [j, i, j, i])
    return np.array(anchor).reshape((-1, 4))


def cal_iou(box1, box1_area, boxes2, boxes2_area):
    """
    box1 [x1,y1,x2,y2]
    boxes2 [Msample,x1,y1,x2,y2]
    """
    x1 = np.maximum(box1[0], boxes2[:, 0])
    x2 = np.minimum(box1[2], boxes2[:, 2])
    y1 = np.maximum(box1[1], boxes2[:, 1])
    y2 = np.minimum(box1[3], boxes2[:, 3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    iou = intersection / (box1_area + boxes2_area[:] - intersection[:])
    return iou


def cal_overlaps(boxes1, boxes2):
    """
    boxes1 [Nsample,x1,y1,x2,y2]  anchor
    boxes2 [Msample,x1,y1,x2,y2]  grouth-box

    """
    area1 = (boxes1[:, 0] - boxes1[:, 2]) * (boxes1[:, 1] - boxes1[:, 3])
    area2 = (boxes2[:, 0] - boxes2[:, 2]) * (boxes2[:, 1] - boxes2[:, 3])

    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))

    # calculate the intersection of  boxes1(anchor) and boxes2(GT box)
    for i in range(boxes1.shape[0]):
        overlaps[i][:] = cal_iou(boxes1[i], area1[i], boxes2, area2)

    return overlaps


def bbox_transfrom(anchors, gtboxes):
    """
     compute relative predicted vertical coordinates Vc ,Vh
        with respect to the bounding box location of an anchor
    """
    regr = np.zeros((anchors.shape[0], 2))
    Cy = (gtboxes[:, 1] + gtboxes[:, 3]) * 0.5
    Cya = (anchors[:, 1] + anchors[:, 3]) * 0.5
    h = gtboxes[:, 3] - gtboxes[:, 1] + 1.0
    ha = anchors[:, 3] - anchors[:, 1] + 1.0

    Vc = (Cy - Cya) / ha
    Vh = np.log(h / ha)

    return np.vstack((Vc, Vh)).transpose()


def bbox_transfor_inv(anchor, regr):
    """
        return predict bbox
    """

    Cya = (anchor[:, 1] + anchor[:, 3]) * 0.5
    ha = anchor[:, 3] - anchor[:, 1] + 1

    Vcx = regr[0, :, 0]
    Vhx = regr[0, :, 1]

    Cyx = Vcx * ha + Cya
    hx = np.exp(Vhx) * ha
    xt = (anchor[:, 0] + anchor[:, 2]) * 0.5

    x1 = xt - 16 * 0.5
    y1 = Cyx - hx * 0.5
    x2 = xt + 16 * 0.5
    y2 = Cyx + hx * 0.5
    bbox = np.vstack((x1, y1, x2, y2)).transpose()

    return bbox


def clip_box(bbox, im_shape):
    # x1 >= 0
    bbox[:, 0] = np.maximum(np.minimum(bbox[:, 0], im_shape[1] - 1), 0)
    # y1 >= 0
    bbox[:, 1] = np.maximum(np.minimum(bbox[:, 1], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    bbox[:, 2] = np.maximum(np.minimum(bbox[:, 2], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    bbox[:, 3] = np.maximum(np.minimum(bbox[:, 3], im_shape[0] - 1), 0)

    return bbox


def filter_bbox(bbox, minsize):
    ws = bbox[:, 2] - bbox[:, 0] + 1
    hs = bbox[:, 3] - bbox[:, 1] + 1
    keep = np.where((ws >= minsize) & (hs >= minsize))[0]
    return keep


def cal_rpn(imgsize, featuresize, scale, gtboxes):
    """
    进程锚框和gt之间的匹配
    :param imgsize: 输入图像大小
    :param featuresize: 产生锚框位置的特征图大小
    :param scale: 特征图相对于原图的缩放比例
    :param gtboxes: gtboxes
    :return:
    """
    imgh, imgw = imgsize

    # 生成锚框
    base_anchor = gen_anchor(featuresize, scale)

    # 计算锚框和gtboxes之间的iou
    overlaps = cal_overlaps(base_anchor, gtboxes)

    # 初始化分类的label 默认全是-1 表示忽略
    labels = np.zeros(base_anchor.shape[0]) -1

    # 选出和每一个gtbbox iou最大的锚框，并设置为正样本
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    labels[gt_argmax_overlaps] = 1

    # 选出和每一个锚框iou最大的gtbbox
    anchor_argmax_overlaps = overlaps.argmax(axis=1)
    # 计算出每一个锚框和gtbbox之间的最大iou
    anchor_max_overlaps = overlaps[range(overlaps.shape[0]), anchor_argmax_overlaps]

    # IOU > IOU_POSITIVE 大于一定阈值的设置为正样本
    labels[anchor_max_overlaps > IOU_POSITIVE] = 1
    # IOU <IOU_NEGATIVE 小于一定阈值的设置为负样本
    labels[anchor_max_overlaps < IOU_NEGATIVE] = 0

    # 超出图片范围的设置为忽略，不参与训练
    outside_anchor = np.where(
        (base_anchor[:, 0] < 0) |
        (base_anchor[:, 1] < 0) |
        (base_anchor[:, 2] >= imgw) |
        (base_anchor[:, 3] >= imgh)
    )[0]
    labels[outside_anchor] = -1

    # 对正样本进行采样，如果正样本的数量太多的话，限制正样本的数量不超过RPN_POSITIVE_NUM(default 128)
    fg_index = np.where(labels == 1)[0]
    if (len(fg_index) > RPN_POSITIVE_NUM):
        labels[np.random.choice(fg_index, len(fg_index) - RPN_POSITIVE_NUM, replace=False)] = -1

    # 对负样本进行采样，如果负样本的数量太多的话
    # 正负样本总数是256，限制正样本数目最多128，
    # 如果正样本数量小于128，差的那些就用负样本补上，凑齐256个样本
    bg_index = np.where(labels == 0)[0]
    num_bg = RPN_TOTAL_NUM - np.sum(labels == 1)
    if (len(bg_index) > num_bg):
        labels[np.random.choice(bg_index, len(bg_index) - num_bg, replace=False)] = -1

    # 对anchor坐标进行变换以便于训练
    bbox_targets = bbox_transfrom(base_anchor, gtboxes[anchor_argmax_overlaps, :])

    return [labels, bbox_targets], base_anchor


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep
