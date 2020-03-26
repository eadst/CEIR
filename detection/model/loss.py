# -*- coding: utf-8 -*-
# @Time    : 3/29/19 11:03 AM
# @Author  : zhoujun
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class CTPNLoss(nn.Module):
    def __init__(self, device, sigma=9.0):
        """Implement PSE Loss.
        """
        super(CTPNLoss, self).__init__()
        self.REGR_Loss = RPN_REGR_Loss(device, sigma)
        self.CLS_Loss = RPN_CLS_Loss(device)

    def forward(self, out_cls, out_regr, gt_cls, gt_regr):
        regr_loss = self.REGR_Loss(out_regr, gt_regr)
        cls_loss = self.CLS_Loss(out_cls, gt_cls)
        return regr_loss, cls_loss


class RPN_REGR_Loss(nn.Module):
    def __init__(self, device, sigma=9.0):
        super(RPN_REGR_Loss, self).__init__()
        self.sigma = sigma
        self.device = device

    def forward(self, input, target):
        '''
        smooth L1 loss
        :param input:y_preds
        :param target: y_true
        :return:
        '''
        try:
            cls = target[0, :, 0]
            regr = target[0, :, 1:3]
            # 只回归文本区域
            regr_keep = (cls == 1).nonzero()[:, 0]
            regr_true = regr[regr_keep]
            regr_pred = input[0][regr_keep]

            diff = torch.abs(regr_true - regr_pred)
            less_one = (diff < 1.0 / self.sigma).float()
            loss = less_one * 0.5 * diff ** 2 * self.sigma + torch.abs(1 - less_one) * (diff - 0.5 / self.sigma)
            loss = torch.sum(loss, 1)
            loss = torch.mean(loss) if loss.numel() > 0 else torch.tensor(0.0)
        except Exception as e:
            print('RPN_REGR_Loss Exception:', e)
            # print(input, target)
            loss = torch.tensor(0.0)

        return loss.to(self.device)


class RPN_CLS_Loss(nn.Module):
    def __init__(self, device):
        super(RPN_CLS_Loss, self).__init__()
        self.device = device

    def forward(self, input, target):
        y_true = target[0][0]
        cls_keep = (y_true != -1).nonzero()[:, 0]
        cls_true = y_true[cls_keep].long()
        cls_pred = input[0][cls_keep]
        loss = F.nll_loss(F.log_softmax(cls_pred, dim=-1),
                          cls_true)  # original is sparse_softmax_cross_entropy_with_logits
        # loss = nn.BCEWithLogitsLoss()(cls_pred[:,0], cls_true.float())  # 18-12-8
        loss = torch.clamp(torch.mean(loss), 0, 10) if loss.numel() > 0 else torch.tensor(0.0)
        return loss.to(self.device)
