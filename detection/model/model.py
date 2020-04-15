# -*- coding: utf-8 -*-
'''
Stage 2: detection
Last time for updating: 04/15/2020
'''

# @Time    : 2019/1/2 17:29
# @Author  : zhoujun
import torch
from torch import nn
import torchvision.models as models


class BasicConv(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 relu=True,
                 bn=True,
                 bias=True):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CTPN_Model(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        base_model = models.vgg16(pretrained)
        layers = list(base_model.features)[:-1]
        self.base_layers = nn.Sequential(*layers)  # block5_conv3 output
        self.rpn = BasicConv(in_planes=512, out_planes=512, kernel_size=3, padding=1, bn=False)
        self.brnn = nn.GRU(512, 128, bidirectional=True, batch_first=False)
        self.lstm_fc = BasicConv(in_planes=256, out_planes=512, kernel_size=1, relu=True, bn=False)
        self.rpn_class = BasicConv(in_planes=512, out_planes=10 * 2, kernel_size=1, relu=False, bn=False)
        self.rpn_regress = BasicConv(in_planes=512, out_planes=10 * 2, kernel_size=1, relu=False, bn=False)

    def forward(self, x):
        # 抽取特征
        x = self.base_layers(x)

        # 其实就是一个3*3卷积 融合一下周围的特征
        x = self.rpn(x)
        # 进lstm之前的预处理
        x1 = x.permute(0, 2, 3, 1).contiguous()  # channels last
        b = x1.size()  # batch_size, h, w, c
        x1 = x1.view(b[0] * b[1], b[2], b[3])  # (b,h,w,c) to (b*h,w,c)
        # x1 = x1.permute(1, 0, 2)  # (b*h,w,c) to (w,b*h,c)
        x2, _ = self.brnn(x1)

        # 进卷积之前需要将通道数缓过来
        xsz = x.size()
        x3 = x2.view(xsz[0], xsz[2], xsz[3], 256)  # (b, h, w, 256)
        x3 = x3.permute(0, 3, 1, 2).contiguous()  # channels first

        x3 = self.lstm_fc(x3)
        x = x3

        # 进行分类和回归
        cls = self.rpn_class(x)
        regr = self.rpn_regress(x)

        # (b,c,h,w) to (b,h,w,c)
        cls = cls.permute(0, 2, 3, 1).contiguous()
        regr = regr.permute(0, 2, 3, 1).contiguous()

        # (b,h,w,c) to (b,h*w*10,2)
        cls = cls.view(cls.size(0), cls.size(1) * cls.size(2) * 10, 2)
        # (b,h,w,c) to (b,h*w*10,4)
        regr = regr.view(regr.size(0), regr.size(1) * regr.size(2) * 10, 2)

        return cls, regr


if __name__ == '__main__':
    device = torch.device('cpu')
    net = CTPN_Model(pretrained=False).to(device)
    input = torch.zeros(1, 3, 608, 1072)
    y = net(input)
    print(net)
