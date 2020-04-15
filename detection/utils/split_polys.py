# -*- coding: utf-8 -*-
'''
Stage 2: detection
Last time for updating: 04/15/2020
'''

# @Time    : 2019/5/10 14:22
# @Author  : zhoujun
import numpy as np
from shapely.geometry import Polygon


def pickTopLeft(poly):
    idx = np.argsort(poly[:, 0])
    if poly[idx[0], 1] < poly[idx[1], 1]:
        s = idx[0]
    else:
        s = idx[1]

    return poly[(s, (s + 1) % 4, (s + 2) % 4, (s + 3) % 4), :]


def orderConvex(p):
    points = Polygon(p).convex_hull
    points = np.array(points.exterior.coords)[:4]
    points = points[::-1]
    points = pickTopLeft(points)
    points = np.array(points).reshape([4, 2])
    return points


def shrink_poly(poly, r=16):
    """
    将一个矩形框切分成多个宽度为16的小框
    :param poly:矩形框
    :param r: 切分时宽度的步长
    :return:
    """
    # y = kx + b
    x_min = int(np.min(poly[:, 0]))
    x_max = int(np.max(poly[:, 0]))
    # 计算上下边的直线系数，k,b
    k1 = (poly[1][1] - poly[0][1]) / (poly[1][0] - poly[0][0])
    b1 = poly[0][1] - k1 * poly[0][0]

    k2 = (poly[2][1] - poly[3][1]) / (poly[2][0] - poly[3][0])
    b2 = poly[3][1] - k2 * poly[3][0]

    res = []

    start = int((x_min // 16 + 1) * 16)
    end = int((x_max // 16) * 16)

    p = x_min
    res.append([p, int(k1 * p + b1),
                start - 1, int(k1 * (p + 15) + b1),
                start - 1, int(k2 * (p + 15) + b2),
                p, int(k2 * p + b2)])

    for p in range(start, end + 1, r):
        res.append([p, int(k1 * p + b1),
                    (p + 15), int(k1 * (p + 15) + b1),
                    (p + 15), int(k2 * (p + 15) + b2),
                    p, int(k2 * p + b2)])
    return np.array(res, dtype=np.int).reshape([-1, 8])

def split_polys(polys):
    res_polys = []
    show_polys = []
    for poly in polys:
        poly = orderConvex(poly) # 猜测是保证 左上角，右上角，右下角，左下角排序的
        # delete polys with width less than 10 pixel 任意一边小于10个像素的不要
        if np.linalg.norm(poly[0] - poly[1]) < 10 or np.linalg.norm(poly[3] - poly[0]) < 10:
            continue
        # 将一个矩形框切分成多个宽度为16的小框
        res = shrink_poly(poly)
        # for p in res:
        #    cv.polylines(re_im, [p.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=1)

        res = res.reshape([-1, 4, 2])
        for r in res:
            x_min = np.min(r[:, 0])
            y_min = np.min(r[:, 1])
            x_max = np.max(r[:, 0])
            y_max = np.max(r[:, 1])
            # 计算出外接水平矩形
            res_polys.append([x_min, y_min, x_max, y_max])
        show_polys.append(res)
    return np.array(res_polys),np.vstack(show_polys)