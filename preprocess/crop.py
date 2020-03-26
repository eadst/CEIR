# -*- coding: UTF-8 -*-

'''
Author: Steve Wang
Time: 2017/12/8 10:00
Environment: Python 3.6.2 |Anaconda 4.3.30 custom (64-bit) Opencv 3.3
Modified: Dong
Time: 02/17/2020
'''
import os
import shutil
import cv2
import numpy as np


def get_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def Gaussian_Blur(gray):
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    return blurred


def Sobel_gradient(blurred):
    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    return gradX, gradY, gradient


def Thresh_and_blur(gradient):
    blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
    (_, thresh) = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)
    return thresh


def image_morphology(thresh):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(thresh.shape[1]/40), int(thresh.shape[0]/18)))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=1)
    closed = cv2.dilate(closed, None, iterations=1)
    return closed


def findcnts_and_box_point(closed):
    cnts, hierarchy = cv2.findContours(closed.copy(),
                                    cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    return box


def drawcnts_and_cut(original_img, box):
    # draw a bounding box arounded the detected barcode and display the image
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)  # (143, 82, 47)
    height = original_img.shape[0]
    weight = original_img.shape[1]
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = max(min(Xs), 0)
    x2 = min(max(Xs), weight)
    y1 = max(min(Ys), 0)
    y2 = min(max(Ys), height)
    hight = y2 - y1
    width = x2 - x1
    crop_img = original_img[y1:y1 + hight, x1:x1 + width]
    # print('crop_img.shape: ', crop_img.shape)
    # print('x1, y1', x1, y1)
    return draw_img, crop_img, x1, y1


def test():
    main_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    image_path = os.path.join(main_path, 'dataset/train/image/')
    file_path = image_path + 'X51005433541.jpg'
    original_img, gray = get_image(file_path)
    blurred = Gaussian_Blur(gray)
    gradX, gradY, gradient = Sobel_gradient(blurred)
    thresh = Thresh_and_blur(gradient)
    closed = image_morphology(thresh)
    box = findcnts_and_box_point(closed)
    draw_img, crop_img, w, h = drawcnts_and_cut(original_img, box)
    image_save_path = '/home/dong/Downloads/receipt/CEIR/result/X51005442376.jpg'
    # cv2.imshow('original_img', original_img)
    # cv2.imshow('blurred', blurred)
    # cv2.imshow('gradX', gradX)
    # cv2.imshow('gradY', gradY)
    # cv2.imshow('final', gradient)
    # cv2.imshow('thresh', thresh)
    # cv2.imshow('closed', closed)
    # cv2.imshow('draw_img', draw_img)
    save_gradient = r'/home/dong/Downloads/receipt/CEIR/result/X51006414631_gradient.png'
    cv2.imwrite(save_gradient, gradient)
    save_thresh = r'/home/dong/Downloads/receipt/CEIR/result/X51006414631_thresh.png'
    cv2.imwrite(save_thresh, thresh)
    save_closed = r'/home/dong/Downloads/receipt/CEIR/result/X51006414631_closed.png'
    cv2.imwrite(save_closed, closed)
    save_draw_img = r'/home/dong/Downloads/receipt/CEIR/result/X51006414631_draw_img.png'
    cv2.imwrite(save_draw_img, draw_img)
    # cv2.imshow('crop_img', crop_img)
    # cv2.waitKey(99999999)


def adjust_label(img, txt_path, w_dif, h_dif):
    new_info = ''
    file = open(txt_path, 'r')
    for line in file.readlines():
        info = line.strip('\r\n').split(',')
        new_info += str(int(info[0]) - w_dif) + ','
        new_info += str(int(info[1]) - h_dif) + ','

        new_info += str(int(info[2]) - w_dif) + ','
        new_info += str(int(info[3]) - h_dif) + ','

        new_info += str(int(info[4]) - w_dif) + ','
        new_info += str(int(info[5]) - h_dif) + ','

        new_info += str(int(info[6]) - w_dif) + ','
        new_info += str(int(info[7]) - h_dif)
        for word in info[8:]:
            new_info += ','
            new_info += word
        new_info += '\r\n'
        # box = [int(info[0]) - w_dif, int(info[1]) - h_dif,
        #        int(info[2]) - w_dif, int(info[3]) - h_dif,
        #        int(info[4]) - w_dif, int(info[5]) - h_dif,
        #        int(info[6]) - w_dif, int(info[7]) - h_dif]
        # print(box)
        # img = cv2.rectangle(img,(int(info[0]) - w_dif, int(info[1]) - h_dif),(int(info[4]) - w_dif, int(info[5]) - h_dif),(0, 0, 255),3)

    file.close()
    # cv2.imshow('img', img)
    #
    # print(w_dif, h_dif)
    # cv2.waitKey(0)
    return new_info


def crop():
    main_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    image_path = os.path.join(main_path, 'dataset/train/failed/')
    label_path = os.path.join(main_path, 'dataset/train/label/')
    image_save_path = os.path.join(main_path, 'result/step1/image/')
    label_save_path = os.path.join(main_path, 'result/step1/label/')
    count = 0
    for _, _, names in os.walk(image_path):
        filenames = []
        for name in names:
            if name[-3:] == 'jpg':
                filenames.append(name)
        print('Total images: ', len(filenames))
        for file in filenames:
            file_path = image_path + file
            txt_path = label_path + file[:-3] + 'txt'
            # file_path = image_path + 'X51005442333.jpg'
            # txt_path = label_path + 'X51005442333.txt'
            print('Cropping image: ', file)

            try:
                original_img, gray = get_image(file_path)
                if original_img.shape[1] > 990:
                    blurred = Gaussian_Blur(gray)
                    gradX, gradY, gradient = Sobel_gradient(blurred)
                    thresh = Thresh_and_blur(gradient)
                    closed = image_morphology(thresh)
                    box = findcnts_and_box_point(closed)
                    draw_img, crop_img, weight, height = drawcnts_and_cut(original_img, box)
                    new_image_save_path = image_save_path + file
                    if crop_img.size == 0:
                        crop_img = original_img
                    cv2.imwrite(new_image_save_path, crop_img)

                    new_label = adjust_label(crop_img, txt_path, weight, height)
                    file_txt_path = label_save_path + file[:-3] + 'txt'
                    with open(file_txt_path, "w") as f:
                        f.write(new_label)
                else:
                    shutil.copy(file_path, image_save_path)
                    shutil.copy(txt_path, label_save_path)
            except BaseException:
                print(BaseException)
                print('Cropping is not executed.')
                shutil.copy(file_path, image_save_path)
                shutil.copy(txt_path, label_save_path)
            count += 1
            print('Finished ', '%.2f%%' % (count/len(filenames) * 100))


if __name__ == '__main__':
    # test()
    crop()
