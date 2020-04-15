# -*- coding: UTF-8 -*-
'''
Stage 1: preprocessing for predicting
Last time for updating: 04/15/2020
'''


import os
import cv2
import numpy as np


class Crop():
    def __init__(self, path):
        self.path = path

    def get_image(self, path):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray

    # gradient
    def step1(self, gray):
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        # Sobel_gradient
        gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
        gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        # thresh_and_blur
        blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
        (_, thresh) = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)
        return thresh

    # Need to set the ellipse size at first and do morphological thing.
    def step2(self, thresh):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(thresh.shape[1] / 40), int(thresh.shape[0] / 18)))
        morpho_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morpho_image = cv2.erode(morpho_image, None, iterations=1)
        morpho_image = cv2.dilate(morpho_image, None, iterations=1)
        return morpho_image

    # Get contour's points, draw Contours and crop image.
    def step3(self, morpho_image, original_img):
        cnts, hierarchy = cv2.findContours(morpho_image.copy(),
                                           cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)
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
        return draw_img, crop_img, x1, y1

    def main(self):
        file_path = self.path
        original_img, gray = self.get_image(file_path)
        if original_img.shape[1] > 990:
            gradient = self.step1(gray)
            morpho_image = self.step2(gradient)
            draw_img, crop_img, weight, height = self.step3(morpho_image, original_img)
        else:
            crop_img = original_img
            draw_img = original_img
        new_name = file_path
        name = new_name.split('/')[-1]
        main_path = os.path.abspath(os.path.join(os.getcwd()))
        print(main_path)
        image_path = os.path.join(main_path, 'result/step1/image/')
        image_save_path = image_path + name
        cv2.imwrite(image_save_path, crop_img)
        image_save_draw = image_path + 'draw_' + name
        cv2.imwrite(image_save_draw, draw_img)
        print('Preprocess Finished.')
        print('Created Image: ', image_save_path)
        print('Created Image: ', image_save_draw)


if __name__ == '__main__':
    crop = Crop('demo.jpg')
    crop.main()