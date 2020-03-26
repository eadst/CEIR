# -*- coding: UTF-8 -*-
import os
import copy
import cv2
import numpy as np

class Crop():
    def __init__(self, path):
        self.path = path

    def get_image(self, path):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray


    def Gaussian_Blur(self, gray):
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        return blurred


    def Sobel_gradient(self, blurred):
        gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
        gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        return gradX, gradY, gradient


    def Thresh_and_blur(self, gradient):
        blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
        (_, thresh) = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)
        return thresh


    def image_morphology(self, thresh):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(thresh.shape[1]/5), int(thresh.shape[0]/18)))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.erode(closed, None, iterations=1)
        closed = cv2.dilate(closed, None, iterations=1)
        return closed

    def findcnts_and_box_point(self, closed):
        cnts, hierarchy = cv2.findContours(closed.copy(),
                                              cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        return box


    def drawcnts_and_cut(self, original_img, box):
        # draw a bounding box arounded the detected barcode and display the image
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
        crop_img = copy.deepcopy(original_img[y1:y1 + hight, x1:x1 + width])
        draw_img = cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
        return draw_img, crop_img, x1, y1

    def main(self):
        file_path = self.path
        original_img, gray = self.get_image(file_path)
        if original_img.shape[1] > 990:
            blurred = self.Gaussian_Blur(gray)
            gradX, gradY, gradient = self.Sobel_gradient(blurred)
            thresh = self.Thresh_and_blur(gradient)
            closed = self.image_morphology(thresh)
            box = self.findcnts_and_box_point(closed)
            draw_img, crop_img, w, h = self.drawcnts_and_cut(original_img, box)
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