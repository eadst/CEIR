# CEIR
This project is for the SPIE paper - Novel Receipt Recognition with Deep Learning Algorithms.
In this paper, we propose an end-to-end novel receipt recognition system for capturing effective information from receipts (CEIR).
The CEIR has three parts: preprocess, detection, recognition.

## Introduction

In preprocessing method, by converting the image to gray scale and obtaining the gradient with the Sobel operator, the outline of the receipt area is decided by morphological transformations with the elliptic kernel. 

In text detection, the modified connectionist text proposal network to execute text detection. 
The `pytorch` implementation of detection is based on [CTPN](https://github.com/WenmuZhou/ctpn.pytorch).

In text recognition, the convolutional recurrent neural network with the connectionist temporal classification with maximum entropy regularization as a loss function to update the weights in networks and extract the characters from receipt. 
The `pytorch` implementation of recognition is based on [CRNN](https://github.com/meijieru/crnn.pytorch) and [ENESCTC](https://github.com/liuhu-bigeye/enctc.crnn).

We validate our system with the scanned receipts optical character recognition and information extraction (SROIE) [database](https://drive.google.com/drive/folders/1ShItNWXyiY1tFDM5W02bceHuJjyeeJl2).

## Dependency

1. torch==1.4
2. torchvision
3. opencv-python
4. lmdb

## Prediction

1. Download pre-trained model from [Google Drive](w) and put the file under `./detection/output/` folder. 

2. Run `python demo.py` in CEIR folder.

3. Select the receipt image.

4. The result will be saved in `./result/`.


## Training

1. Preprocess parameters can be changed in `./preprocess/crop.py`.

2. In detection part, the `./detection/config.py` is used for configuring. After that, run `python train.py` in detection folder.

Preprocess parameters can be changed in `./preprocess/crop.py`.

3. In recognition, you need to change trainroot and other parameters in `train.sh`, then run `sh train.sh` to train.
