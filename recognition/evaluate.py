# -*- coding: utf-8 -*-
'''
Stage 3: recognition
Last time for updating: 04/15/2020
'''
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import glob
import os
import models.crnn as crnn



def sort_lines(txt_file):
    newlist = []
    f = open(txt_file)
    for line in f.readlines():
        newlist.append([int(line.split(',')[1]),line])
    sortedlines = sorted(newlist, key=(lambda x: x[0]))
    new_txt = []
    for line in sortedlines:
        new_txt.append(line[1])
    return new_txt


def predict_this_box(image, model, alphabet):
    converter = utils.strLabelConverter(alphabet)
    transformer = dataset.resizeNormalize((200, 32))
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    # print(preds.data)
    # print('%-30s ==> %-30s' % (raw_pred, sim_pred))
    return sim_pred


def load_images_to_predict():
    print('load_images_to_predict')
    # load model
    model_path = './save/netCRNN_225_1250.pth'
    alphabet = '0123456789,.:(%$!^&-/);<~|`>?+=_[]{}"\'@#*abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\ '
    imgH = 32 # should be 32
    nclass = len(alphabet) + 1
    nhiddenstate = 256

    model = crnn.CRNN(imgH, 1, nclass, nhiddenstate)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from %s' % model_path)
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(model_path, map_location='cpu').items()})

    # load image
    main_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    image_path = os.path.join(main_path, 'result/step2/image2/*.jpg')
    label_path = os.path.join(main_path, 'result/step2/label/')
    image_list = [os.path.splitext(f)[0] for f in glob.glob(image_path)]
    jpg_files = [name + ".jpg" for name in image_list]
    print('total images: ', len(jpg_files))
    count = 1
    for jpg in jpg_files:
        image = Image.open(jpg).convert('L')
        words_list = []
        label = label_path + jpg.split('/')[-1][:-3] + 'txt'
        txt = sort_lines(label)
        for line in txt:
            box = line.split(',')
            crop_image = image.crop((int(box[0]), int(box[1]), int(box[4]), int(box[5])))
            words = predict_this_box(crop_image, model, alphabet)
            words_list.append(words)
        result_path = os.path.join(main_path, 'result/step3/')
        save_path = result_path + jpg.split('/')[-1][:-3] + 'txt'
        with open(save_path, 'w+') as result:
            for line in words_list:
                result.writelines(line+'\n')
        if count % 1000 == 0:
            print(str(count), 'images finished.', 'total images: ', len(jpg_files))
        count += 1


if __name__ == "__main__":
    load_images_to_predict()
