# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : zhoujun
import cv2
import os
import config

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

import shutil
import time
import torch
from torch import nn
import torch.utils.data as Data
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset import MyDataset
from model import CTPN_Model
from model import CTPN_Model
from model.loss import CTPNLoss
from utils.utils import load_checkpoint, save_checkpoint, setup_logger


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# learning rate的warming up操作
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < config.warm_up_epoch:
        lr = 1e-6 + (config.lr - 1e-6) * epoch / (config.warm_up_epoch)
    else:
        lr = config.lr * (config.lr_gamma ** (epoch / config.lr_decay_step[0]))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def train_epoch(net, optimizer, scheduler, train_loader, device, criterion, epoch, all_step, writer, logger):
    net.train()
    train_loss = 0.
    start = time.time()
    scheduler.step()
    # lr = adjust_learning_rate(optimizer, epoch)
    lr = scheduler.get_lr()[0]
    for i, (imgs, gt_cls, gt_regr) in enumerate(train_loader):
        cur_batch = imgs.size()[0]
        imgs, gt_cls, gt_regr = imgs.to(device), gt_cls.to(device), gt_regr.to(device)
        # Forward
        cls, regr = net(imgs)
        regr_loss, clc_loss = criterion(cls, regr, gt_cls, gt_regr)
        loss = regr_loss + clc_loss
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        loss = loss.item()
        cur_step = epoch * all_step + i
        writer.add_scalar(tag='Train/all_loss', scalar_value=loss, global_step=cur_step)
        writer.add_scalar(tag='Train/regr_loss', scalar_value=regr_loss, global_step=cur_step)
        writer.add_scalar(tag='Train/clc_loss', scalar_value=clc_loss, global_step=cur_step)
        writer.add_scalar(tag='Train/lr', scalar_value=lr, global_step=cur_step)

        if i % config.display_interval == 0:
            batch_time = time.time() - start
            logger.info(
                f'[{epoch}/{config.epochs}], [{i}/{all_step}], step: {cur_step}, '
                f'{config.display_interval * cur_batch / batch_time:.3f} samples/sec, '
                f'batch_loss: {loss:.4f}, regr_loss: {regr_loss:.4f}, clc_loss: {clc_loss:.4f}, '
                f'time:{batch_time:.4f}, lr:{lr}')
            start = time.time()
    return train_loss / all_step, lr


def main():
    if config.output_dir is None:
        config.output_dir = 'output'
    if config.restart_training:
        shutil.rmtree(config.output_dir, ignore_errors=True)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    logger = setup_logger(os.path.join(config.output_dir, 'train_log'))
    logger.info(config.print())

    torch.manual_seed(config.seed)  # 为CPU设置随机种子
    if config.gpu_id is not None and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info('train with gpu {} and pytorch {}'.format(config.gpu_id, torch.__version__))
        device = torch.device("cuda:0")
        torch.cuda.manual_seed(config.seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(config.seed)  # 为所有GPU设置随机种子
    else:
        logger.info('train with cpu and pytorch {}'.format(torch.__version__))
        device = torch.device("cpu")

    train_data = MyDataset(config.trainroot, config.MIN_LEN, config.MAX_LEN, transform=transforms.ToTensor())
    train_loader = Data.DataLoader(dataset=train_data, batch_size=config.train_batch_size, shuffle=True,
                                   num_workers=int(config.workers))

    writer = SummaryWriter(config.output_dir)
    model = CTPN_Model(pretrained=config.pretrained)
    if not config.pretrained and not config.restart_training:
        model.apply(weights_init)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    dummy_input = torch.zeros(1, 3, 600, 800).to(device)
    writer.add_graph(model=model, input_to_model=dummy_input)
    criterion = CTPNLoss(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.99)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    if config.checkpoint != '' and not config.restart_training:
        print('Loading Checkpoint...')
        start_epoch = load_checkpoint(config.ch9eckpoint, model, logger, device)
        start_epoch += 1
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_decay_step, gamma=config.lr_gamma,
                                                         last_epoch=start_epoch)
    else:
        start_epoch = config.start_epoch
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_decay_step, gamma=config.lr_gamma)

    all_step = len(train_loader)
    logger.info('train dataset has {} samples,{} in dataloader'.format(train_data.__len__(), all_step))
    epoch = 0
    best_model = {'loss': float('inf')}
    try:
        for epoch in range(start_epoch, config.epochs):
            start = time.time()
            train_loss, lr = train_epoch(model, optimizer, scheduler, train_loader, device, criterion, epoch, all_step,
                                         writer, logger)
            logger.info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
                epoch, config.epochs, train_loss, time.time() - start, lr))
            # if (0.3 < train_loss < 0.4 and epoch % 1 == 0) or train_loss < 0.3:
            if epoch % 10 == 0 or train_loss < best_model['loss']:
                net_save_path = '{}/PSENet_{}_loss{:.6f}.pth'.format(config.output_dir, epoch, train_loss)
                save_checkpoint(net_save_path, model, optimizer, epoch, logger)
                if train_loss < best_model['loss']:
                    best_model['loss'] = train_loss
                    if 'model' in best_model:
                        os.remove(best_model['model'])
                    best_model['model'] = net_save_path
                    shutil.copy(best_model['model'],
                                '{}/best_loss{:.6f}.pth'.format(config.output_dir, best_model['loss']))
        writer.close()
    except KeyboardInterrupt:
        pass
    finally:
        if best_model['model']:
            shutil.copy(best_model['model'], '{}/best_loss{:.6f}.pth'.format(config.output_dir, best_model['loss']))
            logger.info(best_model)


if __name__ == '__main__':
    main()
