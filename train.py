# -*- coding:utf-8 -*-
from config import config
from models.resnet import *
from dataset.dataset import RSDataset
from lr_schedule import step_lr
from utils.plot import draw_curve

import os
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms


def train():
    if config.model == 'resnet18':
        model = resnet18()
    elif config.model == 'resnet34':
        model = resnet18()
    elif config.model == 'resnet50':
        model = resnet18()
    elif config.model == 'resnet101':
        model = resnet18()
    elif config.model == 'resnet152':
        model = resnet18()
    else:
        raise ValueError('No {} models',format(config.model))
    model.cuda()

    # loss
    criterion = nn.CrossEntropyLoss().cuda()

    # train data
    transform = transforms.Compose(
        [
            transforms.Scale(600),
            transforms.RandomSizedCrop(500),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Resize((config.width, config.height)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )
    dst_train = RSDataset(r'.\datalist\train.txt', transform=transform)
    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=config.batch_size)
    
    # validation data
    transform = transforms.Compose(
        [
            transforms.Resize((config.width, config.height)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )
    dst_valid = RSDataset(r'.\datalist\valid.txt', transform=transform)
    dataloader_valid = DataLoader(dst_valid, shuffle=False, batch_size=int(config.batch_size/2))

    # log
    if not os.path.exists('./log'):
        os.makedirs('./log')
    log = open('./log/log.txt', 'a')

    log.write('-' * 30 + '\n')
    log.write(
        'model:{}\nnum_classes:{}\nnum_epoch:{}\nlearning_rate:{}\nim_width:{}\nim_height:{}\niter_smooth:{}\n'.format(
            config.model, config.num_classes, config.num_epochs, config.lr,
            config.width, config.height, config.iter_smooth))

    # train
    train_draw_acc = []
    val_draw_acc = []
    max_acc = [0.0, 0.0]
    for epoch in range(config.num_epochs):
        ep_start = time.time()

        # lr
        lr = step_lr(epoch)

        # optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.0002)

        model.train()
        train_acc_sum = 0.0
        train_loss_sum = 0.0
        for i, (ims, label) in enumerate(dataloader_train):
            input = Variable(ims).cuda()
            target = Variable(label).cuda().long()

            output = model(input)

            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = output.max(1)
            train_acc_sum += pred.eq(target).sum()
            train_loss_sum += loss.item()

        train_acc = train_acc_sum.float() / len(dst_train)
        train_draw_acc.append(train_acc_sum/train_acc)

        epoch_time = (time.time() - ep_start) / 60.
        if epoch < config.num_epochs:
            # eval
            val_time_start = time.time()
            val_loss_sum = 0.0
            val_acc_sum = 0.0

            model.eval()
            for ims, label in dataloader_valid:
                input_val = Variable(ims).cuda()
                target_val = Variable(label).cuda()
                output_val = model(input_val)
                loss = criterion(output_val, target_val)

                _, pred = output_val.max(1)
                val_acc_sum += pred.eq(target_val).sum()
                val_loss_sum += loss.item()

            val_loss = val_loss_sum / len(dst_valid)
            val_acc = val_acc_sum.float() / len(dst_valid)

            val_draw_acc.append(val_acc)
            val_time = (time.time() - val_time_start) / 60.

            if train_acc > max_acc[0]:
                max_acc[0] = train_acc
            if val_acc > max_acc[1]:
                max_acc[1] = val_acc
            print('Epoch [%d/%d], train_loss: %.4f, train_acc: %.4f, epoch_time: %.4f, Val_Loss: %.4f, Val_acc: %.4f, val_time: %.4f s, max_train_acc: %.4f, max_valid_acc: %.4f'
                % (epoch + 1, config.num_epochs, train_loss_sum / len(dst_train), train_acc, epoch_time * 60, val_loss,
                   val_acc, val_time * 60, max_acc[0], max_acc[1]))

            log.write('Epoch [%d/%d], train_loss: %.4f, train_acc: %.4f, epoch_time: %.4f, Val_Loss: %.4f, Val_acc: %.4f, val_time: %.4f s, max_train_acc: %.4f, max_valid_acc: %.4f\n'
                % (epoch + 1, config.num_epochs, train_loss_sum / len(dst_train), train_acc, epoch_time * 60, val_loss,
                   val_acc, val_time * 60, max_acc[0], max_acc[1]))
            draw_curve(train_draw_acc, val_draw_acc)
    log.write('-' * 30 + '\n')
    log.close()


if __name__=='__main__':
    train()