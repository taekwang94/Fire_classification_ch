from __future__ import print_function
import os
import random
import math
import sys
import time
import numpy as np
import shutil
import argparse
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset

from torchvision import datasets, transforms
from torchvision.models import resnet18, shufflenetv2

from torch.autograd import Variable
import dataset

from models.mobilenet import mobilenet_v2
from models.shufflenetv2 import shufflenet_v2_x1_0
from models.mobilenetv3 import mobilenetv3_small
from models.ghostnet import ghostnet
from models.mnasnet import mnasnet1_0
from models.mobilenetv2_ch_in_model import mobilenet_v2_in_ch
from timeit import default_timer as timer

from tools.tool import EarlyStopping
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# Multi-GPUs
from torch.nn.parallel import DistributedDataParallel as DDP

warnings.filterwarnings("ignore")
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:', device)


def main():
    parser = argparse.ArgumentParser(description='FireClassification')
    parser.add_argument('--model', default='ghostnet')  # resnet18, mobilenet_v2, sh
    parser.add_argument('--train_save_path', default='/home/taekwang0094/WorkSpace/FireTraining')
    parser.add_argument('--multi_gpus', default=True)
    parser.add_argument('--root', default='/home/taekwang0094/WorkSpace/Summer_Conference')
    parser.add_argument('--channel_multiplier', default=[1.0,1.0,1.0])  # -l 추가해서 list로 받도록 수정할것
    parser.add_argument('--batch_size', default=258)
    parser.add_argument('--epoch', default=100)
    parser.add_argument('--early_stop', default=True)
    parser.add_argument('--lr_scheduler', default=True)

    args = parser.parse_args()
    print(args.model)

    if type(args.channel_multiplier) == str:
        tmp = args.channel_multiplier
        tmp = tmp.replace('[',"")
        tmp = tmp.replace(']',"")
        tmp = tmp.split(',')
        channel_multiplier = [float(tmp[0]),float(tmp[1]),float(tmp[2])]
    else:
        channel_multiplier = args.channel_multiplier

    if type(args.multi_gpus) ==str:
        if args.multi_gpus == 'True':
            multi_gpus = True
        else:
            multi_gpus = False
    else:
        multi_gpus = args.multi_gpus

    if type(args.early_stop) == str:
        if args.early_stop == 'True':
            is_early_stop = True
        else:
            False
    else:
        is_early_stop = args.early_stop

    if type(args.lr_scheduler) == str:
        if args.lr_scheduler == 'True':
            is_lr_scheduler = True
        else:
            False
    else:
        is_lr_scheduler = args.lr_scheduler

    print("Multi gpus :" , multi_gpus)
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    train_loader = torch.utils.data.DataLoader(
        dataset.FireDataset(args.root, transforms=data_transforms['train'], channel_multiplier=channel_multiplier, ch_preprocess=True),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=8
    )
    val_loader = torch.utils.data.DataLoader(
        dataset.FireDataset(args.root, train='val', transforms=data_transforms['val'],
                            channel_multiplier=channel_multiplier, ch_preprocess=True),
        batch_size=128,
        shuffle=False,
        num_workers=8
    )
    if args.model == 'resnet18':
        model = resnet18()
        num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs,1)
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 2),
            nn.Sigmoid()
        )
    elif args.model == 'mobilenet_v2':
        model = mobilenet_v2(num_classes=2)
    elif args.model == 'shufflenet_v2_x1_0':
        model = shufflenet_v2_x1_0(num_classes=2)
    elif args.model == 'mobilenetv3_small':
        model = mobilenetv3_small(num_classes=2)
    elif args.model == 'ghostnet':
        model = ghostnet(num_classes=2)
    elif args.model == 'mnasnet1_0':
        model = mnasnet1_0(num_classes=2)
    elif args.model == 'mobilenet_v2_in_ch':
        model = mobilenet_v2_in_ch(num_classes = 2)

    print("Training Start , model : ", args.model)

    save_dir = args.model + "_[{0},{1},{2}]_2".format(channel_multiplier[0], channel_multiplier[1], channel_multiplier[2])
    # save_dir = os.path.join(args.model,'_[{0},{1},{2}]'.format(channel_multiplier[0],channel_multiplier[1],channel_multiplier[2]))
    save_dir_path = os.path.join(args.train_save_path, save_dir)
    #save_dir_path = os.path.join(args.train_save_path, args.model)
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    if multi_gpus:
        model = torch.nn.DataParallel(model)
        model.cuda()

    else:
        #model = model.to(device)
        model = model.cuda()

    if args.model == 'mnasnet1_0':
        optimizer = optim.SGD(model.parameters(), lr=0.003)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    #LR scheduler
    if is_lr_scheduler:
        lr_scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor=0.5, patience=5, verbose=1)

    criterion = torch.nn.BCELoss().to(device)

    best_val_loss = sys.maxsize

    file_txt = open(os.path.join(save_dir_path, 'training_log.txt'), 'w')

    time_1 = 0
    time_2 = 0

    # Early Stop
    if is_early_stop:
        early_stopping = EarlyStopping(patience=7, verbose=True)

    # 모델이 학습되는 동안 trainning loss를 track
    train_losses = []
    # 모델이 학습되는 동안 validation loss를 track
    valid_losses = []
    # epoch당 average training loss를 track
    avg_train_losses = []
    # epoch당 average validation loss를 track
    avg_valid_losses = []

    for epochs in range(int(args.epoch)):
        model.train()

        train_correct = 0
        train_total = 0

        val_correct = 0
        val_total = 0
        time_1 = timer()
        for batch_idx, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)
            # image, label = Variable(image).to(device), Variable(label).to(device)
            output = model(image).to(device)
            # print(output)

            cost = criterion(output, label).to(device)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            # print(output)

            train_losses.append(cost.item())

            _, predicted = torch.max(output.data, 1)
            # print("ASD",predicted)
            # print(torch.argmax(label,1))
            train_label_eval = torch.argmax(label, 1)
            train_total += label.size(0)
            train_correct += (predicted == train_label_eval).sum().item()

            print('{0}% 완료,\r'.format(int(batch_idx / len(train_loader) * 100)), end="")

        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': cost,
        }, os.path.join(save_dir_path, 'model_epoch{0}_{1}.pt'.format(epochs, round(cost.item(), 4))))

        for count, (image, label) in enumerate(val_loader):
            image = image.to(device)
            label = label.to(device)

            model.eval()
            with torch.no_grad():
                output = model(image).to(device)
                val_loss = criterion(output, label)
                valid_losses.append(val_loss.item())
                _, predicted = torch.max(output.data, 1)
                val_label_eval = torch.argmax(label, 1)
                val_total += label.size(0)
                val_correct += (predicted == val_label_eval).sum().item()
            print('{0}% 완료,\r'.format(int(count / len(val_loader) * 100)), end="")

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            print("best model renew.")
            torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': cost,
            }, os.path.join(save_dir_path, 'model_best.pt'))
        train_accuracy = round(float(100 * train_correct / train_total), 4)
        val_accuracy = round(float(100 * val_correct / val_total), 4)

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        train_losses = []
        valid_losses = []

        if is_early_stop:
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("Epoch : ", epochs, " Training Loss : ", round(cost.item(), 4), " Training Accuracy : ", train_accuracy,
              " Validation Loss : ", round(val_loss.item(), 4), " Validation Accuracy : ", val_accuracy)
        word = "Epoch : {0} Training Loss : {1} Training Accuracy {2} Validation Loss : {3} Validation Accuracy {4} \n".format(
            epochs, round(cost.item(), 4), train_accuracy, round(val_loss.item(), 4), val_accuracy)
        file_txt.write(word)
        time_2 = timer()
        print("Train time / epoch : ", time_2 - time_1)
        if is_lr_scheduler:
            lr_scheduler.step(val_loss)

    file_txt.close()


if __name__ == "__main__":
    main()
