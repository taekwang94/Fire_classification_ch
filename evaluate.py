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
from torchvision.models import resnet18
from torch.autograd import Variable
import dataset

from models.mobilenet import mobilenet_v2
from models.shufflenetv2 import shufflenet_v2_x1_0
from models.mobilenetv3 import mobilenetv3_small
from models.ghostnet import ghostnet
from torchvision.models import MNASNet

from timeit import default_timer as timer

warnings.filterwarnings("ignore")
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)




def main():
    parser = argparse.ArgumentParser(description='FireClassification')
    parser.add_argument('--model', default='mobilenet_v2')
    parser.add_argument('--train_save_path', default='/home/taekwang0094/WorkSpace/FireTraining')
    parser.add_argument('--multi_gpus', default=True)
    parser.add_argument('--root', default='/home/taekwang0094/WorkSpace/Summer_Conference')
    parser.add_argument('--channel_multiplier', default=[1.3,1,1])  # -l 추가해서 list로 받도록 수정할것
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--epoch', default=100)

    args = parser.parse_args()
    #checkpoint_path = '/home/taekwang0094/WorkSpace/FireTraining/shufflenet_v2_x1_0_no_channel_multiplier/model_best.pt'
    #checkpoint_path = '/home/taekwang0094/WorkSpace/FireTraining/shufflenet_v2_x1_0_1.2/model_epoch77_0.0342.pt'
    #checkpoint_path = '/home/taekwang0094/WorkSpace/FireTraining/mobilenet_v2_no_channel_multiplier/model_best.pt'
    checkpoint_path = '/home/taekwang0094/WorkSpace/FireTraining/mobilenet_v2_1.2/model_best.pt'
    #checkpoint_path = '/home/taekwang0094/WorkSpace/FireTraining/mobilenetv3_small_no_channel_multiplier/model_epoch75_0.0944.pt'
    #checkpoint_path = '/home/taekwang0094/WorkSpace/FireTraining/mobilenetv3_small_1.2/model_best.pt'

    #checkpoint_path = '/home/taekwang0094/WorkSpace/FireTraining/ghostnet_no_channel_multiplier/model_best.pt'
    #checkpoint_path = '/home/taekwang0094/WorkSpace/FireTraining/ghostnet_1.2/model_epoch98_0.0225.pt'

    checkpoint = torch.load(checkpoint_path)
    if args.channel_multiplier is False:
        channel_multiplier = [1,1,1]
    else:
        channel_multiplier = args.channel_multiplier
    if args.model == 'resnet18':
        pass
    elif args.model =='mobilenet_v2':
        model = mobilenet_v2(num_classes=2)
    elif args.model =='shufflenet_v2_x1_0':
        model = shufflenet_v2_x1_0(num_classes=2)
    elif args.model =='mobilenetv3_small':
        model = mobilenetv3_small(num_classes = 2)
    elif args.model =='ghostnet':
        model = ghostnet(num_classes = 2)

    model = torch.nn.DataParallel(model)
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    """
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    """

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
    test_loader = torch.utils.data.DataLoader(
        dataset.FireDataset(args.root, transforms=data_transforms['val'], train='test', channel_multiplier=channel_multiplier),
        batch_size=int(args.batch_size),
        shuffle=True
    )

    val_total = 0
    val_correct = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    model.eval()
    total_inference_time = 0
    count = 0
    for batch_idx, (image, label) in enumerate(test_loader):
        count +=1
        image = image.to(device)
        label = label.to(device)
        # image, label = Variable(image).to(device), Variable(label).to(device)
        with torch.no_grad():
            time_1 = timer()
            output = model(image).to(device)
            time_2 = timer()
            inference_time = time_2-time_1
            total_inference_time += inference_time
            _, predicted = torch.max(output.data, 1)
            #print("Predicted",batch_idx, predicted.item(),label.item() )
            val_label_eval = torch.argmax(label,1)
            val_total += label.size(0)
            val_correct += (predicted == val_label_eval).sum().item()
            #print("pred : " ,predicted.item())
            #print("label : ", val_label_eval.item())

            # Precision, Recall, F1 Score
            if val_label_eval.item() == 1:
                if predicted.item() == 1:
                    TP +=1
                else :
                    FN +=1
            else:
                if predicted.item() == 1:
                    FP ++1
                else:
                    TN +=1



        print('{0}% 완료,\r'.format(int(batch_idx / len(test_loader) * 100)), end="")
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    recall = TP/(TP + FN)
    precision = TP/(TP+FP)
    f1_score = 2*(precision * recall) / (precision+recall)
    print("Model : ", args.model , "Channel Multiplier : [{0},{1},{2}]".format(channel_multiplier[0],channel_multiplier[1],channel_multiplier[2]))
    print("Average inference time : ", round(total_inference_time/count,2), " Average FPS : ", round(count/total_inference_time,2))
    print("Accuracy = ",round(accuracy,4), "Precision = ",round(precision,4), "recall = ", round(recall,4),"f1 score = ",round(f1_score,4))
    #print(val_correct/val_total)


if __name__ =="__main__":
    main()



