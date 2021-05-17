import os,glob
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
def file_count(mode= 'train'): # mode : 'train' or 'test
    file_list = '/home/taekwang0094/WorkSpace/Summer_Conference/{}_list.txt'.format(mode)

    f = open(file_list, 'r')
    fire = 0
    nofire = 0
    total_count = 0
    while True:

        line = f.readline()
        if not line: break
        total_count += 1
        a = line.split()
        if len(a) == 1:
            nofire += 1
        elif len(a) >= 2:
            fire += 1

    print(total_count, fire, nofire)

file_count('test')

class FireDataset(Dataset):
    def __init__(self, root,  transforms = None, train = 'train', normalize = False, channel_multiplier = [2,1,1], ch_preprocess = False):
        if train== 'train':
            self.txt_file = os.path.join(root,'train_list.txt')
        elif train == 'test':
            self.txt_file = os.path.join(root,'test_list.txt')
        elif train == 'val':
            self.txt_file = os.path.join(root, 'val_list.txt')

        self.channel_multiplier = channel_multiplier
        self.normalize = normalize
        with open(self.txt_file,'r') as file:
            self.lines = file.readlines()
        self.image_list = []
        self.label_list = []
        self.ch_preprocess = ch_preprocess
        for word in self.lines:
            image_path = word.split()[0]
            if len(word.split()) >=2:
                self.label_list.append([1,0])
            else:
                self.label_list.append([0,1])
            self.image_list.append(image_path)

        print("image number : ", len(self.image_list), "label number : ", len(self.label_list))
        assert len(self.image_list) == len(self.label_list)

        #self.shuffle = shuffle
        self.transforms = transforms
        #self.batch_size = batch_size

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx]).convert('RGB')
        #print("open",image)
        #image = cv2.imread(self.image_list[idx], cv2.IMREAD_COLOR)
        #cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = image.astype(np.float32)
        image = np.array(image).astype(np.float32)
        #print("AAAAAAAAAAA",image.sum())
        if self.normalize:
            image /=255.
        #print(image)

        if not self.ch_preprocess:
            image[:,:,0] *= self.channel_multiplier[0]
            image[:,:,1] *= self.channel_multiplier[1]
            image[:,:,2] *= self.channel_multiplier[2]
        #print(image.sum())
        image = image.astype(np.uint8)
        #image = Image.fromarray(image)

        #print(np.array(image).shape)
        #print(np.array(image)[:,:,1]*1)
        if self.transforms is not None:
            image = self.transforms(image)
        #print(image.sum())
        #print(image.shape)
        if self.ch_preprocess:
            image[0, :, :] *= self.channel_multiplier[0]
            image[1, :, :] *= self.channel_multiplier[1]
            image[2, :, :] *= self.channel_multiplier[2]
        #print(image.sum())
        #print(image.size)
        label = torch.Tensor(self.label_list[idx])

        return (image, label)
"""
data_transforms = {
        'train': transforms.Compose([
            #transforms.RandomResizedCrop(224),
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
"""
#FireDataset('/home/taekwang0094/WorkSpace/Summer_Conference', transforms=data_transforms['train'], ch_preprocess=True).__getitem__(1)