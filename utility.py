import os
import glob
import random
import torch
from sklearn.model_selection import train_test_split

def train_val_split(mode= 'train'): # mode : 'train' or 'test
    file_list = '/disk2/taekwang/fire_dataset/{}_list.txt'.format(mode)
    save_path = '/home/taekwang0094/WorkSpace/Summer_Conference'
    f = open(file_list, 'r')
    fire = 0
    nofire = 0
    total_count = 0

    nofire_list=[]
    fire_list = []
    while True:

        line = f.readline()
        if not line: break
        total_count += 1
        a = line.split()
        if len(a) == 1:
            nofire += 1
            nofire_list.append(line)
        elif len(a) >= 2:
            fire += 1
            fire_list.append(line)

    train_fire, val_fire = train_test_split(fire_list, test_size= 0.1, random_state=123)
    train_nofire, val_nofire = train_test_split(nofire_list, test_size=0.1, random_state=123)
    train_list = []
    val_list = []
    train_list.extend(train_fire)
    train_list.extend(train_nofire)
    val_list.extend(val_fire)
    val_list.extend(val_nofire)

    random.shuffle(train_list)
    random.shuffle(val_list)
    """
    f = open(os.path.join(save_path,'train_list.txt'),'w')
    for word in train_list:
        f.write(word)
    f.close()
    f = open(os.path.join(save_path,'val_list.txt'),'w')
    for word in val_list:
        f.write(word)
    f.close()
    """

    print(train_list)

    print(len(train_list))

    print(len(train_fire), len(val_fire))
    print(len(train_nofire), len(val_nofire))

    print(total_count, fire, nofire)

train_val_split()

def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1