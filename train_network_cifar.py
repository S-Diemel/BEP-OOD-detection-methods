from __future__ import print_function
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
import sys, os
from utils import *


classes = ('plane', 'car' , 'bird','cat', 'deer', 'dog','frog', 'horse', 'ship', 'truck')
num_classes=10
batch_size = 128


mean_id = (0.4914, 0.4822, 0.4465)  # mean along channels
std_id = (0.2023, 0.1994, 0.2010)  # std along channels
print((mean_id, std_id))

# Define normalization transformation
normalize_transform_id = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean_id, std_id)])

print(os.getcwd())
train_dataset = datasets.CIFAR10(
    root= './data', train = True,
    download =True, transform = normalize_transform_id)
test_dataset = datasets.CIFAR10(
    root= './data', train = False,
    download =True, transform = normalize_transform_id)



train_loader = torch.utils.data.DataLoader(train_dataset
                                           , batch_size = batch_size
                                           , shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset
                                          , batch_size = batch_size
                                          , shuffle = True)
n_total_step = len(train_loader)
print(n_total_step)


classifier = nn.Linear(512, num_classes)
net = resnet9(classifier, mod=True)
net = net.to(device)
print(net)
criterion = CE_Loss(num_classes, device)


adam = optim.Adam([{'params': net.parameters()},], lr=0.01, weight_decay=0.0001, amsgrad=True)
optimizer = Optimizer(adam, train_loader, device)

epochs=25
lr=0.008

sched = torch.optim.lr_scheduler.OneCycleLR(optimizer.optimizer, lr, epochs=epochs,
                                            steps_per_epoch=len(train_loader))


for epoch in range(epochs):
    print(f'Epoch: {epoch + 1}')
    train_loss, correct, conf = 0, 0, 0
    start_time= time.time()
    net.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        optimizer.optimizer.zero_grad()
        loss, Y_pred = criterion.loss(inputs,targets, net)
        loss.backward()
        optimizer.optimizer.step()
        inputs.requires_grad_(False)
        sched.step()
        with torch.no_grad():
            criterion.prox(net)
            train_loss += loss.item()
            confBatch, predicted = Y_pred.max(1)
            correct += predicted.eq(targets).sum().item()
            conf+=confBatch.sum().item()
    execution_time = (time.time() - start_time)
    print('Loss: %.3f | Acc: %.3f%% (%d/%d) | Conf %.2f | time (s): %.2f'% (train_loss/len(train_loader), 100.*correct/len(train_loader.dataset), correct, len(train_loader.dataset), 100*conf/len(train_loader.dataset), execution_time))
    (acc,conf) = optimizer.test_acc(net,criterion, test_loader)
    print(optimizer.optimizer.param_groups[0]['lr'])


print('Saving..')
name='networkcifar'
state = {'net': net.state_dict(),'acc': acc}
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
torch.save(state, './checkpoint/%s.t7'%(name))