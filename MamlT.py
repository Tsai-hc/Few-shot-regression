# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:13:30 2023

@author: user
"""
import numpy as np
# Numerical Operations
import math
import numpy as np
# Reading/Writing Data
import pandas as pd
import os
import csv
# For Progress Bar
from tqdm import tqdm
# Pytorch
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds
class Datasets(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None):
        self.k_shot = 20
        if y is None:
            self.y = y
            
        else:
            self.data = np.hstack([x,y])
            self.a = np.split(self.data,config['e'])
            self.a = np.array(self.a)
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:

            inputs = self.a[:,0:243][idx]
            targets = self.a[:,243:244][idx]
            
            
            inputs = torch.tensor(inputs)
            targets = torch.tensor(targets)
            #return inputs[:self.k_shot], targets[:self.k_shot], inputs[self.k_shot:], targets[self.k_shot:]
            return inputs[:self.k_shot], inputs[self.k_shot:]
        #return self.x[idx], self.y[idx]
            #return self.x[idx], self.y[idx]

    def __len__(self):
        return config['e']
        #return len(self.x)  
    
def truncated_normal(*size, mean=0, std=1):
    size = list(size)
    tensor = torch.empty(size)
    tmp = torch.empty(size+[4,]).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch
#from utils import truncated_normal
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        super(Net, self).__init__()
        self.w1 = Parameter(truncated_normal(243, 112, std=1e-2), requires_grad=True)
        self.w2 = Parameter(truncated_normal(112, 60, std=1e-2), requires_grad=True)
        self.w3 = Parameter(truncated_normal(60, 30, std=1e-2), requires_grad=True)
        self.w4 = Parameter(truncated_normal(30, 1, std=1e-2), requires_grad=True)

        self.b1 = Parameter(torch.zeros(112), requires_grad=True)
        self.b2 = Parameter(torch.zeros(60), requires_grad=True)
        self.b3 = Parameter(torch.zeros(30), requires_grad=True)
        self.b4 = Parameter(torch.zeros(1), requires_grad=True)

        self.params = OrderedDict([('w1', self.w1), ('w2', self.w2), ('w3', self.w3), ('w4', self.w4),
                                   ('b1', self.b1), ('b2', self.b2), ('b3', self.b3), ('b4', self.b4)])

    def forward(self, x, weights=None):
        if weights is None:
            weights = OrderedDict([(name, p) for name, p in self.named_parameters()])

        x = F.relu(x.matmul(weights['w1']) + weights['b1'], inplace=True)
        x = F.relu(x.matmul(weights['w2']) + weights['b2'], inplace=True)
        x = F.relu(x.matmul(weights['w3']) + weights['b3'], inplace=True)
        x = x.matmul(weights['w4']) + weights['b4']

        return x

import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.nn.parameter import Parameter
from collections import OrderedDict
#from utils import *
import random
#from net import Net

class MAML(nn.Module):

    def __init__(self, inner_lr, outer_lr):
        super(MAML, self).__init__()
        self.net = Net()
        self.inner_lr = inner_lr
        self.optimizer = optim.Adam(list(self.net.parameters()), lr=outer_lr)
        #load_state = torch.load('E:\碩論\BSonly\mamltrain\\333.pth', map_location='cpu')
        #self.load_state_dict(load_state['model_state_dict'])

    def forward(self, k_x, k_y, q_x, q_y):
        #load_state = torch.load('332.pth', map_location='cpu')
        #self.load_state_dict(load_state['model_state_dict'])

        task_num = k_x.size(0) #batch_size
        losses = 0

        #print(task_num)
        #j = 0
        for i in range(task_num):
            #load_state = torch.load('335.ckpt', map_location='cpu')
            #self.load_state_dict(load_state['model_state_dict'])
            pred_k = self.net(k_x[i])
            loss_k = F.l1_loss(pred_k, k_y[i])  # huber_loss  mse  smooth_l1_loss
            #j+=1
            #Replaces pow(2.0) with abs() for L1 regularization
            
 
            loss = loss_k 
            
            grad = torch.autograd.grad(loss, list(self.net.parameters())) #,create_graph = True
            
            fast_weights = OrderedDict(
                [(name, p - self.inner_lr * g) for (name, p), g in zip(self.net.named_parameters(), grad)]
            )

            pred_q = self.net(q_x[i], fast_weights)
            loss_q = F.l1_loss(pred_q, q_y[i])

            losses = losses + loss_q

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        #print(j)
        

        return losses.item() / task_num

    def test(self, k_x, k_y, q_x, q_y):
        load_state = torch.load('332.pth', map_location='cpu')
        self.load_state_dict(load_state['model_state_dict'])


        batch_size = k_x.size(0)
        losses = 0

        for i in range(batch_size):
            pred_k = self.net(k_x[i])
            loss_k = F.l1_loss(pred_k, k_y[i])
            grad = torch.autograd.grad(loss_k, self.net.parameters())
            fast_weights = OrderedDict(
                [(name, p - self.inner_lr * g) for (name, p), g in zip(self.net.named_parameters(), grad)]
            )

            pred_q = self.net(q_x[i], fast_weights)
            loss_q = F.l1_loss(pred_q, q_y[i])
            losses = losses + loss_q

        losses /= batch_size

        return losses.item()   
    
config = {
    'seed': 7,      # Your seed number, you can pick your lucky number. :)
    'select_all': True,   # Whether to use all features.
    'epoch': 25800,   
    'e': 25800,            
    'batch_size': 1,        
    'k_shot': 20,
    'query' :30,
    'save_path': '1000.pth'
    
}
train = pd.read_csv("Ba.csv")
train = train.astype('float32')
label_train = pd.read_csv("maml_Ba.csv")
t1 = np.hstack([train,label_train])

train = pd.read_csv("Bb.csv")
train = train.astype('float32')
t2 = np.hstack([train,label_train])

train = pd.read_csv("Bc.csv")
train = train.astype('float32')
t3 = np.hstack([train,label_train])

tw = np.concatenate([t1,t2,t3])
t = np.repeat(tw,100,axis = 0)
np.random.shuffle(t)
#t = t[0:4300]

x_train , y_train= t[:,0:243] , t[:,243:244]
train_dataset= Datasets(x_train, y_train)                             
                               
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
#valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)


#from meta import MetaSGD, MAML
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
print(device)

meta = MAML(inner_lr=1e-3, outer_lr= 3e-4)
meta.to(device)

#train_ds = Sinusoid(k_shot= 5, q_query=10, num_tasks=1)
#train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
train_iter = iter(train_loader)

losses = []
epochs = config['epoch']
batch_size = config['batch_size']
pbar = tqdm(range(epochs))

for epoch in pbar:
    a, b= next(train_iter)
    k_i, k_o, q_i, q_o = a[:,:,0:243],a[:,:,243:244], b[:,:,0:243],b[:,:,243:244]
    k_i, k_o, q_i, q_o = k_i.float().to(device), k_o.float().to(device), q_i.float().to(device), q_o.float().to(device)
    loss = meta(k_i, k_o, q_i, q_o)
    pbar.set_description(f'{epoch}/{60000}iter | L:{loss:.4f}')

    if epoch % 1 == 0:
       losses.append(loss)
torch.save(meta.state_dict(), config['save_path'])


#meta.load_state_dict(torch.load(config['save_path']))

plt.plot(losses)

k_i = k_i.cpu().detach().numpy()
q_i = q_i.cpu().detach().numpy()
k_o = k_o.cpu().detach().numpy()
q_o = q_o.cpu().detach().numpy()