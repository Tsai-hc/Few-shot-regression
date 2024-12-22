# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 00:27:56 2023

@author: user
"""
from tensorflow import keras
# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv

# For Progress Bar
from tqdm import tqdm
import matplotlib.pyplot as plt
# Pytorch
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)
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
class Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from collections import OrderedDict    
def truncated_normal(*size, mean=0, std=1):
    size = list(size)
    tensor = torch.empty(size)
    tmp = torch.empty(size+[4,]).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor    
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

def trainer(train_loader, valid_loader, model, config, device):
    #pos_weight = torch.ones([5])  # All weights are equal to 1
    criterion = nn.L1Loss(reduction='mean') # L1   MSE
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    #writer = SummaryWriter() # Writer of tensoboard.

    if not os.path.isdir('./models'):
        os.mkdir('./models') # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train() # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            x, y = x.to(device), y.to(device)   # Move your data to device. 
            pred = model(x)             
            loss = criterion(pred, y)
            loss.backward()                     # Compute gradient(backpropagation).
            optimizer.step()                    # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())
            
            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
       # writer.add_scalar('Loss/train', mean_train_loss, step)
        #plt.plot(loss_record) 

        model.eval() # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())
            
        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        #writer.add_scalar('Loss/valid', mean_valid_loss, step)
        

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 70,      # Your seed number, you can pick your lucky number. :)
    'select_all': True,   # Whether to use all features.
    'valid_ratio': 0.1,   # validation_size = train_size * valid_ratio
    'n_epochs': 5000,     # Number of epochs.            
    'batch_size': 200, 
    'learning_rate': 0.0003,              
    'early_stop': 200,    # If model has not improved for this many consecutive epochs, stop training.     
    'save_path': '999.pth'  # Your model will be saved here.
}
# Set seed for reproducibility
same_seed(config['seed'])

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

t = np.concatenate([t1,t2,t3])

train_data_label, valid_data_label = train_valid_split(t, config['valid_ratio'], config['seed'])

x_train , y_train= train_data_label[:,0:243] , train_data_label[:,243:244]
x_valid , y_valid= valid_data_label[:,0:243] , valid_data_label[:,243:244]

train_dataset, valid_dataset = Dataset(x_train, y_train), \
                                Dataset(x_valid, y_valid)
                                
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

model = Net().to(device) # put your model and data on the same computation device.
trainer(train_loader, valid_loader, model, config, device)
model.load_state_dict(torch.load(config['save_path']))

 
test = t[:,0:243]
testtrain_dataset = Dataset(test)
testtrain_loader = DataLoader(testtrain_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
preds = predict(testtrain_loader, model, device)



from sklearn.metrics import mean_squared_error  ,mean_absolute_error #mean_squared_error   mean_absolute_error 
label_test = np.concatenate([label_train,label_train,label_train])
errorMSE = mean_squared_error(preds, label_test)
print('\nMSE: %.3f' % errorMSE)
errorMAE = mean_absolute_error(preds, label_test)
print('\nMAE: %.3f' % errorMAE)

