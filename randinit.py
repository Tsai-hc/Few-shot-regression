# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:22:11 2023

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
import torch.nn.functional as F
# For Progress Bar
from tqdm import tqdm

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
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions. 
        self.layers = nn.Sequential(
            #nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 112),
            nn.ReLU(),
            nn.Linear(112, 60),
            nn.ReLU(),
            nn.Linear(60, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
            

        )

    def forward(self, x):
        x = self.layers(x)
        #x = x.squeeze(1) # (B, 1) -> (B)
        return x
from scipy.ndimage import convolve1d    
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none') # gmean
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss
def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = F.l1_loss(inputs, targets, reduction='none') # gmean
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def trainer(train_loader, model, config, device):
    #pos_weight = torch.ones([5])  # All weights are equal to 1
    #criterion = nn.L1Loss(reduction='mean') # L1   MSE
    R = 10
    
    if R == 1:
        print("focal R")
        criterion = weighted_focal_l1_loss
    else:
        print("L1 loss")
        criterion = weighted_l1_loss

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

        if mean_train_loss < best_loss:
            best_loss = mean_train_loss
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
    'seed': 700,      # Your seed number, you can pick your lucky number. :)
    'select_all': True,   # Whether to use all features.
    'valid_ratio': 0,   # validation_size = train_size * valid_ratio
    'n_epochs': 10,     # Number of epochs.            
    'batch_size': 4, 
    'learning_rate': 0.0003,              
    'early_stop': 200,    # If model has not improved for this many consecutive epochs, stop training.     
    'save_path': './models/random.ckpt',  # Your model will be saved here.
    'k_shot': 20,
}
# Set seed for reproducibility
#same_seed(config['seed'])
te1 = pd.read_csv("tB43.csv")
te2 = pd.read_csv("tB42.csv")
te3 = pd.read_csv("tB41.csv")
te4 = pd.read_csv("tB40.csv")
te5 = pd.read_csv("tB39.csv")
te6 = pd.read_csv("tB38.csv")
te7 = pd.read_csv("tB37.csv")
te8 = pd.read_csv("tB36.csv")
te9 = pd.read_csv("tB35.csv")
te10 = pd.read_csv("tB34.csv")
te11 = pd.read_csv("tB33.csv")
te12 = pd.read_csv("tB32.csv")
te13 = pd.read_csv("tB31.csv")
te14 = pd.read_csv("tB30.csv")
te15 = pd.read_csv("tB29.csv")
te16 = pd.read_csv("tB28.csv")
te17 = pd.read_csv("tB27.csv")
te18 = pd.read_csv("tB26.csv")
te19 = pd.read_csv("tB25.csv")
te20 = pd.read_csv("tB24.csv")
te21 = pd.read_csv("tB23.csv")
te22 = pd.read_csv("tB22.csv")
te23 = pd.read_csv("tB21.csv")
te24 = pd.read_csv("tB20.csv")
te25 = pd.read_csv("tB19.csv")
te26 = pd.read_csv("tB18.csv")
te27 = pd.read_csv("tB17.csv")
te28 = pd.read_csv("tB16.csv")
te29 = pd.read_csv("tB15.csv")
te30 = pd.read_csv("tB14.csv")
te31 = pd.read_csv("tB13.csv")
te32 = pd.read_csv("tB12.csv")
te33 = pd.read_csv("tB11.csv")
te34 = pd.read_csv("tB10.csv")
te35 = pd.read_csv("tB9.csv")
te36 = pd.read_csv("tB8.csv")
te37 = pd.read_csv("tB7.csv")
te38 = pd.read_csv("tB6.csv")
te39 = pd.read_csv("tB5.csv")
te40 = pd.read_csv("tB4.csv")
te41 = pd.read_csv("tB3.csv")
te42 = pd.read_csv("tB2.csv")
te43 = pd.read_csv("tB1.csv")
alltest = np.concatenate([te1,te2,te3,te4,te5,te6,te7,te8,te9,te10,te11,te12,te13,te14,te15,te16,
                          te17,te18,te19,te20,te21,te22,te23,te24,te25,te26,te27,te28,te29,te30,
                          te31,te32,te33,te34,te35,te36,te37,te38,te39,te40,te41,te42,te43])                                              
test = alltest.astype('float32')
label_train = pd.read_csv("rlabel_test.csv")
t = np.hstack([test,label_train])

np.random.shuffle(t)
t1 = t[0:config['k_shot']]

#train_data_label, valid_data_label = train_valid_split(t1, config['valid_ratio'], config['seed'])

x_train , y_train= t1[:,0:243] , t1[:,243:244]
train_dataset = Dataset(x_train, y_train)
                                
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

model = My_Model(input_dim=x_train.shape[1]).to(device)

#model = Net().to(device) 

#for parm in model.parameters():
   #print(parm)
#for param in model.parameters():
 #   param.requires_grad = True
trainer(train_loader, model, config, device)
model.load_state_dict(torch.load(config['save_path']))



#trainer(train_loader, valid_loader, model, config, device)
#model.load_state_dict(torch.load(config['save_path']))


testtrain_dataset = Dataset(test)
testtrain_loader = DataLoader(testtrain_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
preds = predict(testtrain_loader, model, device)

from sklearn.metrics import mean_squared_error  ,mean_absolute_error #mean_squared_error   mean_absolute_error 
errorMSE = mean_squared_error(preds, label_train)
print('\nMSE: %.3f' % errorMSE)
errorMAE = mean_absolute_error(preds, label_train)
print('\nMAE: %.3f' % errorMAE)


#for parm in model.parameters():
 #   print(parm)
from matplotlib.pyplot import imshow
import numpy as np
import matplotlib.pyplot as plt # plt 用於顯示圖片
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import os
import pandas as pd
from numpy import resize

from typing import Iterable 
def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x
            
pred = list(flatten(preds))
true = list(flatten(label_train))

def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x)+1) / len(x)
    return(x,y)
def plot_ecdf(data, xlabel = None, ylabel = 'ECDF', label = None):
    x, y = ecdf(data)
    plt.plot(x, y, marker='.', markersize = 3, linestyle = 'none', label = label)
    plt.legend(markerscale = 4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    


plt.plot(label_train, label = 'true')

plt.plot(preds, 'r', label = 'pred')
plt.show()