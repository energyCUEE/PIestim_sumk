import numpy as np
import pandas as pd
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import torch.nn.functional as F
import torch.nn.init as init

import pickle
import sys
from os.path import dirname, join as pjoin

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.networks import *
from utils.trainer import *
from utils.formulations import *

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

datafolderpath = '../data'
saveresultfolderpath = './result'

# Import data
dict_path = os.path.join(datafolderpath, 'data_netload_nonan.pkl')
with open(dict_path, 'rb') as pickle_file:
    data = pickle.load(pickle_file)
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']
X_all = data['X_all']
y_all = data['y_all']

target_col = data['target_col']
features_list = data['features_list']
future_regressor = data['future_regressor']

## Define hyperparameter
gamma_list = [0.2]
delta = 0.1
k_sumk = 0.3
lambda_sumk = 0.5

for i, gamma in enumerate(gamma_list):
    torch.manual_seed(21)
    model = SolarkstepaheadNet_LSTM_exoinput(lag_input_window_size = len(features_list) - len(target_col) - len(future_regressor)
                           , exo_input_window_size = len(future_regressor), num_lag_features = 4
                                             , hidden_size = 30, predicted_step = len(target_col)
                                            , num_layers = 2, lstm_hidden_size = 45)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    criterion = sumk_objective(delta_ = delta, gamma_ = gamma, percentlargest_ = k_sumk, lambda_ = lambda_sumk
                                              , soften_ = 50, smoothfunction = 'tanh')

    train = trainer_multistep(num_epochs = 2000, batch_size = int(0.3*X_train.shape[0]), patience = 100)
    train_loss_list, val_loss_list, model = train.training(X_train, y_train, X_val, y_val, criterion, optimizer, model)

    outputs_train_sumk_lstm = train.predict(X_train, model, ymean = torch.mean(y_train), ystd = torch.std(y_train))
    outputs_val_sumk_lstm = train.predict(X_val, model, ymean = torch.mean(y_train), ystd = torch.std(y_train))
    outputs_test_sumk_lstm = train.predict(X_test, model, ymean = torch.mean(y_train), ystd = torch.std(y_train))
    
    deviceback = torch.device('cpu')
    outputs_train_sumk_lstm = outputs_train_sumk_lstm.to(deviceback)
    outputs_val_sumk_lstm = outputs_val_sumk_lstm.to(deviceback)
    outputs_test_sumk_lstm = outputs_test_sumk_lstm.to(deviceback)
        
    outputs_val_eval = outputs_val_sumk_lstm
    outputs_test_eval = outputs_test_sumk_lstm
        
    outputs_all = train.predict(X_all, model, ymean = torch.mean(y_train), ystd = torch.std(y_train))
    PICP_all = train.PICP(y_all, outputs_all[:,1::2], outputs_all[:,0::2])
    PINALW_all = train.PINALW(outputs_all[:,1::2], outputs_all[:,0::2], y_train)
    print('PICP of entire dataset')
    print(PICP_all)
    print('PINALW of entire dataset')
    print(PINALW_all)
    
    min_val_loss = min(val_loss_list)
    min_train_loss = train_loss_list[np.argmin(val_loss_list)]

saved_result = {'outputs_train':outputs_train_sumk_lstm, 'outputs_val':outputs_val_sumk_lstm, 'outputs_all': outputs_all
                , 'outputs_test':outputs_test_sumk_lstm, 'gamma':gamma_list}

gamma_str = f"{gamma}".replace('.', '')           
lambda_str = f"{lambda_sumk}".replace('.', '')    
k_str = f"{k_sumk}".replace('.', '')

filename = f'sumk_gam{gamma_str}_netload_16step_k{k_str}_lam{lambda_str}.pkl'
print(f'Save as: {filename}')
# Save
dict_path = os.path.join(saveresultfolderpath, filename)
with open(dict_path, 'wb') as pickle_file:
    pickle.dump(saved_result, pickle_file)
    
    
    