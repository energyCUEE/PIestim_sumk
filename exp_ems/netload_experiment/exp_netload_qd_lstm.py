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

# Define
gamma_list = [0.011]
delta = 0.1

for i, gamma in enumerate(gamma_list):
    print(f'---------- gamma = {gamma} ----------')
    torch.manual_seed(290)
    model = SolarkstepaheadNet_LSTM_exoinput(lag_input_window_size = len(features_list) - len(target_col) - len(future_regressor)
                           , exo_input_window_size = len(future_regressor), num_lag_features = 4
                                             , hidden_size = 30, predicted_step = len(target_col)
                                            , num_layers = 2, lstm_hidden_size = 45)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)
    criterion = qd_objective(delta_ = delta, gamma_ = gamma, soften_ = 100, smoothfunction = 'sigmoid')

    train = trainer_multistep(num_epochs = 2000, batch_size = int(0.3*X_train.shape[0]), patience = 100)
    train_loss_list, val_loss_list, model = train.training(X_train, y_train, X_val, y_val, criterion, optimizer, model)

    outputs_train_qd = train.predict(X_train, model, ymean = torch.mean(y_train), ystd = torch.std(y_train))
    outputs_val_qd = train.predict(X_val, model, ymean = torch.mean(y_train), ystd = torch.std(y_train))
    outputs_test_qd = train.predict(X_test, model, ymean = torch.mean(y_train), ystd = torch.std(y_train))

# Evaluate the result in validation and test set
    outputs_val_eval = outputs_val_qd
    outputs_test_eval = outputs_test_qd

    outputs_all = train.predict(X_all, model, ymean = torch.mean(y_train), ystd = torch.std(y_train))
    PICP_all = train.PICP(y_all, outputs_all[:,1::2], outputs_all[:,0::2])
    PINALW_all = train.PINALW(outputs_all[:,1::2], outputs_all[:,0::2], y_train)
    print('PICP of entire dataset')
    print(PICP_all)
    print('PINALW of entire dataset')
    print(PINALW_all)

    gamma_str = f"{gamma}".replace('.', '')           
    saved_result = {'outputs_train':outputs_train_qd, 'outputs_val':outputs_val_qd, 'outputs_all': outputs_all
                    , 'outputs_test':outputs_test_qd,'gamma': gamma_list}

    filename = f'qd_gam{gamma_str}_netload_16step.pkl'
    print(f'Save as: {filename}')
# Save
    dict_path = os.path.join(saveresultfolderpath, filename)
    with open(dict_path, 'wb') as pickle_file:
        pickle.dump(saved_result, pickle_file)



