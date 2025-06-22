import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import torch.nn.functional as F
import torch.nn.init as init
import os
# import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib.gridspec as gridspec
import pickle
import matplotlib.cbook as cbook
import random
import sys
from os.path import dirname, join as pjoin
import scipy.io as sio
# import cvxpy as cp
import numpy as np
from pprint import pprint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import copy
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.networks import *
from utils.trainer import *
from utils.formulations import *

datafolderpath = '../input_data'
saveresultfolderpath = '../experimental_result'
hyperparameterfolderpath = '../hyperparameter'

dict_path = os.path.join(datafolderpath, 'dgp_sin_100trials.pkl')
hyperparam_path = os.path.join(hyperparameterfolderpath, 'hyperparam_cwcli_dgpsin.json')

## Load dataset ##
with open(dict_path, 'rb') as pickle_file:
    data = pickle.load(pickle_file)
x = data['x']
y_all_trials = data['y_all_trials']
y_true = data['y_true']
N_trials = y_all_trials.shape[1]

## Load hyperparameters ##
with open(hyperparam_path, 'r') as file:
    hyperparams = json.load(file)

########### To collect the outputs ###########

# Define to find shape of collecting array
train = trainer(num_epochs = 3000, batch_size = 20000, patience = 50)
Xinput = torch.tensor(x, dtype = torch.float)
yinput = torch.tensor(y_all_trials[:,0].ravel(), dtype = torch.float)
xtrain, ytrain, xval, yval = train.train_test_split(Xinput, yinput, val_ratio = 0.2)

# To collect the outputs
gamma_list = np.linspace(hyperparams['gamma_min'], hyperparams['gamma_max'], 10) 
outputs_val_all = np.zeros((yval.shape[0], 2, len(gamma_list), N_trials)) # no.samples x 2 (LB x UB) x no.method x no.gamma x no.of trials
outputs_train_all = np.zeros((ytrain.shape[0], 2, len(gamma_list), N_trials)) # no.samples x 2 (LB x UB) x no.method x no.gamma x no.of trials
PIwidth = np.zeros((yval.shape[0], len(gamma_list), N_trials)) # no.samples x no.gamma x no.of trials

PICP = np.zeros((len(gamma_list), N_trials)) # no.gamma x no.of trials
PINAW = np.zeros((len(gamma_list), N_trials)) # no.gamma x no.of trials
PINALW = np.zeros((len(gamma_list), N_trials)) # no.gamma x no.of trials
Winkler = np.zeros((len(gamma_list), N_trials)) # no.gamma x no.of trials

allytrain = np.zeros((len(ytrain), N_trials))
allxtrain = np.zeros((xtrain.shape[0], xtrain.shape[1], N_trials))
allyval = np.zeros((len(yval), N_trials))
allxval = np.zeros((xval.shape[0], xval.shape[1], N_trials))

##########################################

########### Setting parameters ###########
X_input = torch.tensor(x, dtype = torch.float)
train = trainer(num_epochs = 3000, batch_size = 500, patience = 500) #Set the trainer
torch.manual_seed(21) #Must have to initialize the network parameters
model = CustomNet(input_size = X_input.shape[1], hidden_size = 100, output_size = 2)

############ Sumklargestlowest ############
delta = 0.1
for j in range(N_trials):
    print(f'---------- Data index: {j} ----------')
    y_input = torch.tensor(y_all_trials[:,j].ravel(), dtype = torch.float) 
    for i, gamma in enumerate(gamma_list):
        print(f'Data index {j}, For gamma {gamma}')
        torch.manual_seed(21)
        model = CustomNet(input_size = X_input.shape[1], hidden_size = 100, output_size = 2)
        optimizer = torch.optim.Adam(model.parameters(), lr = hyperparams['lr'])
        criterion = cwcli_objective(delta_ = delta, gamma_ = gamma, alpha_ = hyperparams['alpha'], 
                                        beta_ = hyperparams['beta'], soften_ = 50, smoothfunction = 'tanh'
                                        , returnseparatedloss = True)

        train = trainer(num_epochs = 2000, batch_size = hyperparams['batch_size'], patience = hyperparams['patience']) 

        X_train, y_train, X_val, y_val = train.train_test_split(X_input, y_input, val_ratio = 0.2)

        train_loss_list, val_loss_list, model, coverageloss_train, widthloss_train = train.training(X_train, y_train, X_val, y_val, criterion, optimizer, model)

        #Evaluation in validation set
        X_val_sorted, y_val_sorted = train.sort_x_toplot(X_val, y_val)
        X_train_sorted, y_train_sorted = train.sort_x_toplot(X_train, y_train)
        
        allytrain[:,j] = y_train_sorted
        allxtrain[:,:,j] = X_train_sorted
        allyval[:,j] = y_val_sorted
        allxval[:,:,j] = X_val_sorted
        
        outputs_val = train.predict(X_val_sorted, model, ymean = torch.mean(y_train), ystd = torch.std(y_train))
        outputs_train = train.predict(X_train_sorted, model, ymean = torch.mean(y_train), ystd = torch.std(y_train))

        deviceback = torch.device('cpu')
        outputs_val = outputs_val.to(deviceback)
        outputs_train = outputs_train.to(deviceback)

        outputs_val_all[:,:,i,j] = outputs_val
        outputs_train_all[:,:,i,j] = outputs_train

        PICP[i,j] = train.PICP(y_val_sorted, outputs_val[:,1], outputs_val[:,0])
        PINAW[i,j] = train.PINAW(outputs_val[:,1], outputs_val[:,0], y_input)
        PINALW[i,j] = train.PINALW(outputs_val[:,1], outputs_val[:,0], y_input, quantile = 0.5)
        Winkler[i,j] = train.Winklerscore(outputs_val[:,1], outputs_val[:,0], y_val_sorted, y_input, delta = 0.1)
        
        width = outputs_val[:,1] - outputs_val[:,0]
        quantile_width_data = np.quantile(y_input, 0.95, axis = 0) - np.quantile(y_input, 0.05, axis = 0)
        norm_width = width/quantile_width_data
        PIwidth[:,i,j] = norm_width
        
        print(f'For gamma: {gamma}: PICP = {PICP[i,j]}, PINAW = {PINAW[i,j]}, PINALW = {PINALW[i,j]}, avgPIwidth = {np.mean(PIwidth[:,i,j])}')

        saved_result = {'outputs_train':outputs_train_all, 'outputs_val':outputs_val_all
                        , 'PICP_val':PICP, 'PINAW': PINAW, 'PINALW':PINALW, 'Winkler':Winkler, 'PIwidth':PIwidth
                        , 'allytrain':allytrain, 'allxtrain': allxtrain, 'allyval': allyval, 'allxval':allxval
                        , 'gamma': gamma_list, 'alpha': hyperparams['alpha'], 'beta': hyperparams['beta']}

        filename = f'cwcli_tradeoff_dgpsin.pkl'
        # # Save
        dict_path = os.path.join(saveresultfolderpath, filename)
        with open(dict_path, 'wb') as pickle_file:
            pickle.dump(saved_result, pickle_file)
    
##########################################
print('---------- Finished ----------')
filenamedone = f'cwcli_tradeoff_dgpsin_done.pkl'
##########################################
# # Save
dict_path = os.path.join(saveresultfolderpath, filenamedone)
with open(dict_path, 'wb') as pickle_file:
    pickle.dump(saved_result, pickle_file)
    