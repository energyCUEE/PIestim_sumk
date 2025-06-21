import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import torch.nn.functional as F
import torch.nn.init as init
import os

import matplotlib.pyplot as plt
import scipy
import matplotlib.gridspec as gridspec
import pickle
import matplotlib.cbook as cbook
import random
import sys
from os.path import dirname, join as pjoin
import scipy.io as sio

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

dict_path = os.path.join(datafolderpath, 'dgp_sin_100trials.pkl')

## Load dataset ##
with open(dict_path, 'rb') as pickle_file:
    data = pickle.load(pickle_file)
x = data['x']
y_all_trials = data['y_all_trials']
y_true = data['y_true']
N_trials = y_all_trials.shape[1]

########### To collect the outputs ###########
# Define to find shape of collecting array
train = DeepEnsemble_trainer(num_epochs = 2000, batch_size = 500, patience = 200) #Set the trainer
Xinput = torch.tensor(x, dtype = torch.float)
yinput = torch.tensor(y_all_trials[:,0].ravel(), dtype = torch.float)
xtrain, ytrain, xval, yval = train.train_test_split(Xinput, yinput, val_ratio = 0.2)

num_ensembles = 5

# Define gamma_list and collect the outputs 
outputs_val_all = np.zeros((yval.shape[0], 2, N_trials)) # no.samples x 2 (LB x UB) x no.of trials
outputs_train_all = np.zeros((ytrain.shape[0], 2, N_trials)) # no.samples x 2 (LB x UB) x no.of trials
PIwidth = np.zeros((yval.shape[0], N_trials)) # no.samples x no.gamma x no.of trials

PICP = np.zeros((N_trials)) # no.of trials
PINAW = np.zeros((N_trials)) # no.of trials
PINALW = np.zeros((N_trials)) # no.of trials
Winkler = np.zeros((N_trials)) # no.of trials

allytrain = np.zeros((len(ytrain), N_trials))
allxtrain = np.zeros((xtrain.shape[0], xtrain.shape[1], N_trials))
allyval = np.zeros((len(yval), N_trials))
allxval = np.zeros((xval.shape[0], xval.shape[1], N_trials))

allmeanpred_train = np.zeros((num_ensembles, len(ytrain), N_trials))
allvarpred_train = np.zeros((num_ensembles, len(ytrain), N_trials))
epistemic_uncertainty_train = np.zeros((len(ytrain), N_trials))
aleatoric_uncertainty_train = np.zeros((len(ytrain), N_trials))
total_uncertainty_train = np.zeros((len(ytrain), N_trials))
predictive_mean_train = np.zeros((len(ytrain), N_trials))

allmeanpred_val = np.zeros((num_ensembles, len(yval), N_trials))
allvarpred_val = np.zeros((num_ensembles, len(yval), N_trials))
epistemic_uncertainty_val = np.zeros((len(yval), N_trials))
aleatoric_uncertainty_val = np.zeros((len(yval), N_trials))
total_uncertainty_val = np.zeros((len(yval), N_trials))
predictive_mean_val = np.zeros((len(yval), N_trials))

##########################################

########### Setting parameters ###########
X_input = torch.tensor(x, dtype = torch.float)
train = DeepEnsemble_trainer(num_epochs = 2000, batch_size = 500, patience = 100) #Set the trainer
# torch.manual_seed(21) #Must have to initialize the network parameters
# model = DeepEnsembleNet(input_size = X_input.shape[1], hidden_size = 100)

delta = 0.1
for j in range(N_trials):
    print(f'---------- Data index: {j} ----------')
    y_input = torch.tensor(y_all_trials[:,j].ravel(), dtype = torch.float) 
    X_train, y_train, X_val, y_val = train.train_test_split(X_input, y_input, val_ratio = 0.2) 
    print(f'Data index {j}')
    torch.manual_seed(21)
    ensemble_models = [DeepEnsembleNet(input_size = X_input.shape[1], hidden_size = 100) for _ in range(num_ensembles)]
    optimizers = [torch.optim.Adam(model.parameters(), lr = 0.001) for model in ensemble_models]
    criterion = MVE_negloglikeGaussian_objective()
    train = DeepEnsemble_trainer(num_epochs = 2000, batch_size = 500, patience = 200) #Set the trainer
    
    ###### ADD DATA NORMALIZATION #######
    # Normalize data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_norm = scaler_X.fit_transform(X_train)
    X_train_norm = torch.tensor(X_train_norm, dtype = torch.float32)
    X_val_norm = scaler_X.transform(X_val)
    X_val_norm = torch.tensor(X_val_norm, dtype = torch.float32)
    
    y_train_norm = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()
    y_train_norm = torch.tensor(y_train_norm, dtype = torch.float32)
    y_val_norm = scaler_y.transform(y_val.reshape(-1,1)).ravel()
    y_val_norm = torch.tensor(y_val_norm, dtype = torch.float32)

    train_loss_list, val_loss_list, ensemble_models = train.training(X_train_norm, y_train_norm, X_val_norm, y_val_norm, criterion, optimizers, ensemble_models, plotloss = False)

    # Sort X, y
    X_train_sorted_norm, y_train_sorted_norm = train.sort_x_toplot(X_train_norm, y_train_norm)
    X_val_sorted_norm, y_val_sorted_norm = train.sort_x_toplot(X_val_norm, y_val_norm)
    
    # Inference mode
    results_train_sorted_norm = train.predict(X_train_sorted_norm, ensemble_models, delta_ = 0.1)
    outputs_train_sorted_norm = results_train_sorted_norm['pi']
    results_val_sorted_norm = train.predict(X_val_sorted_norm, ensemble_models, delta_ = 0.1)
    outputs_val_sorted_norm = results_val_sorted_norm['pi']
    
    # Denormalize data and result
    X_train_sorted = scaler_X.inverse_transform(X_train_sorted_norm)
    X_val_sorted = scaler_X.inverse_transform(X_val_sorted_norm)
    y_train_sorted = scaler_y.inverse_transform(y_train_sorted_norm.reshape(-1, 1)).ravel()
    y_val_sorted = scaler_y.inverse_transform(y_val_sorted_norm.reshape(-1, 1)).ravel()

    outputs_train_sorted = scaler_y.inverse_transform(outputs_train_sorted_norm)
    outputs_val_sorted = scaler_y.inverse_transform(outputs_val_sorted_norm)
    
    allytrain[:,j] = y_train_sorted
    allxtrain[:,:,j] = X_train_sorted
    allyval[:,j] = y_val_sorted
    allxval[:,:,j] = X_val_sorted

    outputs_train_all[:,:,j] = outputs_train_sorted
    outputs_val_all[:,:,j] = outputs_val_sorted
    
    allmeanpred_train[:,:,j] = results_train_sorted_norm['all_mean_predictions']
    allvarpred_train[:,:,j] = results_train_sorted_norm['all_var_predictions']
    epistemic_uncertainty_train[:,j] = results_train_sorted_norm['epistemic_uncertainty']
    aleatoric_uncertainty_train[:,j] = results_train_sorted_norm['aleatoric_uncertainty']
    total_uncertainty_train[:,j] = results_train_sorted_norm['total_variance']
    predictive_mean_train[:,j] = results_train_sorted_norm['predictive_mean']

    allmeanpred_val[:,:,j] = results_val_sorted_norm['all_mean_predictions']
    allvarpred_val[:,:,j] = results_val_sorted_norm['all_var_predictions']
    epistemic_uncertainty_val[:,j] = results_val_sorted_norm['epistemic_uncertainty']
    aleatoric_uncertainty_val[:,j] = results_val_sorted_norm['aleatoric_uncertainty']
    total_uncertainty_val[:,j] = results_val_sorted_norm['total_variance']
    predictive_mean_val[:,j] = results_val_sorted_norm['predictive_mean']

    PICP[j] = train.PICP(y_val_sorted, outputs_val_sorted[:,1], outputs_val_sorted[:,0])
    PINAW[j] = train.PINAW(outputs_val_sorted[:,1], outputs_val_sorted[:,0], y_input)
    PINALW[j] = train.PINALW(outputs_val_sorted[:,1], outputs_val_sorted[:,0], y_input, quantile = 0.5)
    Winkler[j] = train.Winklerscore(outputs_val_sorted[:,1], outputs_val_sorted[:,0], y_val_sorted, y_input, delta = 0.1)

    width = outputs_val_sorted[:,1] - outputs_val_sorted[:,0]
    quantile_width_data = np.quantile(y_input, 0.95, axis = 0) - np.quantile(y_input, 0.05, axis = 0)
    norm_width = width/quantile_width_data
    PIwidth[:,j] = norm_width

    print(f'Data index: {j}: PICP = {PICP[j]}, PINAW = {PINAW[j]}, PINALW = {PINALW[j]}, avgPIwidth = {np.mean(PIwidth[:,j])}')

    saved_result = {'outputs_train':outputs_train_all, 'outputs_val':outputs_val_all
                    , 'PICP_val':PICP, 'PINAW': PINAW, 'PINALW':PINALW, 'Winkler':Winkler, 'PIwidth':PIwidth
                    , 'allytrain':allytrain, 'allxtrain': allxtrain, 'allyval': allyval, 'allxval':allxval
                   , 'allmeanpred_train':allmeanpred_train, 'allvarpred_train':allvarpred_train ,'epistemic_uncertainty_train':epistemic_uncertainty_train, 'aleatoric_uncertainty_train':aleatoric_uncertainty_train ,'total_uncertainty_train':total_uncertainty_train, 'predictive_mean_train':predictive_mean_train 
                   , 'allmeanpred_val':allmeanpred_val, 'allvarpred_val':allvarpred_val ,'epistemic_uncertainty_val':epistemic_uncertainty_val, 'aleatoric_uncertainty_val':aleatoric_uncertainty_val ,'total_uncertainty_val':total_uncertainty_val, 'predictive_mean_val':predictive_mean_val}

    filename = f'deepensemble_performance_dgpsin.pkl'
    # # Save
    dict_path = os.path.join(saveresultfolderpath, filename)
    with open(dict_path, 'wb') as pickle_file:
        pickle.dump(saved_result, pickle_file)
    
##########################################
print('---------- Finished ----------')
filenamedone = f'deepensemble_performance_dgpsin_done.pkl'
##########################################
# # Save
dict_path = os.path.join(saveresultfolderpath, filenamedone)
with open(dict_path, 'wb') as pickle_file:
    pickle.dump(saved_result, pickle_file)
    