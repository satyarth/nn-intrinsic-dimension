import torch
from torch import nn as nn
from torch.autograd import Variable

import numpy as np
from tqdm import tqdm, tqdm_notebook

from IPython import display
from copy import deepcopy
import matplotlib.pyplot as plt

from modules import *


def train_network(network, optimizer, criterion, train_data, test_data=None, epoches=50, flatten = False):
    
    train_loss, val_acc = list(), list()
    
    CUDA_ = torch.cuda.is_available()
    if CUDA_:
        print('Training with cuda')
    
    for epoch in tqdm_notebook(range(epoches)):
        network.train(True)
        for X_batch, y_batch in train_data:
            if flatten:
                X_batch = X_batch.view(X_batch.size(0),-1)
                
            if CUDA_:
                pred = network(Variable(X_batch).cuda())
                target = Variable(y_batch).cuda()
            else:
                pred = network(Variable(X_batch))
                target = Variable(y_batch)

            loss = criterion(pred, target)

            # train on batch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if CUDA_:
                train_loss.append(loss.cpu().data.item())
            else:
                train_loss.append(loss.data.item())
            
        if not test_data is None:
            network.train(False) # disable dropout / use averages for batch_norm
            th=0
            b_score = list()
            print('Validating')
            for X_batch_, y_batch_ in test_data:
                
                if flatten:
                    X_batch_ = X_batch.view(X_batch_.size(0),-1)
                    
                if CUDA_:
                    pred_ = network(Variable(X_batch_).cuda())
                    pred_ = pred_.cpu()
                else:
                    pred_ = network(Variable(X_batch_))
                target_ = pred_.max(1)[1].data.numpy()
                y_pred = y_batch_.numpy()
                b_score.append((target_ == y_pred).mean())
                th+=1
                if th>500:
                    break
            score = np.mean(b_score)
            val_acc.append(score)
            
            #print results
            plt.clf()

            fig, ax = plt.subplots(1,2,figsize=(15,7))

            ax[0].plot(train_loss[100:])
            ax[0].set_title('Train loss')

            ax[1].plot(val_acc)
            ax[1].set_title('Val acc')

            display.clear_output(wait=True)
            display.display(plt.gcf())
            
    return network


def test_network(trained_network, test_data, flatten=False):
    
    CUDA_ = torch.cuda.is_available()
    trained_network.train(False)
    b_score = list()
    
    print('Testing')
    
    for X_batch_, y_batch_ in test_data:
        
        if flatten:
            X_batch_ = X_batch.view(X_batch_.size(0),-1)

        if CUDA_:
            pred_ = trained_network(Variable(X_batch_).cuda())
            pred_ = pred_.cpu()
        else:
            pred_ = trained_network(Variable(X_batch_))
            
        target_ = pred_.max(1)[1].data.numpy()
        y_pred = y_batch_.numpy()
        b_score.append((target_ == y_pred).mean())

    return np.mean(b_score)


def Dgrid_train(network_class, network_args, optimizer_class, optimizer_args,
                criterion, train_data, test_data, epoches=50, flatten = False,
                d_range=None, verbose=True):
    
    d_scores_list = list()
    
    for i, each_d in tqdm(enumerate(d_range)):
        
        network_args['d'] = each_d
        network_ = network_class(**network_args)
        
        network_grads = [param for param in network_.parameters() if param.requires_grad]
        opt_ = optimizer_class(params = network_grads, **optimizer_args)
        
        #train network
        d_trained = train_network(network_, optimizer=opt_, criterion=criterion,
                                  train_data=train_data, test_data=None, epoches=epoches, flatten=flatten)
        
        #test network
        d_score = test_network(trained_network=d_trained, test_data=test_data, flatten=flatten)
        d_scores_list.append(d_score)
        
        if verbose:
            #print results
            plt.clf()

            #fig, ax = plt.subplots(1,2,figsize=(15,7))

            plt.bar(d_range[:i+1], d_scores_list)

            display.clear_output(wait=True)
            display.display(plt.gcf())
            
    return d_range, d_scores_list