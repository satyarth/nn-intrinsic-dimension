import torch
from torch import nn as nn
from torch.autograd import Variable

import numpy as np

from IPython import display
from copy import deepcopy

from modules import *

np.random.seed(123)
torch.manual_seed(123)
random.seed(123)
torch.cuda.manual_seed_all(123)

def train_network(network, optimizer, criterion, train_data, test_data=None, epoches=50, flatten = False):
    
    train_loss, val_acc, train_acc = list(), list(), list()
    try:
        assert network.d == len([param for param in network.parameters() if param.requires_grad][0]), '#parameters != d'
        assert network.d == len(optimizer.param_groups[0]['params'][0]), '#params in optimizer != d'
    except:
        print('d is not explicitly specified')
    
    CUDA_ = torch.cuda.is_available()
    if CUDA_:
        print('Training with cuda')
    
    for epoch in range(epoches):
        print("Epoch", epoch)
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
                try:
                    train_loss.append(loss.cpu().data.item())
                except:
                    train_loss.append(loss.cpu().data.numpy()[0])
            else:
                try:
                    train_loss.append(loss.data.item())
                except:
                    train_loss.append(loss.data.numpy()[0])
                    
        if not test_data is None:
            network.train(False)
            th=0
            b_score = list()
            #print('Validating')
            for X_batch_, y_batch_ in test_data:
                
                if flatten:
                    X_batch_ = X_batch_.view(X_batch_.size(0),-1)
                    
                if CUDA_:
                    pred_ = network(Variable(X_batch_).cuda())
                    pred_ = pred_.cpu()
                else:
                    pred_ = network(Variable(X_batch_))
                target_ = pred_.max(1)[1].data.numpy()
                y_pred = y_batch_.numpy()
                b_score.append((target_ == y_pred).mean())
                th+=1
                if th>np.inf:
                    break
            score = np.mean(b_score)
            val_acc.append(score)
            
            if epoch%4==0:
                print(str(epoch)+':'+str(score), end=' ')
                
            th=0
            b_score = list()
            for X_batch_, y_batch_ in train_data:
                
                if flatten:
                    X_batch_ = X_batch_.view(X_batch_.size(0),-1)
                    
                if CUDA_:
                    pred_ = network(Variable(X_batch_).cuda())
                    pred_ = pred_.cpu()
                else:
                    pred_ = network(Variable(X_batch_))
                target_ = pred_.max(1)[1].data.numpy()
                y_pred = y_batch_.numpy()
                b_score.append((target_ == y_pred).mean())
                th+=1
                if th>np.inf:
                    break
            score = np.mean(b_score)
            train_acc.append(score)
            
            
            #print results
            # plt.clf()

            # fig, ax = plt.subplots(1,2,figsize=(15,7))

            # ax[0].plot(train_loss[100:])
            # ax[0].set_title('Train loss')

            # ax[1].plot(val_acc)
            # ax[1].set_title('Val acc')

            # display.clear_output(wait=True)
            # display.display(plt.gcf())

    output = {"test_acc_history": val_acc,
              "train_acc_history": train_acc,
              "train_loss":train_loss}
            
    return network, output


def test_network(trained_network, test_data, flatten=False):
    
    CUDA_ = torch.cuda.is_available()
    trained_network.train(False)
    b_score = list()
    
    print('Testing')
    
    for X_batch_, y_batch_ in test_data:
        
        if flatten:
            X_batch_ = X_batch_.view(X_batch_.size(0),-1)

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
                d=100, verbose=True):
    
    results = {}
    
    network_args['d'] = int(d)
    
    if torch.cuda.is_available():
        network_ = network_class(**network_args).cuda()
    else:
        network_ = network_class(**network_args)
        
    network_grads = [param for param in network_.parameters() if param.requires_grad]
    opt_ = optimizer_class(params = network_grads, **optimizer_args)
    
    #train network
    d_trained, output = train_network(network_, optimizer=opt_, criterion=criterion,
                              train_data=train_data, test_data=test_data, epoches=epoches, flatten=flatten)
    
    output['d'] = d
    #test network
    d_score = test_network(trained_network=d_trained, test_data=test_data, flatten=flatten)
                
    return output
