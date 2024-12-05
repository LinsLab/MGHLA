import os
import torch
import numpy as np
import math
import pandas as pd
import re
import time
import datetime
import json
import random

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
import warnings
import torch.nn as nn
warnings.filterwarnings("ignore")
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
from torch_geometric.nn import GCNConv, GCN2Conv, GATConv, global_max_pool as gmp, global_add_pool as gap, global_mean_pool as gep, global_sort_pool
from sklearn import metrics
from sklearn import preprocessing
from scipy import interp

from collections import Counter
from functools import reduce
from tqdm import tqdm, trange
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from utils import classtopo_graph

from performance import performances,performances_to_pd,f_mean

threshold=0.5

def train(fold,model, device, train_loader, optimizer, epoch, epochs,loss_fn, TRAIN_BATCH_SIZE=512):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    time_train_ep = 0
    model.train()
    LOG_INTERVAL = 10
    y_true_train_list, y_prob_train_list = [], []
    loss_train_list = []
    
    class_topo_node_features1,class_topo_node_features2=classtopo_encoder(fold)
    class_topo_node_features1=torch.tensor(class_topo_node_features1).to(device)
    class_topo_node_features2=torch.tensor(class_topo_node_features2).to(device)
    edge_index=classtopo_graph(device=device)
    edge_index=torch.tensor(edge_index).transpose(1, 0)
    edge_index=edge_index.to(device)
    for batch_idx,data in tqdm(enumerate(train_loader)):
       
        t1 = time.time()
        data_hla_contact = data[0].to(device)
        data_hla_3D=data[1].to(device)
        data_pep = data[2]
        
        
        output= model(data_hla_contact,data_hla_3D,data_pep,class_topo_node_features1,class_topo_node_features2,edge_index)
        if torch.isnan(output).any():
            print("NaN or Inf in input")
        loss = loss_fn(output, data_hla_contact.y.view(-1, 1).float().to(device))
        time_train_ep += time.time() - t1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * TRAIN_BATCH_SIZE,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

        y_true_train = data_hla_contact.y.view(1, -1).cpu().numpy()
        y_prob_train=output.view(1,-1)[0].cpu().detach().numpy()
        y_true_train=y_true_train.tolist()
        y_true_train_list.extend([i for list in y_true_train for i in list]) 
        y_prob_train_list.extend(y_prob_train)
        loss_train_list.append(loss)
    
    y_pred_train_list = transfer(y_prob_train_list, threshold)
    ys_train = (y_true_train_list, y_pred_train_list, y_prob_train_list)
    print('Train (Ep avg): Epoch-{}/{} | Loss = {:.4f} | Time = {:.4f} sec'.format(epoch, epochs, f_mean(loss_train_list), time_train_ep))
    metrics_train = performances(y_true_train_list, y_pred_train_list, y_prob_train_list, print_ = True)
    return ys_train, loss_train_list, metrics_train, time_train_ep




# predict
def predicting(fold,model, device, loader,epoch, epochs,loss_fn=nn.BCELoss()):
    model.eval()
    y_true_val_list, y_prob_val_list = [], []
    loss_val_list = []
    class_topo_node_features1,class_topo_node_features2=classtopo_encoder(fold)
    class_topo_node_features1=torch.tensor(class_topo_node_features1).to(device)
    class_topo_node_features2=torch.tensor(class_topo_node_features2).to(device)
    edge_index=classtopo_graph(device=device)
    
    edge_index=torch.tensor(edge_index).transpose(1, 0)
    edge_index=edge_index.to(device)
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(loader)):
       
            data_hla_contact = data[0].to(device)
            data_hla_3D=data[1].to(device)
            data_pep = data[2]
            
            output= model(data_hla_contact,data_hla_3D,data_pep,class_topo_node_features1,class_topo_node_features2,edge_index)
            loss = loss_fn(output, data_hla_contact.y.view(-1, 1).float().to(device))
            
            y_true_val = data_hla_contact.y.view(1, -1).cpu().numpy()
            y_prob_val = output.view(1,-1)[0].cpu().detach().numpy()
            y_true_val=y_true_val.tolist()
            y_true_val_list.extend([i for list in y_true_val for i in list])
            y_prob_val_list.extend(y_prob_val)
            loss_val_list.append(loss)
        y_pred_val_list = transfer(y_prob_val_list, threshold)
        ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)
        
        print('Val  Epoch-{}/{}: Loss = {:.6f}'.format(epoch, epochs, f_mean(loss_val_list)))
        metrics_val = performances(y_true_val_list, y_pred_val_list, y_prob_val_list, print_ = True)
    return ys_val, loss_val_list, metrics_val


def eval_step(model, val_loader,threshold = 0.5, use_cuda = False):
    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()
    y_true_test_list, y_prob_test_list = [], []
    y_preb_test_list=[]
    loss_test_list = []
   
    class_topo_node_features1,class_topo_node_features2=classtopo_encoder('test')
    class_topo_node_features1=torch.tensor(class_topo_node_features1).to(device)
    class_topo_node_features2=torch.tensor(class_topo_node_features2).to(device)
    edge_index=classtopo_graph(device=device)
    edge_index=torch.tensor(edge_index).transpose(1, 0)
    edge_index=edge_index.to(device)
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate( val_loader)):
            data_hla_contact = data[0].to(device)
            data_hla_3D=data[1].to(device)
            data_pep = data[2]
          
            output = model(data_hla_contact,data_hla_3D,data_pep,class_topo_node_features1,class_topo_node_features2,edge_index)
            #print(output)
            y_true_test = data_hla_contact.y.view(1, -1).cpu().numpy()
            y_prob_test = output.view(1,-1)[0].cpu().detach().numpy()
            y_true_test=y_true_test.tolist()
            y_true_test_list.extend([i for list in y_true_test for i in list])
            y_prob_test_list.extend(y_prob_test)
            
        y_pred_test_list = transfer(y_prob_test_list, threshold)
        ys_test = (y_true_test_list, y_pred_test_list, y_prob_test_list)
        
        metrics_test = performances(y_true_test_list, y_pred_test_list, y_prob_test_list, print_ = True)
    return ys_test, metrics_test

def collate(data_list):
    batchA_contact = Batch.from_data_list([data[0] for data in data_list])
    batchA_3D=Batch.from_data_list([data[1] for data in data_list])
    
    hla_key=[]
    batchB=[]
    for data in data_list:
        data_pep=data[2]
        batchB.append(data_pep)
        hla_k=data[0].hla_key
        hla_key.append(hla_k)
    return batchA_contact,batchA_3D,batchB,hla_key


def transfer(y_prob, threshold = 0.5):
    return np.array([[0, 1][x > threshold] for x in (y_prob)])



def classtopo_encoder(fold_test):
    #初始化整个类-拓扑图的节点特征
   
    hla_feature_mean_file='../data/fold_data/fold_data_new2/train_pos/classtopo/hla_onehot_1.txt'
    if type(fold_test)==int:
        cate_hla_key_feature_file='../data/fold_data/fold_data_new2/train_pos/2008_no_onehot/cate_hla_key_feature{}.txt'.format(fold_test)
    elif fold_test=='test':
        cate_hla_key_feature_file='../data/fold_data/fold_data_new2/train_pos/2008_no_onehot/test_cate_hla_key_feature.txt'
    
    with open(hla_feature_mean_file,'r') as f:
        data_hla=json.load(f)   
    with open(cate_hla_key_feature_file, 'r') as file1:
        data = json.load(file1)
    #hla_nodes_feature = data['hla_nodes']     #也是一个字典
    class_nodes_feature = data['class_nodes']
    hla_nodes_feature=data_hla
    hlas_features=[]
    class_features=[]
    all_feature=[]
    for key ,value in hla_nodes_feature.items():
        hlas_features.append(value)
    for key ,value in class_nodes_feature.items():
        class_features.append(value)
    return hlas_features,class_features


