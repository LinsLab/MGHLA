import os
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
from torch_geometric.nn import GCNConv, GCN2Conv, GATConv, global_max_pool as gmp, global_add_pool as gap, \
    global_mean_pool as gep, global_sort_pool
import torch
import numpy as np
import re


import math
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import re
import time
import datetime
import random
import torch.nn as nn
from data_transform import transform_data, CATHDataset
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from scipy import interp
import warnings
warnings.filterwarnings("ignore")

#from model2 import make_data,MyDataSet
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

import traceback
import pandas as pd
import numpy as np
import networkx as nx
import sys
import os
import random

import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm
from utils import *

sys.path.append('/')
from utils import hla_key_and_setTrans,hla_key_and_setTrans_2,hla_key_full_sequence
#from feature_extraction_contact import sequence_to_graph,batch_seq_feature
from feature_extraction_contact_only_onehot import sequence_to_graph,batch_seq_feature_only_onehot


TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
max_pro_seq_len = 348

vocab = np.load('./model/vocab_dict.npy', allow_pickle = True).item()
vocab_size = len(vocab)

def read_blosum(path):
    '''
    Read the blosum matrix from the file blosum50.txt
    Args:
        1. path: path to the file blosum50.txt
    Return values:
        1. The blosum50 matrix
    '''
    f = open(path,"r")
    blosum = []
    for line in f:
        blosum.append([(float(i))/10 for i in re.split("\t",line)])
        #The values are rescaled by a factor of 1/10 to facilitate training
    f.close()
    return blosum

aa={"A":0,"R":1,"N":2,"D":3,"C":4,"Q":5,"E":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19}     
#Load the blosum matrix for encoding
path_blosum = '../data/blosum50.txt'
blosum_matrix = read_blosum(path_blosum)


def P_or_N(list_entry):
    peptide_seq=[]
    p_entries=[]
    n_entries=[]
    for i in range(len(list_entry)):
        peptide_seq.append(list_entry[i][1])
        if float(list_entry[i][2])==1:
            p_entries.append(list_entry[i])
        if float(list_entry[i][2])==0:
            n_entries.append(list_entry[i])
    peptide_type=list(set(peptide_seq))
    return p_entries,n_entries,peptide_type


def train_predict_div_pep_new_blousm_onlyonehot_weight(train_file,vaild_file,dataset_structure,seed):    
    
    common_hla_file='../data/contact/common_hla_sequence.csv'
    train_entries,train_hla_keys,train_hla_dict,hla_dict=hla_key_and_setTrans(common_hla_file,train_file)
    vaild_entries,vaild_hla_keys,vaild_hla_dict,hla_dict=hla_key_and_setTrans(common_hla_file,vaild_file)
    
      
    train_p_entries,train_n_entries,train_peptide_type=P_or_N(train_entries)
    vaild_p_entries,vaild_n_entries,vaild_peptide_type=P_or_N(vaild_entries)
    
    random.seed(seed)
    random.shuffle(train_entries)
    random.shuffle(vaild_entries)
  
    
    


    hla_full_sequence_dict=dict()
    full_seq_dict=json.load(open('../data/contact/common_hla_key_full_sequence_new.txt'), object_pairs_hook=OrderedDict)
    for key,value in full_seq_dict.items():
        key=int(key)
        hla_full_sequence_dict[key]=value
    
    process_dir = os.path.join('..', 'data/pre_process')
    hla_distance_dir = os.path.join(process_dir, 'contact/distance_map')  # numpy .npy file  Provide the path for the contact map
   
    hla_graph = dict()    
    all_hla_len=[]
    for i in tqdm(range(len(hla_dict))):
        key = i
        seq=hla_full_sequence_dict[key]
        g_h = sequence_to_graph(key, seq, hla_distance_dir)
        hla_graph[key] = g_h
        all_hla_length=len(seq)-24
        all_hla_len.append(all_hla_length)
           
    train_hlas,train_peptides,train_Y=np.asarray(train_entries)[:,0],np.asarray(train_entries)[:,1],np.asarray(train_entries)[:,2]
    vaild_hlas,vaild_peptides,vaild_Y=np.asarray(vaild_entries)[:,0],np.asarray(vaild_entries)[:,1],np.asarray(vaild_entries)[:,2]
    
    train_pep_feature=batch_seq_feature_only_onehot(train_peptides,15,21)
    vaild_pep_feature=batch_seq_feature_only_onehot(vaild_peptides,15,21)
    
    train_dataset=HPIDataset_peps_new_only_onehot(root='data', dataset=train_file + '_' + 'train', xd=train_hlas, peptide_key=train_peptides,
                               y=train_Y.astype(float), hla_contact_graph=hla_graph,hla_3d_graph=dataset_structure,peps_feature=train_pep_feature,all_hla_len=all_hla_len)
    
    vaild_dataset=HPIDataset_peps_new_only_onehot(root='data', dataset=vaild_file + '_' + 'dev', xd=vaild_hlas, peptide_key=vaild_peptides,
                               y=vaild_Y.astype(float), hla_contact_graph=hla_graph,hla_3d_graph=dataset_structure,peps_feature=vaild_pep_feature,all_hla_len=all_hla_len)
    
    return train_dataset, vaild_dataset

class HPIDataset_peps_new_only_onehot(InMemoryDataset):
    def __init__(self,root='../data',xh=None, y=None, transform=None,
                 pre_transform=None, hla_contact_graph=None, hla_blousm=None,peptide_key=None,hla_3d_graph=None,peps_feature=None,all_hla_len=None):
        super(HPIDataset_peps_new_only_onehot, self).__init__(root, transform, pre_transform)
 
        self.hla=xh
        self.peptide_key=peptide_key    #The key for the peptide is the peptide
        self.y=y
        self.hla_contact_graph=hla_contact_graph
        self.hla_3d_graph=hla_3d_graph
        self.hla_blousm=hla_blousm
        self.peps_feature=peps_feature
        self.all_hla_len=all_hla_len
        self.process(xh,peptide_key,y,hla_contact_graph,hla_3d_graph,all_hla_len,peps_feature)

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_hla.pt', self.dataset + '_data_pep.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, peptide_key, y, hla_contact_graph,hla_3d_graph,all_hla_len,peps_feature):
        
        assert (len(xd) == len(peptide_key) and len(xd) == len(y)), 'The three lists must have the same length!'
        data_list_hlacontact = []
        data_list_hla3D=[]
        data_list_pep = []
        data_list_pep_feature=[]
        data_len = len(xd)
        for i in tqdm(range(data_len)):
            hla = int(xd[i])
            pep_key = peptide_key[i]
            pep_feature=peps_feature[i]
           
            labels = y[i]
            #contact_graph_data
            hla_size,hla_contact_features,hla_edge_index,hla_edges_weights=hla_contact_graph[hla]
            hla_features=torch.tensor(hla_contact_features,dtype=torch.float32)
            #hla_features=torch.cat((torch.Tensor(hla_contact_features),torch.Tensor(hla_blousm[hla])),axis=-1)
            residue_indices = [7,9,24,45,59,62,63,66,67,69,70,73,74,76,77,80,81,84,95,97,99,114,116,118,143,147,150,
                       152,156,158,159,163,167,171]
            residue_indices = [i - 1 for i in residue_indices]
            
            valid_indices = [i for i in residue_indices if i < hla_features.size(0)]
            hla_features[valid_indices] = hla_features[valid_indices] * 3.0  # Scale these node features by a factor of 3
            ContactData_hla = DATA.Data(x=torch.Tensor(hla_features),
                                    edge_index=torch.LongTensor(hla_edge_index).transpose(1, 0),
                                    edge_weight=torch.FloatTensor(hla_edges_weights),hla_len=all_hla_len[hla],hla_key=hla,
                                    y=torch.FloatTensor([labels]))
            ContactData_hla.__setitem__('hla_size', torch.LongTensor([hla_size]))
            
            #3-d-mol-data
            ThreeD_hla=hla_3d_graph[hla]  #ThreeD_hla type=orch_geometric.data.Data
            
            Data_pep = {
                        'x': pep_feature,
                        'sequence': peptide_key,
                        'length': len(pep_feature)
                    }
            
            data_list_hlacontact.append(ContactData_hla)
            data_list_hla3D.append(ThreeD_hla)
            
            data_list_pep.append(Data_pep)
            #data_list_pep_feature.append(pep_feature)
            #hla_key.append(hla)

           
        self.data_hla_contact = data_list_hlacontact
        self.data_hla_3D=data_list_hla3D
        self.data_pep=data_list_pep
        #self.peps_feature=data_list_pep_feature
        #self.data_hla_key=hla_key
        
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # return GNNData_mol, GNNData_pro
        return self.data_hla_contact[idx], self.data_hla_3D[idx],self.data_pep[idx]



    
def test_data_div(test_file):
    
    common_hla_file='../data/contact/common_hla_sequence.csv'
    test_entries,test_hla_keys,test_hla_dict,hla_dict=hla_key_and_setTrans(common_hla_file,test_file)
    
    hla_full_sequence_dict=dict()
    full_seq_dict=json.load(open('../data/contact/common_hla_key_full_sequence_new.txt'), object_pairs_hook=OrderedDict)
    for key,value in full_seq_dict.items():
        key=int(key)
        hla_full_sequence_dict[key]=value
        
    process_dir = os.path.join('..', 'data/pre_process')
    hla_distance_dir = os.path.join(process_dir, 'contact/distance_map')  # numpy .npy file   Provide the path for the contact map
    hla_graph = dict()    
    all_hla_len=[]
    for i in tqdm(range(len(hla_dict))):
        key = i
        seq=hla_full_sequence_dict[key]
        g_h = sequence_to_graph(key, seq, hla_distance_dir)
        hla_graph[key] = g_h
        all_hla_length=len(seq)-24
        all_hla_len.append(all_hla_length)
   
    
    test_hlas,test_peptides,test_Y=np.asarray(test_entries)[:,0],np.asarray(test_entries)[:,1],np.asarray(test_entries)[:,2]
    cath = CATHDataset(os.path.join('../data/aphlafold2', 'structure.jsonl'))
    dataset_structure = transform_data(cath.data, max_pro_seq_len)
   
    test_pep_feature=batch_seq_feature_only_onehot(test_peptides,15,21)
    
    test_dataset=HPIDataset_peps_new_only_onehot(xh=test_hlas, peptide_key=test_peptides,
                               y=test_Y.astype(float), hla_contact_graph=hla_graph,hla_3d_graph=dataset_structure,peps_feature=test_pep_feature,all_hla_len=all_hla_len
                               )
    return test_dataset

