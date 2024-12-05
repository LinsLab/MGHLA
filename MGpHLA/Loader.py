import os
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
from torch_geometric.nn import GCNConv, GCN2Conv, GATConv, global_max_pool as gmp, global_add_pool as gap, \
    global_mean_pool as gep, global_sort_pool
import torch
import numpy as np

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
from data_transform import transform_data, CATHDataset

sys.path.append('/')
from utils import hla_key_and_setTrans,hla_key_and_setTrans_2,hla_key_full_sequence
#from feature_extraction_contact import sequence_to_graph,batch_seq_feature,batch_seq_feature_Bi
from feature_extraction import sequence_to_graph,batch_seq_feature_Bi

#还需要引入数据处理文件中的函数(得到hla_dict等)
#还需要引入特征提取文件中提取序列特征的函数
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


class HPIDataset_peps_new_blousm_pep(InMemoryDataset):
    def __init__(self,root='../data',xh=None, y=None, transform=None,
                 pre_transform=None, hla_contact_graph=None, hla_blousm=None,peptide_key=None,hla_3d_graph=None,peps_feature=None,all_hla_len=None):
        super(HPIDataset_peps_new_blousm_pep, self).__init__(root,transform, pre_transform)

        self.hla=xh
        self.peptide_key=peptide_key    #peptide的key就是它本身
        self.y=y
        self.hla_contact_graph=hla_contact_graph
        self.hla_3d_graph=hla_3d_graph
        self.hla_blousm=hla_blousm
        self.peps_feature=peps_feature
        self.all_hla_len=all_hla_len
        self.process(xh,peptide_key,y,hla_contact_graph,hla_blousm,hla_3d_graph,all_hla_len,peps_feature)

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

    def process(self, xh, peptide_key, y, hla_contact_graph,hla_blousm,hla_3d_graph,all_hla_len,peps_feature):
        assert (len(xh) == len(peptide_key) and len(xh) == len(y)), 'The three lists must have the same length!'
        data_list_hlacontact = []
        data_list_hla3D=[]
        data_list_pep = []
        data_list_pep_feature=[]
        data_len = len(xh)
        for i in tqdm(range(data_len)):
            hla = int(xh[i])
            pep_key = peptide_key[i]
            pep_feature=peps_feature[i]
            labels = int(y[i])
            #contact_graph_data
            hla_size,hla_contact_features,hla_edge_index,hla_edges_weights=hla_contact_graph[hla]
            hla_features=torch.cat((torch.Tensor(hla_contact_features),torch.Tensor(hla_blousm[hla])),axis=-1)
            residue_indices = [7,9,24,45,59,62,63,66,67,69,70,73,74,76,77,80,81,84,95,97,99,114,116,118,143,147,150,
                       152,156,158,159,163,167,171]
            residue_indices = [i - 1 for i in residue_indices]
            valid_indices = [i for i in residue_indices if i < hla_features.size(0)]
            hla_features[valid_indices] = hla_features[valid_indices] * 3.0  # 将这些节点特征放大2倍
        
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
           
        self.data_hla_contact = data_list_hlacontact
        self.data_hla_3D=data_list_hla3D
        self.data_pep=data_list_pep

        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # return GNNData_mol, GNNData_pro
        return self.data_hla_contact[idx], self.data_hla_3D[idx],self.data_pep[idx]
    
    
    
def train_predict_div(train_file,vaild_file,dataset_structure,seed):    #数据集应该为未划分训练集测试集验证集的全数据集
    
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
    
    # 构建接触图/home/layomi/drive1/项目代/home/layomi/drive1/项目代码/MMGHLA_Classification_Topography码/MMGHLA_Classification_Topography
    process_dir = os.path.join('..', 'data/pre_process')
    hla_distance_dir = os.path.join(process_dir, 'contact/distance_map')  # numpy .npy file   这里给出接触图的路径
    hla_key_blousm_dict=read_hla_blousm()   #对hla进行blousm编码
    #对hla进行接触图获取和理化性质计算
    hla_graph = dict()    
    all_hla_len=[]
    for i in tqdm(range(len(hla_dict))):
        key = i
        seq=hla_full_sequence_dict[key]
        g_h = sequence_to_graph(key, seq, hla_distance_dir)
        hla_graph[key] = g_h
        all_hla_length=len(seq)
        all_hla_len.append(all_hla_length)
        
    
           
    train_hlas,train_peptides,train_Y=np.asarray(train_entries)[:,0],np.asarray(train_entries)[:,1],np.asarray(train_entries)[:,2]
    vaild_hlas,vaild_peptides,vaild_Y=np.asarray(vaild_entries)[:,0],np.asarray(vaild_entries)[:,1],np.asarray(vaild_entries)[:,2]
    #对pepetide进行理化性质计算
    train_pep_lihua=batch_seq_feature_Bi(train_peptides,15,12)
    vaild_pep_lihua=batch_seq_feature_Bi(vaild_peptides,15,12)
    #对peptide进行blousm计算
    train_pep_blousm=read_pep_blousm(train_peptides)
    vaild_pep_blousm=read_pep_blousm(vaild_peptides)    
   
    train_pep_feature=torch.cat((torch.tensor(train_pep_lihua),torch.tensor(train_pep_blousm)),axis=-1)
    vaild_pep_feature=torch.cat((torch.tensor(vaild_pep_lihua),torch.tensor(vaild_pep_blousm)),axis=-1)
    
    #这一段的意义是给训练测试数据集中每个hla配备了接触图，所以，我必须把3-D分子图也输入进入，给每个hla配备上
    train_dataset=HPIDataset_peps_new_blousm_pep(xh=train_hlas, peptide_key=train_peptides,
                               y=train_Y.astype(float), hla_contact_graph=hla_graph,hla_blousm=hla_key_blousm_dict,hla_3d_graph=dataset_structure,peps_feature=train_pep_feature,all_hla_len=all_hla_len)
    
    vaild_dataset=HPIDataset_peps_new_blousm_pep(xh=vaild_hlas, peptide_key=vaild_peptides,
                               y=vaild_Y.astype(float), hla_contact_graph=hla_graph,hla_blousm=hla_key_blousm_dict,hla_3d_graph=dataset_structure,peps_feature=vaild_pep_feature,all_hla_len=all_hla_len)

    return train_dataset, vaild_dataset
    

def test_data_div(test_file):
    #得到文件中hla键的顺序
    common_hla_file='../data/contact/common_hla_sequence.csv'
    test_entries,test_hla_keys,test_hla_dict,hla_dict=hla_key_and_setTrans(common_hla_file,test_file)
    
    hla_full_sequence_dict=dict()
    full_seq_dict=json.load(open('../data/contact/common_hla_key_full_sequence_new.txt'), object_pairs_hook=OrderedDict)
    for key,value in full_seq_dict.items():
        key=int(key)
        hla_full_sequence_dict[key]=value
        
    # 构建接触图/home/layomi/drive1/项目代/home/layomi/drive1/项目代码/MMGHLA_Classification_Topography码/MMGHLA_Classification_Topography
    process_dir = os.path.join('..', 'data/pre_process')
    hla_distance_dir = os.path.join(process_dir, 'contact/distance_map')  # numpy .npy file   这里给出接触图的路径
    hla_key_blousm_dict=read_hla_blousm()   #对hla进行blousm编码
    hla_graph = dict()    
    all_hla_len=[]
    for i in tqdm(range(len(hla_dict))):
        key = i
        seq=hla_full_sequence_dict[key]
        g_h = sequence_to_graph(key, seq, hla_distance_dir)
        hla_graph[key] = g_h
        all_hla_length=len(seq)
        all_hla_len.append(all_hla_length)
   
    
    test_hlas,test_peptides,test_Y=np.asarray(test_entries)[:,0],np.asarray(test_entries)[:,1],np.asarray(test_entries)[:,2]
    cath = CATHDataset(os.path.join('/home1/layomi/项目代码/MMGHLA_CT_blousm/data/aphlafold2', 'structure.jsonl'))
    dataset_structure = transform_data(cath.data, max_pro_seq_len)
   
    test_pep_lihua=batch_seq_feature_Bi(test_peptides,15,12)
    test_pep_blousm=read_pep_blousm(test_peptides)  
    test_pep_feature=torch.cat((torch.tensor(test_pep_lihua),torch.tensor(test_pep_blousm)),axis=-1)  
    
    test_dataset=HPIDataset_peps_new_blousm_pep(xh=test_hlas, peptide_key=test_peptides,
                               y=test_Y.astype(float), hla_contact_graph=hla_graph,hla_blousm=hla_key_blousm_dict,hla_3d_graph=dataset_structure,peps_feature=test_pep_feature,all_hla_len=all_hla_len
                               )
    return test_dataset
    



def read_hla_blousm():  #得到所有序列的blousm编码
    hla_full_sequence_dict=dict()
    full_seq_dict=json.load(open('../data/contact/common_hla_key_full_sequence_new.txt'), object_pairs_hook=OrderedDict)
    for key,value in full_seq_dict.items():
        key=int(key)
        hla_full_sequence_dict[key]=value[24:]
        
    seq_dict_24=hla_full_sequence_dict
    hla_key_blosum_dict=dict()
    for allele in seq_dict_24.keys():
        hla_blosum=[]
        for ra in seq_dict_24[allele]:
            hla_blosum.append(blosum_matrix[aa[ra]])
        hla_blousm = np.array(hla_blosum)
        
        hla_key_blosum_dict[allele]=hla_blousm
    
    return hla_key_blosum_dict


def read_pep_blousm(peptides):      
    pep_blousm_list=[]   
    for i in range(len(peptides)):    
        pep=peptides[i]
        pep_blosum=[]
        for residue_index in range(15):
            #Encode the peptide sequence in the 1-12 columns, with the N-terminal aligned to the left end
            #If the peptide is shorter than 12 residues, the remaining positions on
            #the rightare filled will zero-padding
            if residue_index < len(pep):
                pep_blosum.append(blosum_matrix[aa[pep[residue_index]]])
            else:
                pep_blosum.append(np.zeros(20).tolist())
        for residue_index in range(15):
            #Encode the peptide sequence in the 13-24 columns, with the C-terminal aligned to the right end
            #If the peptide is shorter than 12 residues, the remaining positions on
            #the left are filled will zero-padding
            if 15 - residue_index > len(pep):
                pep_blosum.append(np.zeros(20).tolist()) 
            else:
                pep_blosum.append(blosum_matrix[aa[pep[len(pep) - 15 + residue_index]]])
                
        pep_blosum = np.array(pep_blosum)
        pep_blousm_list.append(pep_blosum)
    return pep_blousm_list
        
        