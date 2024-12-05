import torch
import math
import os
import numpy as np
import pandas as pd
import argparse
import json
import re
import dgl
from tqdm import tqdm
import pickle
import torch.nn as nn 
from collections import OrderedDict
import random
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def hla_key_and_setTrans(file1,file2):#设置hla的key,然后对数据集进行一个转化，其中hla用key表示
    
    #此函数值的是提取全部hla构分子间相互作用图的情况
    #file1为hla全部分子数据,file2值的是训练数据文件或其他相互作用文件

    with open(file1,'r') as f1:
        data=pd.read_csv(f1)
    f1.close()

    hla_data=np.array(data)
    hla_dict=dict()
    for i in range(len(hla_data)):
        hla_dict[i]=hla_data[i][0]
    hla_dict_rev={v:k for k,v in hla_dict.items()}
    
    with open(file2,'r') as f2:
        data1=pd.read_csv(f2)
    f2.close()
    train=np.array(data1)

    train_hla=[]
    train_entry=[]
    train_hla_dict=dict()
    for i in range(len(train)):
        train_hla.append(hla_dict_rev[train[i][0]])
        train_entry.append([train_hla[i],train[i][1],train[i][2]])
        if train_hla[i] not in train_hla_dict.keys():
            train_hla_dict[train_hla[i]]=train[i][0]

    
    return train_entry,train_hla,train_hla_dict,hla_dict

def hla_key_and_setTrans_2(file1,file2):#设置hla的key,然后对数据集进行一个转化，其中hla用key表示
    
    #此函数值的是提取全部hla构分子间相互作用图的情况
    #file1为hla全部分子数据,file2值的是训练数据文件或其他相互作用文件
    #与第一个的区别是提取hla,peptide,y的列数不同，由于文件数据存储不同

    with open(file1,'r') as f1:
        data=pd.read_csv(f1)
    f1.close()

    hla_data=np.array(data)
    hla_dict=dict()
    for i in range(len(hla_data)):
        hla_dict[i]=hla_data[i][0]
    hla_dict_rev={v:k for k,v in hla_dict.items()}
    
    with open(file2,'r') as f2:
        data1=pd.read_csv(f2)
    f2.close()
    train=np.array(data1)

    train_hla=[]
    train_entry=[]
    train_hla_dict=dict()
    for i in range(len(train)):
        #train_hla.append(k for k,v in hla_dict.items() if v==train[i][0])
        train_hla.append(hla_dict_rev[train[i][0]])
        train_entry.append([train_hla[i],train[i][1],train[i][2]])
        #print('train_entry',train_entry)
        if train_hla[i] not in train_hla_dict.keys():
            train_hla_dict[train_hla[i]]=train[i][0]

    
    return train_entry,train_hla,train_hla_dict,hla_dict

   
    
#建立函数，得到comm_hla_sequence中所有hla的全序列
def hla_full_sequence():
    with open('data/contact/common_hla_sequence.csv','r') as f:
            data=pd.read_csv(f)
    f.close()
    hla_list=np.array(data)
    hla_full_sequence_dict=dict()
    hla_have=[]
    with open('data/contact/hla_prot.fasta','r') as f:
        for line in f.readlines():
            if '>' in line:
                flag=True    #用此标记表示是否需要进行序列后续添加
                hla=''       #设定
                list= line.split(' ')
                elem=re.split(r'[*:]',list[1])    #将不符合规定的HLA进行拆分
                if len(elem)<3:
                    flag=False
                    continue
                hla='HLA-'+elem[0]+'*'+elem[1]+':'+elem[2]   #合并成符合规定的形式
                if hla in hla_have:
                    flag=False
                    continue
                
                elif hla in hla_list:
                    hla_have.append(hla)
                    hla_full_sequence_dict[hla]=''
                else:
                    flag=False
            else:
                if flag==True:
                    line.strip('')
                    hla_full_sequence_dict[hla]=str(hla_full_sequence_dict[hla])+line

    hla_full_sequence_rev={v:k for k,v in hla_full_sequence_dict.items()}    #得到反序列字典
    
    save_dir='data/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file1=save_dir+'common_hla_full_sequence.fasta'
    save_file2=save_dir+'common_hla_full_sequence_rev.fasta'
    write_f=open(save_file1,'w')
    write_f.write (json.dumps (hla_full_sequence_dict))
    write_f.close()
    write_f=open(save_file2,'w')
    write_f.write (json.dumps (hla_full_sequence_rev))
    write_f.close()



def hla_key_full_sequence():    #得到以hla对应键值为键，value为全序列的文件

    data=json.load(open('data/contact/common_hla_full_sequence.txt'), object_pairs_hook=OrderedDict)
    data_keys=list(data.keys())
    with open('data/contact/common_hla_sequence.csv','r') as f1:
        data1=pd.read_csv(f1)
    f1.close()

    hla_data=np.array(data1)
    hla_dict=dict()
    for i in range(len(hla_data)):
        #hla_dict[1000+i]=hla_data[i][0]
        hla_dict[i]=hla_data[i][0]
    hla_dict_rev={v:k for k,v in hla_dict.items()}

    key_hla_full_sequence=dict()
    for i in range(len(data_keys)):
        key=hla_dict_rev[data_keys[i]]
        key_hla_full_sequence[key]=data[data_keys[i]]

    save_dir='data/contact/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file1=save_dir+'common_hla_key_full_sequence.txt'
    write_f=open(save_file1,'w')
    write_f.write (json.dumps (key_hla_full_sequence))
    write_f.close()
    #print('写入成功')
    


#构建HLA分子间关系图网络
def classtopo_graph(data_path='../data', flod=0,type=0,device=torch.device('cpu')):
    edges = []
    edge_type = {}
    e_feat = []
    #得到图的边
    save_dir='../data/hla_hla/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if type==0:
        load_file=save_dir+'categraph2008_newB07.txt'
    else:
        load_file=save_dir+'categegories_train{}.txt'.format(fold)   
    with open(load_file, 'r') as file:
        for line in tqdm(file.readlines(), desc='loading hla_categegories'):
            h, r, t = line.strip().split(',')
            [h, r, t] = [int(h), int(r), int(t)]
            if h == t:
                continue
            edges.append([h, t])
            
    return edges

