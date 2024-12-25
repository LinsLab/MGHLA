'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-05-21 22:36:55
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-06-14 22:30:19
'''
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
import torch.nn as nn
import pickle

def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']    
pro_res_aromatic_table = ['F', 'W', 'Y']    
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']     
pro_res_acidic_charged_table = ['D', 'E']     
pro_res_basic_charged_table = ['H', 'K', 'R']        

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}
res_weight_table['X'] = np.average([res_weight_table[k] for k in res_weight_table.keys()])

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}     
res_pka_table['X'] = np.average([res_pka_table[k] for k in res_pka_table.keys()])

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}    
res_pkb_table['X'] = np.average([res_pkb_table[k] for k in res_pkb_table.keys()])

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}    
res_pkx_table['X'] = np.average([res_pkx_table[k] for k in res_pkx_table.keys()])

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}
res_pl_table['X'] = np.average([res_pl_table[k] for k in res_pl_table.keys()])

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}
res_hydrophobic_ph2_table['X'] = np.average([res_hydrophobic_ph2_table[k] for k in res_hydrophobic_ph2_table.keys()])

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}
res_hydrophobic_ph7_table['X'] = np.average([res_hydrophobic_ph7_table[k] for k in res_hydrophobic_ph7_table.keys()])

# nomarlize the residue feature
res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


# one ont encoding
def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


# one ont encoding with unknown symbol
def one_hot_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def seq_feature(seq):      
   
    residue_feature = []
    for residue in seq:
        # replace some rare residue with 'X'
        if residue not in pro_res_table:
            residue = 'X'
        res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                         1 if residue in pro_res_polar_neutral_table else 0,
                         1 if residue in pro_res_acidic_charged_table else 0,
                         1 if residue in pro_res_basic_charged_table else 0]
        res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue],
                         res_pkx_table[residue],
                         res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
        residue_feature.append(res_property1 + res_property2)

    pro_hot = np.zeros((len(seq), len(pro_res_table)))
    pro_property = np.zeros((len(seq), 12))   
    for i in range(len(seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_hot_encoding_unk(seq[i], pro_res_table)
        pro_property[i,] = residue_feature[i]

    seq_feature = np.concatenate((pro_hot, pro_property), axis=1)
    return seq_feature


def seq_feature_no_onehot(seq):    
   
    residue_feature = []
    for residue in seq:
        # replace some rare residue with 'X'
        if residue not in pro_res_table:
            residue = 'X'
        res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                         1 if residue in pro_res_polar_neutral_table else 0,
                         1 if residue in pro_res_acidic_charged_table else 0,
                         1 if residue in pro_res_basic_charged_table else 0]
        res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue],
                         res_pkx_table[residue],
                         res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
        residue_feature.append(res_property1 + res_property2)

    #pro_hot = np.zeros((len(seq), len(pro_res_table)))
    pro_property = np.zeros((len(seq), 12))     
    for i in range(len(seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        #pro_hot[i,] = one_hot_encoding_unk(seq[i], pro_res_table)
        pro_property[i,] = residue_feature[i]

    #seq_feature = np.concatenate((pro_hot, pro_property), axis=1)
    seq_feature = np.array(pro_property)
    return seq_feature

# hla/Peptide sequence to target graph
def sequence_to_graph(target_key, target_sequence, distance_dir):
    target_edge_index = []
    target_edge_distance = []
    target_size = len(target_sequence)-24
    # print('***',(os.path.abspath(os.path.join(distance_dir, target_key + '.npy'))))
    contact_map_file = os.path.join(distance_dir, str(target_key) + '.npy')
    distance_map = np.load(contact_map_file)
    # the neighbor residue should have a edge
    # add self loop
    for i in range(target_size):
        distance_map[i, i] = 1
        if i + 1 < target_size:
            distance_map[i, i + 1] = 1    
    # print(distance_map)
    index_row, index_col = np.where(distance_map >= 0.5)  # for threshold     #Get the indices of all contact nodes (with edges)
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])  # dege
        target_edge_distance.append(distance_map[i, j])  # edge weight
    target_feature = seq_feature_no_onehot(target_sequence[24:])
    return target_size, target_feature, target_edge_index, target_edge_distance

# hla/Peptide sequence to target graph
def batch_seq_feature(seq_list,max_len,emb_len):
    # Assume all input tables and one_hot_encoding_unk function are defined outside this function
    batch_features = []

    for seq in seq_list:
        padding_length = max_len - len(seq)
        residue_feature = []
        padding_feature=[]
        for residue in seq:
            if residue not in pro_res_table:
                residue = 'X'
            res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 
                             1 if residue in pro_res_aromatic_table else 0,
                             1 if residue in pro_res_polar_neutral_table else 0,
                             1 if residue in pro_res_acidic_charged_table else 0,
                             1 if residue in pro_res_basic_charged_table else 0]
            res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue],
                             res_pkx_table[residue], res_pl_table[residue], 
                             res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
            residue_feature.append(res_property1 + res_property2)
            
            

        #pro_hot = np.array([one_hot_encoding_unk(res, pro_res_table) for res in seq])
        pro_property = np.array(residue_feature)
        seq_feature=[]
        #seq_feature = np.concatenate((pro_hot, pro_property), axis=1)
        if padding_length!=0:
            for _ in range(padding_length):
                padding_feature.append([0] *emb_len)
                
            padding_feature=np.array(padding_feature)
            seq_feature=np.concatenate((pro_property,padding_feature),axis=0)
                
            batch_features.append(seq_feature)
        else:
            seq_feature=pro_property
            batch_features.append(seq_feature)
            
        
        

    return batch_features


def batch_seq_feature_Bi(seq_list,max_len,emb_len):   #Bidirectional encoding
    # Assume all input tables and one_hot_encoding_unk function are defined outside this function
    batch_features = []

    for seq in seq_list:
        #padding_length = max_len - len(seq)
        residue_feature = []
        #padding_feature=[]
        for residue_index in range(15):
            if residue_index < len(seq):
                residue=seq[residue_index]
                if residue not in pro_res_table:
                    residue = 'X'
                res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 
                                1 if residue in pro_res_aromatic_table else 0,
                                1 if residue in pro_res_polar_neutral_table else 0,
                                1 if residue in pro_res_acidic_charged_table else 0,
                                1 if residue in pro_res_basic_charged_table else 0]
                res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue],
                                res_pkx_table[residue], res_pl_table[residue], 
                                res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
                #residue_feature.append(res_property1 + res_property2)
            
                #pro_hot=one_hot_encoding_unk(residue, pro_res_table)
                reaidue_propert=res_property1 + res_property2
                residue_feature.append(reaidue_propert)
            else:
                residue_feature.append(np.zeros(12).tolist())
        
        for reaidue_index in range(15):
            if 15 - residue_index > len(seq):
                residue_feature.append(np.zeros(12).tolist())
            else:
                residue=seq[len(seq) - 15 + residue_index]
                if residue not in pro_res_table:
                    residue = 'X'
                res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 
                                1 if residue in pro_res_aromatic_table else 0,
                                1 if residue in pro_res_polar_neutral_table else 0,
                                1 if residue in pro_res_acidic_charged_table else 0,
                                1 if residue in pro_res_basic_charged_table else 0]
                res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue],
                                res_pkx_table[residue], res_pl_table[residue], 
                                res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
                #residue_feature.append(res_property1 + res_property2)
            
                #pro_hot=one_hot_encoding_unk(residue, pro_res_table)
                reaidue_propert=res_property1 + res_property2
                residue_feature.append(reaidue_propert)
         
        batch_features.append(residue_feature)

    return batch_features

def batch_seq_feature_topo(seq_list):
    # Assume all input tables and one_hot_encoding_unk function are defined outside this function
    batch_features = []

    for seq in seq_list:
        padding_length = max_len - len(seq)
        residue_feature = []
        padding_feature=[]
        for residue in seq:
            if residue not in pro_res_table:
                residue = 'X'
            res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 
                             1 if residue in pro_res_aromatic_table else 0,
                             1 if residue in pro_res_polar_neutral_table else 0,
                             1 if residue in pro_res_acidic_charged_table else 0,
                             1 if residue in pro_res_basic_charged_table else 0]
            res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue],
                             res_pkx_table[residue], res_pl_table[residue], 
                             res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
            residue_feature.append(res_property1 + res_property2)
            
            

        pro_hot = np.array([one_hot_encoding_unk(res, pro_res_table) for res in seq])
        pro_property = np.array(residue_feature)

        seq_feature = np.concatenate((pro_hot, pro_property), axis=1)
        
        for _ in range(padding_length):
            padding_feature.append([0] *emb_len)
            
        padding_feature=np.array(padding_feature)
        seq_feature=np.concatenate((seq_feature,padding_feature),axis=0)
            
        batch_features.append(seq_feature)

    return batch_features
