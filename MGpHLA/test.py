'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-08-20 17:08:22
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-08-26 17:34:02
FilePath: \研究生学习档案\项目代码\HLAB\base_experminent.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import pandas as pd
import argparse
import json
import re
from collections import OrderedDict
import pickle
from Loader import *
from model.main_model import MGHLA
from model.ablation_models import *
from train_test import *
import  random
from tqdm import tqdm
from data_transform import *
from model.gvp_gnn import StructureEncoder
from feature_extraction import *
from Loader import test_data_div
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.00001
# 定义必要的参数
struc_hid_dim = 16
node_in_dim = (6, 3)  # node dimensions in input graph, should be (6, 3) if using original features
node_h_dim = (struc_hid_dim, 16)  # node dimensions to use in GVP-GNN layers
edge_in_dim = (32, 1)  # edge dimensions in input graph, should be (32, 1) if using original features
edge_h_dim = (32, 1)  # edge dimensions to embed to before use in GVP-GNN layers
struc_encoder_layer_num = 2
struc_dropout = 0.1
pep_max_len = 15
max_pro_seq_len = 348

def test(dataset_file,model_file):
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Test！')
    
    test_data=test_data_div(dataset_file)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)
    threshold=0.5
    model_eval = MGHLA(
                struc_hid_dim=struc_hid_dim,
                node_in_dim=node_in_dim,
                node_h_dim=node_h_dim,
                edge_in_dim=edge_in_dim,
                edge_h_dim=edge_h_dim,
                struc_encoder_layer_num=struc_encoder_layer_num,
                struc_dropout=struc_dropout,
                pep_max_len=pep_max_len,
                num_features_hla=32,
                num_features_pep=32,
                output_dim=128,
                hidden_channels=64,
                n_output=1,
                dropout=0.1
            )
    model_eval.to(device)
    model_eval.load_state_dict(torch.load(model_file, map_location='cpu'), strict = False)#加载一些训练好的模型参数,其实相当于参数初始化，比直接初始化为0效果好
    model_eval.eval()#加载模型
    
    ys,metrics=eval_step(model_eval,test_loader,float(threshold),True)
    return ys
   
def transfer(y_prob, threshold = 0.5):
    return np.array([[0, 1][x > threshold] for x in (y_prob)]) 
    
if __name__ == '__main__':
   
    file_Texternal='../data/fold_data/fold_data_new2/T_external.csv'
    file_iedb1424='../data/fold_data/fold_data_new2/iedb1424_new_remove_repeat.csv'
    file_independent='../data/fold_data/fold_data_new2/independent_1.csv'
    file_ann='../data/ideb_subset_new/ann_subset.csv'
    file_smmpmbec='../data/ideb_subset_new/smmpmbec_subset.csv'
    file_smm_Texternal='../data/T_external_subset_new/smm_subset.csv'
    file_ann_independent='../data/Independent_subset_new/ann_subset.csv'
    file_concensus_independent='../data/Independent_subset_new/consensus_subset.csv'
    file_pickpocket_independent='../data/Independent_subset_new/pickpocket_subset.csv'
    file_smm_independent='../data/Independent_subset_new/smm_subset.csv'
    file_smmpmbec_independent='../data/Independent_subset_new/smmpmbec_subset.csv'
    file_Anthemiedb1424='../data/ideb_subset_new/Anthem_sub.txt'
    file_AnthemTexternal='../data/T_external_subset_new/Anthem_subset.txt'
    file_AnthemIndependent='/home1/layomi/项目代码/MMGHLA_CT_blousm_weight/results/test_set_metrics/all_independent1_owndata_0.5.txt'
    file_ACMEiedb1424='../data/ideb_subset_new/iedb_result_new_last_2_ACME_yuan.txt'
    file_ACMEtexternal='../data/T_external_subset_new/Texternal_result_new_last_2_ACME_yuan.txt'
    file_ACMEindependent='../data/Independent_subset_new/Independent_result_new_last_2_ACME_yuan.txt'
    file_ANN_iedb='../data/ideb_subset_new/ann_subset.csv'
    #file_list=[file_Texternal,file_iedb1424,file_independent]
    #file_list=[file_Anthemiedb1424,file_AnthemTexternal,file_AnthemIndependent]
    #file_list=[file_ann_independent,file_concensus_independent,file_pickpocket_independent,file_smm_independent,file_smmpmbec_independent]
    #file_list=[file_ann,file_smmpmbec]
    #file_list=[file_smm_Texternal]
    #file_list=[file_ACMEiedb1424,file_ACMEtexternal,file_ACMEindependent]
    for dataset_file in file_list:
        
        ys_prob_list=[]
        model_folder='../models/fold_best'
        for file in os.listdir(model_folder):
            model_file_path=os.path.join(model_folder,file)
            (ys_true,ys_preb,ys_prob)=test(dataset_file,model_file_path)
            ys_prob_list.append(np.array(ys_prob))
            
        y_prob_mean = [np.mean(scores) for scores in zip(*ys_prob_list)]     
        y_preb_list=transfer(y_prob_mean, threshold = 0.5)
        y_preb_list=[int(d) for d in y_preb_list]    
            
        #为了便于不同长度和超型下性能的统计，重读文件且将预测结果添加到文件中并保存
        all_data_list=pd.read_csv(dataset_file)
        all_data_list=all_data_list.values.tolist()
        
        if len(all_data_list)!=len(y_prob_mean):
            print('Error!')
        y_true_list=[]
        for i in range(len(all_data_list)):
            all_data_list[i].extend([y_prob_mean[i],all_data_list[i][2],y_preb_list[i]]) #预测分数，真实类别，预测类别
            y_true_list.append(int(all_data_list[i][2]))
            
        result_folder='../results/data_result_file/{}'.format(dataset_file.split('/')[-2])  
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        result_dataset_write_file=os.path.join(result_folder,'{}'.format(dataset_file.split('/')[-1]))
        with open(result_dataset_write_file,'w') as f:
            for sublist in all_data_list:
                line = ','.join(map(str, sublist)) + '\n'
                f.write(line)
        print('结果数据存储完成') 
        
        
        metrics_set = performances(y_true_list, y_preb_list, y_prob_mean, print_ = True)
        metrics_dict=dict()
        metrics_wfile1=os.path.join(result_folder,'metrics.txt')
        metrics_dict['{}'.format(file.split("/")[-1])]=metrics_set
        recording_w(metrics_wfile1, metrics_dict,'a+')
      
    
#将结果进行追加记录到file1
def recording_w(file1,record,w_or_a='w'):
    if isinstance(record,dict):
        #print('True')

        with open (file1,w_or_a) as f:
            for key,value in record.items():
                print('{}:{}'.format(key,value),file=f)
        f.close()
