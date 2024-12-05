import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import shutil  
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import numpy as np
import pandas as pd
import argparse
import json
import re
import random

# 设置随机种子
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 导入自定义模块
from Loader import train_predict_div
from model.main_model import MGHLA
from model.gvp_gnn import StructureEncoder
from data_transform import transform_data, CATHDataset
from pytorchtools import EarlyStopping
from train_test import collate, train, predicting, eval_step
from transformers import AdamW

if __name__ == '__main__':
    HPIdatasets = ['hpi']
    ratio_list = [1, 3, 5]
    ratio = 1
    TRAIN_BATCH_SIZE = 512
    TEST_BATCH_SIZE = 512
    LR = 1e-3
    NUM_EPOCHS = 18

    print('dataset:', HPIdatasets)
    print('ratio', ratio)
    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)
    
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

    # 简化设备选择，仅支持单GPU或CPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    models_dir = '../models/'
    results_dir = '../results/'
    metrics_log = ['seed', 'epoch', 'type','fold' ,'auc','acc' ,'mcc', 'f1', 'sensitivity', 'specificity',
                  'precision', 'recall', 'aupr', 'metrics_ep_avg', 'fold_metric_best',
                  'fold_ep_best', 'fold_best', 'metric_best', 'ep_best']

    # 创建必要的目录
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, HPIdatasets[0]), exist_ok=True)

    # 创建 epoch 目录
    epoch_save_dir = os.path.join(models_dir, 'MGHLA_epoch')
    os.makedirs(epoch_save_dir, exist_ok=True)

    save_file = os.path.join(results_dir, 'epoch_result_{:.0e}_{}'.format(LR, TRAIN_BATCH_SIZE))

    with open(save_file, 'w') as write_f:
        header = '\t\t'.join(metrics_log) + '\n'
        write_f.write(header)

    cath = CATHDataset(os.path.join('../data/aphlafold2', 'structure.jsonl'))
    dataset_structure = transform_data(cath.data, max_pro_seq_len)

    fold_best, metric_best, ep_best = 0, 0, -1
    scores = []  # 存储每次训练的结果,因为采用的是K折交叉验证，方便最后取均值
    seeds = [0]

    for seed in seeds:
        for fold in range(5):
            train_file = '../data/fold_data/fold_data_new2/train_fold{}.csv'.format(fold)
            valid_file = '../data/fold_data/fold_data_new2/val_fold{}.csv'.format(fold)
            test_file = '../data/fold_data/fold_data_new2/independent_1.csv'

            train_data, dev_data = train_predict_div(train_file, valid_file, dataset_structure, seed)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                                       collate_fn=collate)
            dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                                     collate_fn=collate)

            # 实例化模型
            model = MGHLA(
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
            model.to(device)

            loss_fn = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=LR)  # 使用定义的 LR
            fold_metric_best, fold_ep_best = 0, -1
            early_stopping = EarlyStopping(patience=5, verbose=True)

            # 清空 CUDA 缓存
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            for epoch in range(NUM_EPOCHS):
                ys_train, loss_train_list, metrics_train, time_train_ep = train(
                    fold, model, device, train_loader, optimizer, epoch + 1, NUM_EPOCHS, loss_fn, TRAIN_BATCH_SIZE
                )
                print('Predicting for validation data...')
                ys_val, loss_val_list, metrics_val = predicting(
                    fold, model, device, dev_loader, epoch, NUM_EPOCHS, loss_fn
                )

                save_dir = models_dir

                # 计算平均指标（假设 metrics_val 前四个是需要平均的）
                if len(metrics_val) >= 4:
                    metrics_ep_avg = sum(metrics_val[:4]) / 4
                else:
                    metrics_ep_avg = 0  # 或者其他适当的默认值
                if metrics_ep_avg > fold_metric_best:
                    fold_metric_best, fold_ep_best = metrics_ep_avg, epoch
                    os.makedirs(save_dir, exist_ok=True)

                # 更新全局最佳指标
                if metric_best < fold_metric_best:
                    fold_best, metric_best, ep_best = fold, fold_metric_best, epoch

                metric_train_list = list(metrics_train)
                metric_val_list = list(metrics_val)

                # 写入训练日志
                with open(save_file, 'a') as write_f:
                    # 训练日志
                    train_log = '\t\t'.join([
                        str(seed), str(epoch), 'train', str(fold)
                    ] + [str(m) for m in metric_train_list] + [
                        str(metrics_ep_avg), str(fold_metric_best), str(fold_ep_best),
                        str(fold_best), str(metric_best), str(ep_best)
                    ]) + '\r\n'
                    write_f.write(train_log)

                    # 验证日志
                    predict_log = '\t\t'.join([
                        str(seed), str(epoch), 'predict', str(fold)
                    ] + [str(m) for m in metric_val_list] + [
                        str(metrics_ep_avg), str(metric_best), str(ep_best),
                        str(fold_best)
                    ]) + '\r\n'
                    write_f.write(predict_log)
                    
                # 保存模型
                path_saver = os.path.join(epoch_save_dir, 'model_fold{}_epoch{}.pkl'.format(fold, epoch))
                print('*****Path saver: ', path_saver)
                torch.save(model.state_dict(), path_saver)

                # F1 作为早停条件
                if epoch > 10:
                    if len(metrics_val) > 3:
                        F1 = metrics_val[3]  
                        early_stopping(F1, model)
                        if early_stopping.counter == 0:  # 如果早停机制计数器为0，即没有提升
                            best_test_score = fold_metric_best  # 最好的测试分数为当前测试分数
                        if early_stopping.early_stop or epoch == NUM_EPOCHS - 1:  # 如果早停机制停止或者达到最大 epoch
                            scores.append(fold_metric_best)  # 将最好的测试分数加入到 scores 中
                            break

                

            # 保存每个fold的最佳模型
            best_model_path = os.path.join(epoch_save_dir, 'model_fold{}_epoch{}.pkl'.format(fold, ep_best))
            fold_best_folder = os.path.join(models_dir, 'fold_best')
            os.makedirs(fold_best_folder, exist_ok=True)

            if os.path.exists(best_model_path):
                shutil.copy(best_model_path, fold_best_folder)
                print(f"文件 {best_model_path} 已成功复制到 {fold_best_folder}")
            else:
                print(f"文件 {best_model_path} 不存在")

            # 释放资源
            del train_data
            del dev_data
            del train_loader
            del dev_loader




    
