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

# Set random seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Import custom modules
from Loader import train_predict_div
from model.ablation_models import MGHLA_mol,MGHLA_unstructure,MGHLA_classtopo,MGHLA_KAN,MGHLA_KAN_2,MGHLA_KAN_3
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
    NUM_EPOCHS = 25

    print('dataset:', HPIdatasets)
    print('ratio', ratio)
    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)
    
    # Define necessary parameters
    struc_hid_dim = 16
    node_in_dim = (6, 3)  # node dimensions in input graph, should be (6, 3) if using original features
    node_h_dim = (struc_hid_dim, 16)  # node dimensions to use in GVP-GNN layers
    edge_in_dim = (32, 1)  # edge dimensions in input graph, should be (32, 1) if using original features
    edge_h_dim = (32, 1)  # edge dimensions to embed to before use in GVP-GNN layers
    struc_encoder_layer_num = 2
    struc_dropout = 0.1
    pep_max_len = 15
    max_pro_seq_len = 348

    # Device selection, supports only single GPU or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    models_dir = '../models_ablation_main'
    results_dir = '../results_ablation_main'
    metrics_log = ['seed', 'epoch', 'type','fold' ,'auc','acc' ,'mcc', 'f1', 'sensitivity', 'specificity',
                  'precision', 'recall', 'aupr', 'metrics_ep_avg', 'fold_metric_best',
                  'fold_ep_best', 'fold_best', 'metric_best', 'ep_best']
    # Create necessary directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, HPIdatasets[0]), exist_ok=True)

    epoch_save_dir = os.path.join(models_dir, 'MGHLA_unstructure_epoch_4')
    os.makedirs(epoch_save_dir, exist_ok=True)

    save_file = os.path.join(results_dir, 'MGHLA_unstructure_epoch_result_{:.0e}_{}_4'.format(LR, TRAIN_BATCH_SIZE))

    with open(save_file, 'w') as write_f:
        header = '\t\t'.join(metrics_log) + '\n'
        write_f.write(header)

    cath = CATHDataset(os.path.join('../data/aphlafold2', 'structure.jsonl'))
    dataset_structure = transform_data(cath.data, max_pro_seq_len)

    fold_best, metric_best, ep_best = 0, 0, -1
    scores = []  
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

            # Instantiate the model
            model = MGHLA_unstructure(
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
            optimizer = optim.Adam(model.parameters(), lr=LR)  
            fold_metric_best, fold_ep_best = 0, -1
            early_stopping = EarlyStopping(patience=5, verbose=True)

            # Clear CUDA cache
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

                # Compute average metrics (average the first four)
                if len(metrics_val) >= 4:
                    metrics_ep_avg = sum(metrics_val[:4]) / 4
                else:
                    metrics_ep_avg = 0  
                if metrics_ep_avg > fold_metric_best:
                    fold_metric_best, fold_ep_best = metrics_ep_avg, epoch
                    os.makedirs(save_dir, exist_ok=True)

                # Update global best metrics
                if metric_best < fold_metric_best:
                    fold_best, metric_best, ep_best = fold, fold_metric_best, epoch

                metric_train_list = list(metrics_train)
                metric_val_list = list(metrics_val)

                # Write training logs
                with open(save_file, 'a') as write_f:
                    train_log = '\t\t'.join([
                        str(seed), str(epoch), 'train', str(fold)
                    ] + [str(m) for m in metric_train_list] + [
                        str(metrics_ep_avg), str(fold_metric_best), str(fold_ep_best),
                        str(fold_best), str(metric_best), str(ep_best)
                    ]) + '\r\n'
                    write_f.write(train_log)

                    predict_log = '\t\t'.join([
                        str(seed), str(epoch), 'predict', str(fold)
                    ] + [str(m) for m in metric_val_list] + [
                        str(metrics_ep_avg), str(metric_best), str(ep_best),
                        str(fold_best)
                    ]) + '\r\n'
                    write_f.write(predict_log)
                    
                # Save the model
                path_saver = os.path.join(epoch_save_dir, 'MGHLA_unstructure_model_fold{}_epoch{}.pkl'.format(fold, epoch))
                print('*****Path saver: ', path_saver)
                torch.save(model.state_dict(), path_saver)

                # Using the average as the early stopping condition
                if epoch > 10:
                    early_stopping(metrics_ep_avg, model)  
                    if early_stopping.counter == 0:  
                        best_test_score = fold_metric_best  
                    if early_stopping.early_stop or epoch == NUM_EPOCHS - 1:  # If the early stopping mechanism is triggered or the maximum epoch is reached
                        scores.append(fold_metric_best)  
                        break


                

            # Save the best model for each fold
            best_model_path = os.path.join(epoch_save_dir, 'MGHLA_unstructure_model_fold{}_epoch{}.pkl'.format(fold, ep_best))
            fold_best_folder = os.path.join(models_dir, 'fold_best')
            os.makedirs(fold_best_folder, exist_ok=True)

            if os.path.exists(best_model_path):
                shutil.copy(best_model_path, fold_best_folder)
                print(f"File {best_model_path} has been successfully copied to {fold_best_folder}")
            else:
                print(f"File {best_model_path} does not exist")

            # Release resources
            del train_data
            del dev_data
            del train_loader
            del dev_loader




    
