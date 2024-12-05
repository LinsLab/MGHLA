import math
from sklearn import metrics
from sklearn import preprocessing
import numpy as np
import pandas as pd
import re
import time
import datetime
import random
random.seed(0)
from scipy import interp
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm, trange

from collections import Counter
from collections import OrderedDict
from functools import reduce
from tqdm import tqdm, trange
from copy import deepcopy

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import difflib

#plt.rc('font',family='Times New Roman')
#plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

pep_max_len = 15 # peptide; enc_input max sequence length
hla_max_len = 34 # hla; dec_input(=dec_output) max sequence length
tgt_len = pep_max_len*2 + hla_max_len

#vocab = np.load('E:/研究生学习档案/项目代码/TransPHLA-AOMP-master/TransPHLA-AOMP-master/TransPHLA-AOMP/vocab_dict.npy', allow_pickle = True).item()
vocab = np.load('/home1/layomi/项目代码/MMGHLA_CT_blousm/MGpHLA/model/vocab_dict.npy', allow_pickle = True).item()

vocab_size = len(vocab)

# Transformer Parameters
#d_model = 64  # Embedding Size
d_model = 32  # Embedding Size
d_model_phla=64
d_ff = 512 # FeedForward dimension
#d_k = d_v = 64  # dimension of K(=Q), V
d_k = d_v = 64  # dimension of K(=Q), V

#batch_size = 1024

n_layers, n_heads, fold = 1, 9, 4
#n_layers, n_heads, fold = 2, 8, 4

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")


class PositionalEncoding(nn.Module):
    #位置嵌入类
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)#d_model是设定好的embeding_size
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)#向量相乘
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)#transpose()函数的作用就是调换数组的行列值的索引值，类似于求矩阵的转置
        self.register_buffer('pe', pe)#register_buffer函数功能为注册，注册pe名为pe

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]  #这里是将前面嵌入得到的结果与位置嵌入叠加,这里使用了三维+二维
        return self.dropout(x)
    
    
class PositionalEncoding_decoder(nn.Module):
    #位置嵌入类
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding_decoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model*2)#d_model是设定好的embeding_size
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        
        div_term = torch.exp(torch.arange(0, d_model*2, 2).float() * (-math.log(10000.0) / d_model*2))
        pe[:, 0::2] = torch.sin(position * div_term)#向量相乘
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)#transpose()函数的作用就是调换数组的行列值的索引值，类似于求矩阵的转置
        self.register_buffer('pe', pe)#register_buffer函数功能为注册，注册pe名为pe

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]  #这里是将前面嵌入得到的结果与位置嵌入叠加,这里使用了三维+二维
        return self.dropout(x)
    
class PositionalEncoding_scale(nn.Module):
    #位置嵌入类
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding_scale, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)#d_model是设定好的embeding_size
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        scale_factor = 0.5  # 或者更大的值，根据实验调整
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0 * scale_factor) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)#向量相乘
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)#transpose()函数的作用就是调换数组的行列值的索引值，类似于求矩阵的转置
        self.register_buffer('pe', pe)#register_buffer函数功能为注册，注册pe名为pe

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]  #这里是将前面嵌入得到的结果与位置嵌入叠加,这里使用了三维+二维
        return self.dropout(x)
    
def get_attn_pad_mask(seq_q, seq_k):
    '''
    padding 部分的注意力
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    ## 通过形状，获取batch大小和序列长度
    batch_size, len_q = seq_q.size()  # Q
    batch_size, len_k = seq_k.size()  #K 
    #eq(0)表示的是Padding的部分，因为我们字典中把0作为了Padding
    # eq(zero) is PAD token eq函数是留下seq_k等于0的坐标
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    #把形状扩充到(batch_size, len_q, len_k)
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


import torch

def get_peptide_pad_mask_new(pep_lengths):
    """
    创建适用于注意力机制的掩码，形状为 [batch_size, seq_len, seq_len]。

    参数:
    pep_lengths (torch.Tensor): 每个样本的有效长度，形状为 [batch_size]

    返回:
    torch.Tensor: 掩码张量，形状为 [batch_size, seq_len, seq_len]
                  填充位置为 True，有效数据位置为 False
    """
    batch_size = pep_lengths.size(0)
    max_len = pep_lengths.max().item()  # 最大序列长度

    # 创建一个从0到max_len-1的序列，并扩展到整个批次
    range_tensor = torch.arange(max_len).expand(batch_size, max_len).to(pep_lengths.device)

    # 生成二维掩码 [batch_size, seq_len]
    mask = range_tensor >= pep_lengths.unsqueeze(1)  # True 在需要屏蔽的地方

    # 复制这个掩码到第三维，以形成 [batch_size, seq_len, seq_len] 形状
    mask = mask.unsqueeze(1).expand(batch_size, max_len, max_len)

    return mask



class ScaledDotProductAttention(nn.Module):
    """
        带缩放的点积注意力
        返回值：各个key的注意力及注意力与key与V的矩阵乘积
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]，除以 np.sqrt(d_k)是防止乘积结果过大
        # 这里用到了attn_mask，即attn_mask的地方分数，我们是不要的，这里设置了一个很小的值,-1e9表示 -1000000000.0
        # 维度不变, 还是[1, 8, 5, 5]，这里是padding的位置，即最后一个位置时很小的值
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        # 计算attention值，softmax，在最后一个维度计算, attn的维度torch.Size([1, 8, 5, 5]) ，即每个单词位置的注意力，也可以说成关注点
        attn = nn.Softmax(dim=-1)(scores)
        # 和V矩阵相乘 context 形状 torch.Size([1, 8, 5, 64]), 即对V来说，哪些位置是值得注意的，意思是对于输入句子来说，每个单词相对于整个句子来说的重要性
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        #print('context的size',context.size())
        return context, attn
    
class MultiHeadAttention(nn.Module):
    """
        attention 结构包含多头注意力+FeedForward（前馈层）
        attention 结构中的 Attention 组件
        
        返回值：多头注意力归一化后的结果和注意力分数
        
     """
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.use_cuda = use_cuda
        #初始化QKV
        device = torch.device("cuda" if self.use_cuda else "cpu")
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False).to(device)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False).to(device)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False).to(device)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False).to(device)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        # residual 代表的是论文中的残差的意思，即residual+attention后的输出作为下一个阶段的输入
        # residual残差来源于ResNet架构，目的是可以训练更深的网络，防止模型退化
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        #print('Q的size',Q.size())
        #print('K的size',K.size())
        #print('V的size',K.size())
        # 因为每个头都是一样的Mask计算，就是某些部分不计算注意力,掩膜
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]
         
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        #带缩放的点积注意力， 得到最后的注意力，和经过注意力计算的V的到的context
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        #  context.transpose(1, 2)得到的维度, torch.Size([1, 5, 8, 64])
        # context的形状变成从 例子：torch.Size([1, 5, 9, 64])--> torch.Size([1, 5, 512]), 即，合并了nheads, 就是合并了9个头
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        #NLP中常用的归一化操作：LayerNorm,对每个单词的embedding进行归一化
        return nn.LayerNorm(d_model).to(device)(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    """
        特征优化块，螺旋通道，先升后降
    """
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.use_cuda = use_cuda
        device = torch.device("cuda" if self.use_cuda else "cpu")
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(0.1)
        ).to(device)
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output + residual) # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention().to(device)
        self.pos_ffn = PoswiseFeedForwardNet().to(device)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model).to(device)
         #初始化Embedding， (5,512)， 注意vocab_size是输入单词表的维度，输出是我们embedding的维度，这里是随机的初始化向量，这个向量可以被训练
        self.pos_emb = PositionalEncoding(d_model).to(device)
         #一个序列的位置嵌入的 Embedding初始化 
         #nn.Sequential内部实现了forward函数，因此可以不用写forward函数。而nn.ModuleList则没有实现内部forward函数，但是他们都是容器
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)]).to(device)#n_layer为1

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        #print('embeding结果大小： ',enc_outputs.size())
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        #print('位置嵌入结果大小： ',enc_outputs.size())
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len] 得到padding部分的注意力
        #print('掩膜结果大小：',enc_self_attn_mask.size())
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        #print('编码器部分输出的size',enc_outputs.size())
        return enc_outputs, enc_self_attns
    
 
class pep_Encoder(nn.Module):
    def __init__(self):
        super(pep_Encoder, self).__init__()

        self.use_cuda = use_cuda
        device = torch.device("cuda" if self.use_cuda else "cpu")
        #self.pos_emb = PositionalEncoding(d_model)
        self.line=nn.Linear(32,64)
        self.pos_emb = PositionalEncoding(64).to(device)
         #一个序列的位置嵌入的 Embedding初始化 
         #nn.Sequential内部实现了forward函数，因此可以不用写forward函数。而nn.ModuleList则没有实现内部forward函数，但是他们都是容器
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)]).to(device)#n_layer为1

    def forward(self,pep_features,pep_lens):
        
        #enc_inputs: [batch_size, src_len]
        enc_outputs=self.line(pep_features.float())
        #enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        #print('embeding结果大小： ',enc_outputs.size())
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        enc_outputs=enc_outputs*3
        #print('位置嵌入结果大小： ',enc_outputs.size())
        enc_self_attn_mask = get_peptide_pad_mask_new(pep_lens) # [batch_size, src_len, src_len] 得到padding部分的注意力
        #print('掩膜结果大小：',enc_self_attn_mask.size())
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        #print('编码器部分输出的size',enc_outputs.size())
        return enc_outputs, enc_self_attns
    
class pep_Encoder_4(nn.Module):    #d_model=32,不进行维度的填充
    def __init__(self):
        super(pep_Encoder_4, self).__init__()

        self.use_cuda = use_cuda
        device = torch.device("cuda" if self.use_cuda else "cpu")
        #self.pos_emb = PositionalEncoding(d_model)
        self.line=nn.Linear(21,32)
        self.pos_emb = PositionalEncoding(32).to(device)
         #一个序列的位置嵌入的 Embedding初始化 
         #nn.Sequential内部实现了forward函数，因此可以不用写forward函数。而nn.ModuleList则没有实现内部forward函数，但是他们都是容器
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)]).to(device)#n_layer为1

    def forward(self,pep_features,pep_lens):
        
        #enc_inputs: [batch_size, src_len]
        enc_outputs=pep_features.float()
        enc_outputs=self.line(pep_features.float())
        #enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        #print('embeding结果大小： ',enc_outputs.size())
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        enc_outputs=enc_outputs*3
        #print('位置嵌入结果大小： ',enc_outputs.size())
        enc_self_attn_mask = get_peptide_pad_mask_new(pep_lens) # [batch_size, src_len, src_len] 得到padding部分的注意力
        #print('掩膜结果大小：',enc_self_attn_mask.size())
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        #print('编码器部分输出的size',enc_outputs.size())
        return enc_outputs, enc_self_attns
class pep_Encoder_3(nn.Module):    #d_model=32,不进行维度的填充
    def __init__(self):
        super(pep_Encoder_3, self).__init__()

        self.use_cuda = use_cuda
        device = torch.device("cuda" if self.use_cuda else "cpu")
        #self.pos_emb = PositionalEncoding(d_model)
        #self.line=nn.Linear(32,64)
        self.pos_emb = PositionalEncoding(32).to(device)
         #一个序列的位置嵌入的 Embedding初始化 
         #nn.Sequential内部实现了forward函数，因此可以不用写forward函数。而nn.ModuleList则没有实现内部forward函数，但是他们都是容器
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)]).to(device)#n_layer为1

    def forward(self,pep_features,pep_lens):
        
        #enc_inputs: [batch_size, src_len]
        enc_outputs=pep_features.float()
        #enc_outputs=self.line(pep_features.float())
        #enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        #print('embeding结果大小： ',enc_outputs.size())
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        enc_outputs=enc_outputs*3
        #print('位置嵌入结果大小： ',enc_outputs.size())
        enc_self_attn_mask = get_peptide_pad_mask_new(pep_lens) # [batch_size, src_len, src_len] 得到padding部分的注意力
        #print('掩膜结果大小：',enc_self_attn_mask.size())
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        #print('编码器部分输出的size',enc_outputs.size())
        return enc_outputs, enc_self_attns
    
class pep_Encoder_scale(nn.Module):
    def __init__(self):
        super(pep_Encoder_scale, self).__init__()

        self.use_cuda = use_cuda
        device = torch.device("cuda" if self.use_cuda else "cpu")
        #self.pos_emb = PositionalEncoding(d_model)
        self.line=nn.Linear(32,64)
        self.pos_emb = PositionalEncoding_scale(64).to(device)
         #一个序列的位置嵌入的 Embedding初始化 
         #nn.Sequential内部实现了forward函数，因此可以不用写forward函数。而nn.ModuleList则没有实现内部forward函数，但是他们都是容器
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)]).to(device)#n_layer为1

    def forward(self,pep_features,pep_lens):
        
        #enc_inputs: [batch_size, src_len]
        enc_outputs=self.line(pep_features.float())
        #enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        #print('embeding结果大小： ',enc_outputs.size())
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        enc_outputs=enc_outputs*3
        #print('位置嵌入结果大小： ',enc_outputs.size())
        enc_self_attn_mask = get_peptide_pad_mask_new(pep_lens) # [batch_size, src_len, src_len] 得到padding部分的注意力
        #print('掩膜结果大小：',enc_self_attn_mask.size())
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        #print('编码器部分输出的size',enc_outputs.size())
        return enc_outputs, enc_self_attns    

class pep_Encoder_2(nn.Module):
    def __init__(self):
        super(pep_Encoder_2, self).__init__()

        self.use_cuda = use_cuda
        device = torch.device("cuda" if self.use_cuda else "cpu")
        #self.pos_emb = PositionalEncoding(d_model)
        self.line=nn.Linear(32,64)
        self.pos_emb = PositionalEncoding(64).to(device)
         #一个序列的位置嵌入的 Embedding初始化 
         #nn.Sequential内部实现了forward函数，因此可以不用写forward函数。而nn.ModuleList则没有实现内部forward函数，但是他们都是容器
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)]).to(device)#n_layer为1

    def forward(self,pep_features,pep_lens):
        
        #enc_inputs: [batch_size, src_len]
        enc_outputs=self.line(pep_features.float())
        #enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        #print('embeding结果大小： ',enc_outputs.size())
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        enc_outputs=enc_outputs*3
        enc_outputs=important_pep_index(enc_outputs, pep_lens)
        #print('位置嵌入结果大小： ',enc_outputs.size())
        enc_self_attn_mask = get_peptide_pad_mask_new(pep_lens) # [batch_size, src_len, src_len] 得到padding部分的注意力
        #print('掩膜结果大小：',enc_self_attn_mask.size())
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        #print('编码器部分输出的size',enc_outputs.size())
        return enc_outputs, enc_self_attns
 
 
 
class phla_Encoder(nn.Module):
    def __init__(self):
        super(pep_Encoder, self).__init__()

        self.use_cuda = use_cuda
        device = torch.device("cuda" if self.use_cuda else "cpu")
        #self.pos_emb = PositionalEncoding(d_model)
        #self.line=nn.Linear(33,64)
        self.pos_emb = PositionalEncoding(128).to(device)
         #一个序列的位置嵌入的 Embedding初始化 
         #nn.Sequential内部实现了forward函数，因此可以不用写forward函数。而nn.ModuleList则没有实现内部forward函数，但是他们都是容器
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)]).to(device)#n_layer为1

    def forward(self,pha_features,hla_lens,pep_lens):
        
        #enc_inputs: [batch_size, src_len]
        enc_outputs=self.line(pep_features.float())
        #enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        #print('embeding结果大小： ',enc_outputs.size())
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        #print('位置嵌入结果大小： ',enc_outputs.size())
        enc_self_attn_mask = get_phla_pad_mask_new(hla_lens,pep_lens) # [batch_size, src_len, src_len] 得到padding部分的注意力
        #print('掩膜结果大小：',enc_self_attn_mask.size())
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        #print('编码器部分输出的size',enc_outputs.size())
        return enc_outputs, enc_self_attns
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, dec_self_attn_mask): # dec_inputs = enc_outputs
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs) # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.use_cuda = use_cuda
        device = torch.device("cuda" if self.use_cuda else "cpu")
        #self.pos_emb = PositionalEncoding(d_model)
        self.line=nn.Linear(33,64)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
        self.tgt_len = tgt_len
        
    def forward(self, dec_inputs): # dec_inputs = enc_outputs (batch_size, peptide_hla_maxlen_sum, d_model)
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
#         dec_outputs = self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        dec_inputs=self.line(dec_inputs)
        dec_outputs = self.pos_emb(dec_inputs.transpose(0, 1)).transpose(0, 1).to(device) # [batch_size, tgt_len, d_model]
        #dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda() # [batch_size, tgt_len, tgt_len]
        dec_self_attn_pad_mask = torch.LongTensor(np.zeros((dec_inputs.shape[0], 15, 15))).bool().to(device)

        dec_self_attns = []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_pad_mask)
            dec_self_attns.append(dec_self_attn)
            
        return dec_outputs, dec_self_attns
    
class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
#         self.tgt_emb = nn.Embedding(d_model * 2, d_model)
        self.use_cuda = use_cuda
        device = torch.device("cuda" if self.use_cuda else "cpu")
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
        self.tgt_len = tgt_len
        
    def forward(self, dec_inputs): # dec_inputs = enc_outputs (batch_size, peptide_hla_maxlen_sum, d_model)
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
#         dec_outputs = self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_inputs.transpose(0, 1)).transpose(0, 1).to(device) # [batch_size, tgt_len, d_model]
#         dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda() # [batch_size, tgt_len, tgt_len]
        dec_self_attn_pad_mask = torch.LongTensor(np.zeros((dec_inputs.shape[0], tgt_len, tgt_len))).bool().to(device)

        dec_self_attns = []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_pad_mask)
            dec_self_attns.append(dec_self_attn)
            
        return dec_outputs, dec_self_attns
    
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.use_cuda = use_cuda
        device = torch.device("cuda" if use_cuda else "cpu")
        self.pep_encoder = Encoder().to(device)
        self.hla_encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)
        self.tgt_len = tgt_len
        self.projection = nn.Sequential(
                                        nn.Linear(tgt_len * d_model, 256),
                                        nn.ReLU(True),

                                        nn.BatchNorm1d(256),
                                        nn.Linear(256, 64),
                                        nn.ReLU(True),

                                        #output layer
                                        nn.Linear(64, 2)
                                        ).to(device)
        
    def forward(self, pep_inputs, hla_inputs):
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        pep_enc_outputs, pep_enc_self_attns = self.pep_encoder(pep_inputs)
        hla_enc_outputs, hla_enc_self_attns = self.hla_encoder(hla_inputs)
        enc_outputs = torch.cat((pep_enc_outputs, hla_enc_outputs), 1) # concat pep & hla embedding
        #print('enc_outputs结合体的size',enc_outputs.size())
        
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns = self.decoder(enc_outputs)
        #print("",dec_outputs)
        #print('dec_outputs的size',dec_outputs)
        dec_outputs = dec_outputs.view(dec_outputs.shape[0], -1) # Flatten [batch_size, tgt_len * d_model]
        #print("\n\n\n解码器decoder\n",dec_outputs)
        dec_logits = self.projection(dec_outputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        #print('\n\n\nprojection快的输出\n',dec_logits)
        return dec_logits.view(-1, dec_logits.size(-1)), pep_enc_self_attns, hla_enc_self_attns, dec_self_attns
    

        
def transfer(y_prob, threshold = 0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])



def important_pep_index(pep_features, lengths):
    # 假设 pep_features 已经是一个 tensor 或在外部转换为 tensor
    # pep_features: shape [batch_size, seq_len, feature_dim], e.g., [batch_size, 30, 64]
    
    # 扩大权重
    scale_factor = 4/3

    # 批量处理所有peptides
    batch_size, seq_len, feature_dim = pep_features.shape
    for i in range(batch_size):
        # 计算特定的重要位置索引
        residue_indices = [0, 1, lengths[i]-1, 30-lengths[i], 30-lengths[i]+1, 29]
        valid_indices = [idx for idx in residue_indices if idx < seq_len]

        # 扩大这些位置的向量
        if valid_indices:
            pep_features[i, valid_indices] *= scale_factor

    # 检查形状是否正确
    if pep_features.shape != (batch_size, 30, 64):
        print('形状有误')

    return pep_features