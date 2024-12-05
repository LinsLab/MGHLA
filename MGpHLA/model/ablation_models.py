import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
import random
import numpy as np
from model.gvp_gnn import StructureEncoder

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from torch_geometric.utils import dropout_adj
from model.pep_encoder import *
from torch_geometric.nn import GCNConv, GCN2Conv, GATConv, global_max_pool as gmp, global_add_pool as gap, \
    global_mean_pool as gep, global_sort_pool
from torch_scatter import scatter_mean
from model.kan import KAN
    


max_target=65
max_hla_len=372
batch_size=64
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
# Parameters of protein structure encoder
max_nodes = 348    #372-24
struc_hid_dim = 16
struc_encoder_layer_num = 2
node_in_dim = (6, 3)  # node dimensions in input graph, should be (6, 3) if using original features
node_h_dim = (struc_hid_dim, 16)  # node dimensions to use in GVP-GNN layers
edge_in_dim = (32, 1)  # edge dimensions in input graph, should be (32, 1) if using original features
edge_h_dim = (32, 1)  # edge dimensions to embed to before use in GVP-GNN layers
struc_dropout = 0.1
struc_dropout2=0.1




class Contact_GNN(torch.nn.Module):
    def __init__(self,num_features_hla=32,head_list=[1,2],hidden_channels=64,dropout=0.1):
        super(Contact_GNN,self).__init__()
        self.head_list=head_list
        self.dropout=dropout
        self.dim=hidden_channels
        self.relu = nn.ReLU().to(device)
        #Contact graph
        self.hla_conv1 = GCNConv(num_features_hla, num_features_hla, dropout).to(device)
        self.hla_conv2 = GATConv(num_features_hla, num_features_hla, self.head_list[0],self.dropout).to(device)
        self.hla_conv3 = GATConv(num_features_hla, num_features_hla,self.head_list[1], self.dropout).to(device)
        self.hla_fc=nn.Sequential(nn.Linear(num_features_hla * 2,self.dim*2),
                                  nn.ReLU(),
                                  nn.BatchNorm1d(self.dim*2),
                                  nn.Dropout(dropout),
                                  
                                  nn.Linear(self.dim*2,self.dim),
                                  nn.ReLU()                        
        ).to(device)
        
    def forward(self,data_hla_contact):
        hla_x, hla_edge_index, hla_edge_weight,hla_key, hla_batch = data_hla_contact.x, data_hla_contact.edge_index, data_hla_contact.edge_weight,data_hla_contact.hla_key, data_hla_contact.batch
        x_contact = self.hla_conv1(hla_x, hla_edge_index,hla_edge_weight)
        x_contact = self.relu(x_contact)
        x_contact = self.hla_conv2(x_contact, hla_edge_index)
        x_contact = self.relu(x_contact)
        x_contact = self.hla_conv3(x_contact, hla_edge_index)
        x_contact = self.relu(x_contact)
        x_contact = gep(x_contact, hla_batch)  # global pooling
        # flatten
        x_contact=self.hla_fc(x_contact)
        return x_contact
    
class classtopo_GNN(torch.nn.Module):
    def __init__(self,num_features_hla=32,hidden_channels=64,dropout=0.1):
        super(classtopo_GNN,self).__init__()
        self.fc=nn.Linear(32,112).to(device)
        #self.fc=nn.Linear(21,112).to(device)
        self.class_gat1=GATConv(112, hidden_channels,1, dropout).to(device)
        self.class_gat2=GATConv(64, hidden_channels,2, dropout).to(device)
        
        self.class_fc=nn.Sequential(nn.Linear(hidden_channels * 2, 256),
                                 nn.ReLU(),
                                 nn.BatchNorm1d(256),
                                 
                                 nn.Linear(256,128),
                                 nn.ReLU()                      
        ).to(device)
    
    def forward(self,hla_nodes,class_nodes,class_topo_edge_index):
        class_nodes=self.fc(class_nodes)
        class_topo_node_features = torch.cat([hla_nodes, class_nodes], dim=0)
        class_embedding=self.class_gat1(class_topo_node_features,class_topo_edge_index)
        class_embedding=self.class_gat2(class_embedding,class_topo_edge_index)
        class_emb_final=self.class_fc(class_embedding)
        hla_class_emb=class_emb_final[:112]
        return hla_class_emb

class MGHLA_mol(torch.nn.Module):   #contact and structure
    def __init__(self,struc_hid_dim,node_in_dim=64, node_h_dim=128, edge_in_dim=32, edge_h_dim=64,
                 struc_encoder_layer_num=3, struc_dropout=0.2, pep_max_len=15,
                 num_features_hla=32, num_features_pep=32, output_dim=128, hidden_channels=64, n_output=1,dropout=0.1):
        super(MGHLA_mol, self).__init__() 
        print('MGHLA_mol loading ...')
        self.relu = nn.ReLU().to(device)
        self.head_list=[1,2]
        self.n_output=n_output
        self.hidden_channels=hidden_channels
        self.dropout=dropout
        #contact graph
        self.mol_contact=Contact_GNN(num_features_hla,self.head_list,hidden_channels,dropout)
        
        #3-D structure
        self.struc_encoder = StructureEncoder(node_in_dim, node_h_dim, edge_in_dim, edge_h_dim, seq_in=False, num_layers=struc_encoder_layer_num, drop_rate=struc_dropout).to(device)
        self.struc_fc=nn.Sequential(nn.Linear(struc_hid_dim*3,64),
                                  nn.ReLU()
        ).to(device)  
        
        #hla (contact、structure、classtopo)concat
        self.concat_fc=nn.Sequential(nn.Linear(output_dim, 1024),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(1024),
                                    nn.Dropout(dropout),
                              
                                    nn.Linear(1024, 512),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(512),
                                    
                                    nn.Linear(512, 128),
                                    nn.ReLU()
        ).to(device)
         
        #peptide encoder
        self.pep_decoder=pep_Encoder_3().to(device)
        self.pep_fc=nn.Sequential(
            nn.Linear(pep_max_len*2*d_model, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, output_dim),
            nn.ReLU()
            
        ).to(device)
        
        
        #classifier
        self.phla_fc=nn.Sequential(
            nn.Linear(2 * output_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),

            nn.Linear(1024,512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            
            nn.Linear(512,128),
            nn.ReLU(),
            
            nn.Linear(128, self.n_output)
        ).to(device)
        self.kan=KAN([2 * output_dim,512,256,1],grid_size=10 ).to(device)
        
        
        self.out = nn.Linear(2, self.n_output).to(device)
        self.sigmoid = nn.Sigmoid().to(device)
        
        
    def forward(self,data_hla_contact,data_hla_3D,data_pep,class_topo_node_features1,class_topo_node_features2,class_topo_edge_index):
        hla_x, hla_edge_index, hla_edge_weight,hla_key, hla_batch = data_hla_contact.x, data_hla_contact.edge_index, data_hla_contact.edge_weight,data_hla_contact.hla_key, data_hla_contact.batch
        #contact_encoder
        x_contact=self.mol_contact(data_hla_contact)
        #3-d structure encoder
        data_hla_3D=data_hla_3D.to(device)
        struc_emb = self.struc_encoder((data_hla_3D.node_s,data_hla_3D.node_v),(data_hla_3D.edge_s,data_hla_3D.edge_v),edge_index=data_hla_3D.edge_index,seq=data_hla_3D.seq)
        struc_emb,hla_batch=important_nodes_new_nodeedge(struc_emb,hla_batch)
        struc_emb=torch.cat((struc_emb,struc_emb,struc_emb),dim=-1)
        struc_emb = gep(struc_emb, hla_batch)
        struc_emb=self.struc_fc(struc_emb)
        
        
        #concat
        x_hla=torch.cat((x_contact,struc_emb), dim=-1)
        x_hla=self.concat_fc(x_hla)
        
        #peptide encoder
        pep_feature= torch.stack([torch.tensor(data['x'], dtype=torch.float32) for data in data_pep]).to(device)
        pep_len=torch.tensor([data['length'] for data in data_pep]).to(device)
        
        
        pep_enc_outputs, pep_enc_self_attns=self.pep_decoder(pep_feature,pep_len)
        xp = pep_enc_outputs.view( pep_enc_outputs.shape[0], -1)
        xp=self.pep_fc(xp)
        
        #classifier
        xc=torch.cat((x_hla,xp),dim=-1)
        xmlp=self.phla_fc(xc)
        
        xkan=self.kan(xc)
        out_0=torch.cat([xmlp,xkan],dim=1).to(device)
        final_predictions=self.sigmoid(self.out(out_0))
        return final_predictions
        
     
class MGHLA_unstructure(torch.nn.Module):     #contact and classtopo
    def __init__(self,struc_hid_dim,node_in_dim=64, node_h_dim=128, edge_in_dim=32, edge_h_dim=64,
                 struc_encoder_layer_num=3, struc_dropout=0.2, pep_max_len=15,
                 num_features_hla=32, num_features_pep=32, output_dim=128, hidden_channels=64, n_output=1,dropout=0.1):
        super(MGHLA_unstructure, self).__init__() 
        print('MGHLA_unstructure loading ...')
        self.relu = nn.ReLU().to(device)
        self.head_list=[1,2]
        self.n_output=n_output
        self.hidden_channels=hidden_channels
        self.dropout=dropout
        #contact graph
        self.mol_contact=Contact_GNN(num_features_hla,self.head_list,hidden_channels,dropout)
        
        #classtopo graph
        self.classtopo_encoder=classtopo_GNN(num_features_hla,hidden_channels=64,dropout=0.1)
        #hla (contact、structure、classtopo)concat
        self.concat_fc=nn.Sequential(nn.Linear(int(output_dim+output_dim/2), 1024),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(1024),
                                    nn.Dropout(dropout),
                              
                                    nn.Linear(1024, 512),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(512),
                                    
                                    nn.Linear(512, 128),
                                    nn.ReLU()
        ).to(device)
         
        #peptide encoder
        self.pep_decoder=pep_Encoder_3().to(device)
        self.pep_fc=nn.Sequential(
            nn.Linear(pep_max_len*2*d_model, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, output_dim),
            nn.ReLU()
            
        ).to(device)
        
        
        #classifier
        self.phla_fc=nn.Sequential(
            nn.Linear(2 * output_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),

            nn.Linear(1024,512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            
            nn.Linear(512,128),
            nn.ReLU(),
            
            nn.Linear(128, self.n_output)
        ).to(device)
        self.kan=KAN([2 * output_dim,512,256,1],grid_size=10 ).to(device)
        
        
        self.out = nn.Linear(2, self.n_output).to(device)
        self.sigmoid = nn.Sigmoid().to(device)
        
        
    def forward(self,data_hla_contact,data_hla_3D,data_pep,class_topo_node_features1,class_topo_node_features2,class_topo_edge_index):
        hla_x, hla_edge_index, hla_edge_weight,hla_key, hla_batch = data_hla_contact.x, data_hla_contact.edge_index, data_hla_contact.edge_weight,data_hla_contact.hla_key, data_hla_contact.batch
        #contact_encoder
        x_contact=self.mol_contact(data_hla_contact)
     
        #classtopo_encoder
        input_hla_que=hla_key
        hla_class_emb=self.classtopo_encoder(class_topo_node_features1,class_topo_node_features2,class_topo_edge_index)
        x_class_emb=[hla_class_emb[input_hla_que[i]] for i in range(len(input_hla_que))]
        x_class_emb= torch.tensor([item.cpu().detach().numpy() for item in x_class_emb]).cuda().to(device)
        
        #concat
        x_hla=torch.cat((x_contact,x_class_emb), dim=-1)
        x_hla=self.concat_fc(x_hla)
        
        #peptide encoder
        pep_feature= torch.stack([torch.tensor(data['x'], dtype=torch.float32) for data in data_pep]).to(device)
        pep_len=torch.tensor([data['length'] for data in data_pep]).to(device)
        
        
        pep_enc_outputs, pep_enc_self_attns=self.pep_decoder(pep_feature,pep_len)
        xp = pep_enc_outputs.view( pep_enc_outputs.shape[0], -1)
        xp=self.pep_fc(xp)
        
        #classifier
        xc=torch.cat((x_hla,xp),dim=-1)
        xmlp=self.phla_fc(xc)
        
        xkan=self.kan(xc)
        out_0=torch.cat([xmlp,xkan],dim=1).to(device)
        final_predictions=self.sigmoid(self.out(out_0))
        return final_predictions
        
        
class MGHLA_classtopo(torch.nn.Module):   #only classtopo
    def __init__(self,struc_hid_dim,node_in_dim=64, node_h_dim=128, edge_in_dim=32, edge_h_dim=64,
                 struc_encoder_layer_num=3, struc_dropout=0.2, pep_max_len=15,
                 num_features_hla=32, num_features_pep=32, output_dim=128, hidden_channels=64, n_output=1,dropout=0.1):
        super(MGHLA_classtopo, self).__init__() 
        print('MGHLA_classtopo loading ...')
        self.relu = nn.ReLU().to(device)
        self.head_list=[1,2]
        self.n_output=n_output
        self.hidden_channels=hidden_channels
        self.dropout=dropout
        
        #classtopo graph
        self.classtopo_encoder=classtopo_GNN(num_features_hla,hidden_channels=64,dropout=0.1)
     
        #peptide encoder
        self.pep_decoder=pep_Encoder_3().to(device)
        self.pep_fc=nn.Sequential(
            nn.Linear(pep_max_len*2*d_model, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, output_dim),
            nn.ReLU()
            
        ).to(device)
        
        
        #classifier
        self.phla_fc=nn.Sequential(
            nn.Linear(2 * output_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),

            nn.Linear(1024,512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            
            nn.Linear(512,128),
            nn.ReLU(),
            
            nn.Linear(128, self.n_output)
        ).to(device)
        self.kan=KAN([2 * output_dim,512,256,1],grid_size=10 ).to(device)
        
        
        self.out = nn.Linear(2, self.n_output).to(device)
        self.sigmoid = nn.Sigmoid().to(device)
        
        
    def forward(self,data_hla_contact,data_hla_3D,data_pep,class_topo_node_features1,class_topo_node_features2,class_topo_edge_index):
        hla_x, hla_edge_index, hla_edge_weight,hla_key, hla_batch = data_hla_contact.x, data_hla_contact.edge_index, data_hla_contact.edge_weight,data_hla_contact.hla_key, data_hla_contact.batch
        
        #classtopo_encoder
        input_hla_que=hla_key
        hla_class_emb=self.classtopo_encoder(class_topo_node_features1,class_topo_node_features2,class_topo_edge_index)
        x_class_emb=[hla_class_emb[input_hla_que[i]] for i in range(len(input_hla_que))]
        x_class_emb= torch.tensor([item.cpu().detach().numpy() for item in x_class_emb]).cuda().to(device)
        
        #concat
        x_hla=x_class_emb
        
        #peptide encoder
        pep_feature= torch.stack([torch.tensor(data['x'], dtype=torch.float32) for data in data_pep]).to(device)
        pep_len=torch.tensor([data['length'] for data in data_pep]).to(device)
        
        
        pep_enc_outputs, pep_enc_self_attns=self.pep_decoder(pep_feature,pep_len)
        xp = pep_enc_outputs.view( pep_enc_outputs.shape[0], -1)
        xp=self.pep_fc(xp)
        
        #classifier
        xc=torch.cat((x_hla,xp),dim=-1)
        xmlp=self.phla_fc(xc)
        
        xkan=self.kan(xc)
        out_0=torch.cat([xmlp,xkan],dim=1).to(device)
        final_predictions=self.sigmoid(self.out(out_0))
        return final_predictions     
    
    
class MGHLA_onlyonehot(torch.nn.Module):
    def __init__(self,struc_hid_dim,node_in_dim=64, node_h_dim=128, edge_in_dim=32, edge_h_dim=64,
                 struc_encoder_layer_num=3, struc_dropout=0.2, pep_max_len=15,
                 num_features_hla=32, num_features_pep=32, output_dim=128, hidden_channels=64, n_output=1,dropout=0.1):
        super(MGHLA_onlyonehot, self).__init__() 
        print('MGHLA_onlyonehot loading ...')
        self.relu = nn.ReLU().to(device)
        self.head_list=[1,2]
        self.n_output=n_output
        self.hidden_channels=hidden_channels
        self.dropout=dropout
        #contact graph
        self.mol_contact=Contact_GNN(num_features_hla,self.head_list,hidden_channels,dropout)
        
        #3-D structure
        self.struc_encoder = StructureEncoder(node_in_dim, node_h_dim, edge_in_dim, edge_h_dim, seq_in=False, num_layers=struc_encoder_layer_num, drop_rate=struc_dropout).to(device)
        self.struc_fc=nn.Sequential(nn.Linear(struc_hid_dim*3,64),
                                  nn.ReLU()
        ).to(device)  
        
        #classtopo graph
        self.classtopo_encoder=classtopo_GNN(num_features_hla,hidden_channels=64,dropout=0.1)
        #hla (contact、structure、classtopo)concat
        self.concat_fc=nn.Sequential(nn.Linear(output_dim*2, 1024),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(1024),
                                    nn.Dropout(dropout),
                              
                                    nn.Linear(1024, 512),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(512),
                                    
                                    nn.Linear(512, 128),
                                    nn.ReLU()
        ).to(device)
         
        #peptide encoder
        self.pep_decoder=pep_Encoder_4().to(device)
        self.pep_fc=nn.Sequential(
            nn.Linear(pep_max_len*d_model, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, output_dim),
            nn.ReLU()
            
        ).to(device)
        
        
        #classifier
        self.phla_fc=nn.Sequential(
            nn.Linear(2 * output_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),

            nn.Linear(1024,512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            
            nn.Linear(512,128),
            nn.ReLU(),
            
            nn.Linear(128, self.n_output)
        ).to(device)
        self.kan=KAN([2 * output_dim,512,256,1],grid_size=10 ).to(device)
        
        
        self.out = nn.Linear(2, self.n_output).to(device)
        self.sigmoid = nn.Sigmoid().to(device)
        
        
    def forward(self,data_hla_contact,data_hla_3D,data_pep,class_topo_node_features1,class_topo_node_features2,class_topo_edge_index):
        hla_x, hla_edge_index, hla_edge_weight,hla_key, hla_batch = data_hla_contact.x, data_hla_contact.edge_index, data_hla_contact.edge_weight,data_hla_contact.hla_key, data_hla_contact.batch
        #contact_encoder
        x_contact=self.mol_contact(data_hla_contact)
        #3-d structure encoder
        data_hla_3D=data_hla_3D.to(device)
        struc_emb = self.struc_encoder((data_hla_3D.node_s,data_hla_3D.node_v),(data_hla_3D.edge_s,data_hla_3D.edge_v),edge_index=data_hla_3D.edge_index,seq=data_hla_3D.seq)
        struc_emb,hla_batch=important_nodes_new_nodeedge(struc_emb,hla_batch)
        struc_emb=torch.cat((struc_emb,struc_emb,struc_emb),dim=-1)
        struc_emb = gep(struc_emb, hla_batch)
        struc_emb=self.struc_fc(struc_emb)
        
        #classtopo_encoder
        input_hla_que=hla_key
        hla_class_emb=self.classtopo_encoder(class_topo_node_features1,class_topo_node_features2,class_topo_edge_index)
        x_class_emb=[hla_class_emb[input_hla_que[i]] for i in range(len(input_hla_que))]
        x_class_emb= torch.tensor([item.cpu().detach().numpy() for item in x_class_emb]).cuda().to(device)
        
        #concat
        x_hla=torch.cat((x_contact,struc_emb,x_class_emb), dim=-1)
        x_hla=self.concat_fc(x_hla)
        
        #peptide encoder
        pep_feature= torch.stack([torch.tensor(data['x'], dtype=torch.float32) for data in data_pep]).to(device)
        pep_len=torch.tensor([data['length'] for data in data_pep]).to(device)
        
        
        pep_enc_outputs, pep_enc_self_attns=self.pep_decoder(pep_feature,pep_len)
        xp = pep_enc_outputs.view( pep_enc_outputs.shape[0], -1)
        xp=self.pep_fc(xp)
        
        #classifier
        xc=torch.cat((x_hla,xp),dim=-1)
        xmlp=self.phla_fc(xc)
        
        xkan=self.kan(xc)
        out_0=torch.cat([xmlp,xkan],dim=1).to(device)
        final_predictions=self.sigmoid(self.out(out_0))
        return final_predictions   


