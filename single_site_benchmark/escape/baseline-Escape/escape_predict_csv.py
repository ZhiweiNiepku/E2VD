from transformers import BertTokenizer, BertModel, T5Tokenizer, T5EncoderModel
from torch.utils.data import DataLoader
import re
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import DataLoader
import random
from sklearn import preprocessing
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, mean_squared_error
import os
from sys import getsizeof
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import KFold
import math
import time
import random
import gc

os.environ['CUDA_LAUNCH_BLOCKING']='1'

device=torch.device('cuda')


class MLP(nn.Module):
    """A multilayer perceptron model."""

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_layer_dims):
        """Initialize the model.

        :param input_dim: The dimensionality of the input to the model.
        :param output_dim: The dimensionality of the output of the model.
        :param hidden_layer_dims: The dimensionalities of the hidden layers.
        """
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer_dims = hidden_layer_dims

        self.layer_dims = [self.input_dim] + list(self.hidden_layer_dims) + [self.output_dim]

        # Create layers
        self.layers = nn.ModuleList([
            nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])
            for i in range(len(self.layer_dims) - 1)
        ])

        # Create activation function
        self.activation = nn.ReLU()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Runs the model on the data.

        :param x: A FloatTensor containing an embedding of the antibody and/or antigen.
        :return: A FloatTensor containing the model's predicted escape score.
        """
        # Apply layers
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i != len(self.layers) - 1:
                x = self.activation(x)

        return x


class EmbeddingCoreModel(nn.Module):
    """The core neural model that predicts escape scores by using antibody/antigen embeddings."""

    def __init__(self,
                 input_dim: int,
                 hidden_layer_dims = (100, 100)):

        super(EmbeddingCoreModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_layer_dims = hidden_layer_dims

        self.output_dim = 1

        # Create MLP model
        self.mlp = MLP(
            input_dim=self.input_dim,
            output_dim=self.hidden_layer_dims[-1],
            hidden_layer_dims=self.hidden_layer_dims
        )

        self.mlp_cls = MLP(
            input_dim=self.hidden_layer_dims[-1],
            output_dim=self.output_dim,
            hidden_layer_dims=self.hidden_layer_dims
        )

        self.mlp_reg = MLP(
            input_dim=self.hidden_layer_dims[-1],
            output_dim=self.output_dim,
            hidden_layer_dims=self.hidden_layer_dims
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        """Runs the model on the data.

        :param x: A FloatTensor containing an embedding of the antibody and/or antigen
                  or a tuple of FloatTensors containing the antibody chain and antigen embeddings if using attention.
        :return: A FloatTensor containing the model's predicted escape score.
        """
        # Apply MLP
        x = self.mlp(x)

        output_cls = self.sigmoid(self.mlp_cls(x))
        output_reg = self.mlp_reg(x)

        #print(output_cls.shape)
        #print(output_reg.shape)

        return output_cls, output_reg


#######################################
# new loss define
from torch.nn.modules.loss import _Loss

class WeightMSELoss(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(WeightMSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target, k):
        input=input.reshape(1,-1)
        target=target.reshape(1,-1)
        mask=(target>=1)
        #print(input)
        return (torch.sum((input*mask-target*mask)**2)*k+torch.sum((input*(~mask)-target*(~mask))**2))/len(input)

class RegFocalLoss(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(RegFocalLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, pred, target, gamma=1):
        pred=pred.reshape(1,-1)
        target=target.reshape(1,-1)
        se_=torch.abs(pred-target)
        a_=torch.pow(se_,gamma).detach()
        a_sum=torch.sum(a_).detach()
        a_=(a_/a_sum).detach()
        return torch.sum(torch.pow(se_,2)*a_)

class ClsFocalLoss(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(ClsFocalLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, pred, target, gamma=1, alpha=0.5):
        assert alpha<1 and alpha>0

        epsilon=1e-7
        pred=pred.reshape(-1,)
        target=target.reshape(-1,)

        pt_0=1-pred[target==0]
        pt_1=pred[target==1]

        loss_0=(-torch.pow(1-pt_0,gamma)*torch.log(pt_0+epsilon)).sum()
        loss_1=(-torch.pow(1-pt_1,gamma)*torch.log(pt_1+epsilon)).sum()

        loss=(1-alpha)*loss_0+alpha*loss_1

        return loss/len(pred)

#######################################

#######################################
# sampling
def list_remove_(input,remove):
    input_=input.copy()
    for i in remove:
        input_.remove(i)
    return input_

def MyFold(label,k=5,num=50):
    random.seed(27)
    ret=[]
    label=label.reshape(-1,)

    pos_idxs=np.where(label==1)[0]
    neg_idxs=np.where(label==0)[0]

    pos_chosen_idxs_all=random.sample(pos_idxs.tolist(),k*num)
    neg_chosen_idxs_all=random.sample(neg_idxs.tolist(),k*num)

    for i in range(k):
        pos_train_=list_remove_(pos_idxs.tolist(),pos_chosen_idxs_all[i*num:i*num+num])
        neg_train_=list_remove_(neg_idxs.tolist(),neg_chosen_idxs_all[i*num:i*num+num])
        train_=pos_train_+neg_train_
        val_=pos_chosen_idxs_all[i*num:i*num+num]+neg_chosen_idxs_all[i*num:i*num+num]
        ret.append([train_,val_])
    
    return ret

########################################

# load embedding
RBD_emb=np.load('<path to data>/RBD_embedding_all_data.npy')#.astype(np.float16)
HSEQ_emb=np.load('<path to data>/antibody_heavy_sequence_embedding_all_data.npy')#.astype(np.float16)
LSEQ_emb=np.load('<path to data>/antibody_light_sequence_embedding_all_data.npy')#.astype(np.float16)

RBD_ori_emb=np.load('RBD_ori_embedding.npy') # 1,201,1280

# scale
mean_rbd=np.load('<path to data>/RBD_embedding_flatten_mean.npy')
std_rbd=np.load('<path to data>/RBD_embedding_flatten_std.npy')
mean_hseq=np.load('<path to data>/HSEQ_embedding_flatten_mean.npy')
std_hseq=np.load('<path to data>/HSEQ_embedding_flatten_std.npy')
mean_lseq=np.load('<path to data>/LSEQ_embedding_flatten_mean.npy')
std_lseq=np.load('<path to data>/LSEQ_embedding_flatten_std.npy')


shape_=RBD_emb.shape
RBD_emb=RBD_emb.reshape(shape_[0],-1)
RBD_emb=((RBD_emb-mean_rbd)/std_rbd).reshape(shape_)

shape_=RBD_ori_emb.shape
RBD_ori_emb=RBD_ori_emb.reshape(shape_[0], -1)
RBD_ori_emb=((RBD_ori_emb-mean_rbd)/std_rbd).reshape(shape_)

shape_=HSEQ_emb.shape
HSEQ_emb=HSEQ_emb.reshape(shape_[0],-1)
HSEQ_emb=((HSEQ_emb-mean_hseq)/std_hseq).reshape(shape_)

shape_=LSEQ_emb.shape
LSEQ_emb=LSEQ_emb.reshape(shape_[0],-1)
LSEQ_emb=((LSEQ_emb-mean_lseq)/std_lseq).reshape(shape_)

print('loaded embedding')

df_RBD = pd.read_csv('data/single_site_benchmark/escape/RBD_sequences.csv').values
emb_dict_RBD = dict()
for i in range(len(df_RBD)):
    emb_dict_RBD[str(df_RBD[i][0])+df_RBD[i][1]] = RBD_emb[i]

df_hseq = pd.read_csv('data/single_site_benchmark/escape/antibody_heavy_sequence.csv').values
emb_dict_hseq = dict()
for i in range(len(df_hseq)):
    emb_dict_hseq[df_hseq[i][0]] = HSEQ_emb[i]
    
df_lseq = pd.read_csv('data/single_site_benchmark/escape/antibody_light_sequence.csv').values
emb_dict_lseq = dict()
for i in range(len(df_lseq)):
    emb_dict_lseq[df_lseq[i][0]] = LSEQ_emb[i]

print('embedding dict done')

class DataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.antibodies = df['antibody'].values
        self.rbds = df.apply(lambda row: str(row['site']) + row['mutation'], axis=1).values
        self.sites = df['site'].values
        self.labels = df['mut_escape'].values
        self.RBD_ori_emb = torch.from_numpy(RBD_ori_emb).float()[0]

    def __getitem__(self, index):
        antibody_ = self.antibodies[index]
        rbd_ = self.rbds[index]
        site_ = self.sites[index]
        
        rbd_feat = torch.from_numpy(emb_dict_RBD[rbd_])
        hseq_feat = torch.from_numpy(emb_dict_hseq[antibody_])
        lseq_feat = torch.from_numpy(emb_dict_lseq[antibody_])
        reg_label = self.labels[index]
        cls_label = reg_label>0.4


        hseq_feat = hseq_feat.mean(dim = 0) # [1280, ]
        lseq_feat = lseq_feat.mean(dim = 0) # [1280, ]

        rbd_feat_res = rbd_feat[site_ - 331] - self.RBD_ori_emb[site_ - 331] # [1280, ]


        return torch.cat([rbd_feat_res, hseq_feat, lseq_feat], dim = 0), torch.tensor([cls_label]).float(), torch.tensor([reg_label]).float()

        #return torch.from_numpy(rbd_feat).float(), torch.from_numpy(hseq_feat).float(), torch.from_numpy(lseq_feat).float(), torch.tensor([cls_label]).float(), torch.tensor([reg_label]).float()
        #return torch.from_numpy(rbd_feat), torch.from_numpy(hseq_feat), torch.from_numpy(lseq_feat), torch.tensor([cls_label]), torch.tensor([reg_label])
        
        #return self.rbd_feats[index], self.hseq_feats[index], self.lseq_feats[index], self.cls_labels[index], self.reg_labels[index]
    def __len__(self):
        return len(self.labels)



df = pd.read_csv('data/single_site_benchmark/escape/split/raw_train.csv')
antibodies = list(set(df['antibody'].values))
ab_feats = np.eye(len(antibodies)).astype(float)

emb_dict_ab = dict()
for i in range(len(antibodies)):
    emb_dict_ab[antibodies[i]] = ab_feats[i]

class DataFrameDataset_onehot(torch.utils.data.Dataset):
    def __init__(self, df):
        self.antibodies = df['antibody'].values
        self.rbds = df.apply(lambda row: str(row['site']) + row['mutation'], axis=1).values
        self.sites = df['site'].values
        self.labels = df['mut_escape'].values
        self.RBD_ori_emb = torch.from_numpy(RBD_ori_emb).float()[0]

    def __getitem__(self, index):
        antibody_ = self.antibodies[index]
        rbd_ = self.rbds[index]
        site_ = self.sites[index]
        
        rbd_feat = torch.from_numpy(emb_dict_RBD[rbd_])
        ab_feat = torch.from_numpy(emb_dict_ab[antibody_])
        reg_label = self.labels[index]
        cls_label = reg_label>0.4

        rbd_feat_res = rbd_feat[site_ - 331] - self.RBD_ori_emb[site_ - 331] # [1280, ]

        return torch.cat([rbd_feat_res, ab_feat], dim = 0).float(), torch.tensor([cls_label]).float(), torch.tensor([reg_label]).float()

        #return torch.from_numpy(rbd_feat).float(), torch.from_numpy(hseq_feat).float(), torch.from_numpy(lseq_feat).float(), torch.tensor([cls_label]).float(), torch.tensor([reg_label]).float()
        #return torch.from_numpy(rbd_feat), torch.from_numpy(hseq_feat), torch.from_numpy(lseq_feat), torch.tensor([cls_label]), torch.tensor([reg_label])
        
        #return self.rbd_feats[index], self.hseq_feats[index], self.lseq_feats[index], self.cls_labels[index], self.reg_labels[index]
    def __len__(self):
        return len(self.labels)




if __name__ == '__main__':
    BATCH_SIZE = 100
    RBD_SEQ_LEN=201
    H_SEQ_LEN=136
    L_SEQ_LEN=136

    ONEHOT = True

    model_dir="<trained model path>/model_9.pth.tar"
    
    test_df = pd.read_csv('data/single_site_benchmark/escape/split/test.csv')

    if not ONEHOT:
        Test_dataset = DataFrameDataset(test_df)
        test_loader_single = DataLoader(dataset=Test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = EmbeddingCoreModel(1280*3, hidden_layer_dims=(1280, 640, 320, 100))
        model.load_state_dict(torch.load(model_dir)['state_dict'])
        model = model.to(device)

    else:
        Test_dataset = DataFrameDataset_onehot(test_df)
        test_loader_single = DataLoader(dataset=Test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = EmbeddingCoreModel(1280 + len(antibodies), hidden_layer_dims=(1280, 640, 320, 100))
        model.load_state_dict(torch.load(model_dir)['state_dict'])
        model = model.to(device)


    # testing
    model.eval()
    with torch.no_grad():
        y_cls = []
        y_reg = []
        val_cls_label = []
        val_reg_label = []
                
        for step, (feat_, cls_label_, reg_label_) in enumerate(test_loader_single):
            #label_ = label_.to(device) #.cuda()
            feat_=feat_.to(device)

            cls_output, reg_output = model(feat_)
            cls_output=cls_output.cpu().data.numpy().squeeze()
            reg_output=reg_output.cpu().data.numpy().squeeze()
            
            y_cls.append(cls_output.reshape(-1,1))
            y_reg.append(reg_output.reshape(-1,1))
            
            val_cls_label.append(cls_label_.data.numpy().reshape(-1,1))
            val_reg_label.append(reg_label_.data.numpy().reshape(-1,1))
                    
        val_cls_label = np.concatenate(val_cls_label,axis=0)
        val_reg_label = np.concatenate(val_reg_label,axis=0)

        # metrics for classification
        y_cls = np.concatenate(y_cls,axis=0).flatten()
        y_cls_bool=(y_cls>0.5)
        val_cls_label=val_cls_label.flatten().astype(bool)
        cm_=confusion_matrix(val_cls_label,y_cls_bool)
        vc_=(cm_[0][0]+cm_[1][1])/(cm_[0][0]+cm_[0][1]+cm_[1][0]+cm_[1][1])
        precision_=precision_score(val_cls_label,y_cls_bool) #cm_[1][1]/(cm_[0][1]+cm_[1][1])
        recall_=recall_score(val_cls_label,y_cls_bool) #cm_[1][1]/(cm_[1][0]+cm_[1][1])
        f1_=f1_score(val_cls_label,y_cls_bool)
        auc_=roc_auc_score(val_cls_label,y_cls)

        # metrics for regression
        y_reg = np.concatenate(y_reg,axis=0).flatten()
        val_reg_label=val_reg_label.flatten()
        mse_=mean_squared_error(val_reg_label,y_reg)
        corr_=np.corrcoef(val_reg_label,y_reg)[0][1]


    print('mse: {:.4f}, corr: {:.4f}, vc: {:.4f}\n'.format(mse_,corr_,vc_))
    print('auc: {:.4f}, f1: {:.4f}, precision: {:.4f} recall: {:.4f}\n'.format(auc_,f1_,precision_,recall_))
    print('[[{},{}],\n[{},{}]]\n\n\n'.format(cm_[0][0],cm_[0][1],cm_[1][0],cm_[1][1]))