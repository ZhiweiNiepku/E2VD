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


class ContextPooling(nn.Module):
    def __init__(self,seq_len,in_dim=1024):
        super(ContextPooling,self).__init__()
        self.seq_len=seq_len
        self.conv=nn.Sequential(
            nn.Conv1d(in_dim,in_dim*2,3,stride=1,padding=1),
            nn.LayerNorm((in_dim*2,seq_len)),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim*2,in_dim*2,3,stride=1,padding=1),
            nn.LayerNorm((in_dim*2,seq_len)),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim*2,2,3,stride=1,padding=1),
            nn.LayerNorm((2,seq_len)),
            nn.LeakyReLU(True),
        )

    def _local_normal(self,s,center,r=0.1):
        PI=3.1415926
        std_=(r*self.seq_len*s[:,center]).unsqueeze(1) #[B,1]
        mean_=center
        place=torch.arange(self.seq_len).float().repeat(std_.shape[0],1).to(device) # [B,L]

        #print(std_)

        ret=pow(2*PI,-0.5)*torch.pow(std_,-1)*torch.exp(-torch.pow(place-mean_,2)/(1e-5+2*torch.pow(std_,2)))

        #ret-=torch.max(ret,dim=1)[0].unsqueeze(1)
        #ret=torch.softmax(ret,dim=1)

        ret/=torch.max(ret,dim=1)[0].unsqueeze(1)


        return ret

    def forward(self,feats): # feats: [B,L,1024]
        feats_=feats.permute(0,2,1)
        feats_=self.conv(feats_) #output: [B,2,L]
        s,w=feats_[:,0,:].squeeze(1),feats_[:,1,:].squeeze(1) #[B,L]
        s=torch.softmax(s,1)
        w=torch.softmax(w,1)

        out=[]

        for i in range(self.seq_len):
            w_=self._local_normal(s,i)*w
            w_=w_.unsqueeze(2) # [B,L,1]
            out.append((w_*feats).sum(1,keepdim=True)) # w_ [B,L,1], feats [B,L,1024]

        out=torch.cat(out,dim=1) # [B,L,1024]
        return out

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, output_size, dropout_prob):   
        super(SelfAttention, self).__init__()

        assert output_size%num_attention_heads==0

        self.num_attention_heads = num_attention_heads
        #self.attention_head_size = int(hidden_size / num_attention_heads)
        self.attention_head_size= int(output_size/num_attention_heads)
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # dropout
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        # INPUT:  x'shape = [bs, seqlen, hid_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        
        mixed_query_layer = self.query(hidden_states)   # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(hidden_states)       # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(hidden_states)   # [bs, seqlen, hid_size]

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)   # [bs, 8, seqlen, seqlen]

        attention_probs = nn.Softmax(dim=-1)(attention_scores)    # [bs, 8, seqlen, seqlen]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs).to(torch.float32)
        
        context_layer = torch.matmul(attention_probs, value_layer)   # [bs, 8, seqlen, 16]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()   # [bs, seqlen, 8, 16]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)   # [bs, seqlen, 128]
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class SoluModel(nn.Module):
    def __init__(self, seq_len_rbd, seq_len_hseq, seq_len_lseq, in_dim=1280, sa_out=1280, conv_out=1280):
        super(SoluModel, self).__init__()
        
        #self.self_attention=SelfAttention(in_dim,4,sa_out,0.6) # input: [B,L,1024] output: [B,L,1024]
        
        #rbd
        self.contextpooling_rbd=ContextPooling(seq_len_rbd,in_dim)
        self.conv_rbd=nn.Sequential( #input: [B,1024,L] output: [B,1024,L]
            nn.Conv1d(in_dim,in_dim*2,3,stride=1,padding=1),
            nn.LayerNorm((in_dim*2,seq_len_rbd)),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim*2,in_dim*2,3,stride=1,padding=1),
            nn.LayerNorm((in_dim*2,seq_len_rbd)),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim*2,conv_out,3,stride=1,padding=1),
            nn.LayerNorm((conv_out,seq_len_rbd)),
            nn.LeakyReLU(True),
        )
        
        #hseq
        self.contextpooling_hseq=ContextPooling(seq_len_hseq,in_dim)
        self.conv_hseq=nn.Sequential( #input: [B,1024,L] output: [B,1024,L]
            nn.Conv1d(in_dim,in_dim*2,3,stride=1,padding=1),
            nn.LayerNorm((in_dim*2,seq_len_hseq)),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim*2,in_dim*2,3,stride=1,padding=1),
            nn.LayerNorm((in_dim*2,seq_len_hseq)),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim*2,conv_out,3,stride=1,padding=1),
            nn.LayerNorm((conv_out,seq_len_hseq)),
            nn.LeakyReLU(True),
        )
        
        #lseq
        self.contextpooling_lseq=ContextPooling(seq_len_lseq,in_dim)
        self.conv_lseq=nn.Sequential( #input: [B,1024,L] output: [B,1024,L]
            nn.Conv1d(in_dim,in_dim*2,3,stride=1,padding=1),
            nn.LayerNorm((in_dim*2,seq_len_lseq)),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim*2,in_dim*2,3,stride=1,padding=1),
            nn.LayerNorm((in_dim*2,seq_len_lseq)),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim*2,conv_out,3,stride=1,padding=1),
            nn.LayerNorm((conv_out,seq_len_lseq)),
            nn.LeakyReLU(True),
        )
        

        self.cls_dim=(sa_out+conv_out)*3

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim, self.cls_dim // 4),
            nn.LeakyReLU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 4),
            nn.LeakyReLU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 64),
            nn.LeakyReLU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Linear(self.cls_dim // 64, 1),
            
            nn.Sigmoid())

        self.regressor = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim, self.cls_dim // 4),
            nn.LeakyReLU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 4),
            nn.LeakyReLU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 64),
            nn.LeakyReLU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Linear(self.cls_dim // 64, 1))
        
        self._initialize_weights()

    def forward(self, feats_rbd, feats_seqh, feats_seql):
        
        out_cp_rbd=self.contextpooling_rbd(feats_rbd)+feats_rbd
        out_conv_rbd=self.conv_rbd(feats_rbd.permute(0,2,1))
        out_conv_rbd=out_conv_rbd.permute(0,2,1)+feats_rbd
        
        
        out_cp_seqh=self.contextpooling_hseq(feats_seqh)+feats_seqh
        out_conv_seqh=self.conv_hseq(feats_seqh.permute(0,2,1))
        out_conv_seqh=out_conv_seqh.permute(0,2,1)+feats_seqh
        
        
        out_cp_seql=self.contextpooling_lseq(feats_seql)+feats_seql
        out_conv_seql=self.conv_lseq(feats_seql.permute(0,2,1))
        out_conv_seql=out_conv_seql.permute(0,2,1)+feats_seql
        

        out_rbd=torch.cat([out_cp_rbd,out_conv_rbd],dim=2)
        out_rbd=torch.max(out_rbd,dim=1)[0].squeeze()
        
        out_seqh=torch.cat([out_cp_seqh,out_conv_seqh],dim=2)
        out_seqh=torch.max(out_seqh,dim=1)[0].squeeze()
        
        out_seql=torch.cat([out_cp_seql,out_conv_seql],dim=2)
        out_seql=torch.max(out_seql,dim=1)[0].squeeze()
        
        out=torch.cat([out_rbd,out_seqh,out_seql],dim=1)

        cls_out = self.classifier(out)
        reg_out = self.regressor(out)

        #print(cls_out)

        return cls_out,reg_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class ScaleDataset(torch.utils.data.Dataset):
    def __init__(self,rbd_feats,hseq_feats,lseq_feats,cls_label, reg_label):
        self.rbd_feats = torch.from_numpy(rbd_feats).float()
        self.hseq_feats = torch.from_numpy(hseq_feats).float()
        self.lseq_feats = torch.from_numpy(lseq_feats).float()
        
        self.cls_labels = torch.from_numpy(cls_label).float()
        self.reg_labels = torch.from_numpy(reg_label).float()

    def __getitem__(self, index):
        return self.rbd_feats[index], self.hseq_feats[index], self.lseq_feats[index], self.cls_labels[index], self.reg_labels[index]
    def __len__(self):
        return len(self.reg_labels)



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
        self.labels = df['mut_escape'].values

    def __getitem__(self, index):
        antibody_ = self.antibodies[index]
        rbd_ = self.rbds[index]
        
        rbd_feat = emb_dict_RBD[rbd_]
        hseq_feat = emb_dict_hseq[antibody_]
        lseq_feat = emb_dict_lseq[antibody_]
        reg_label = self.labels[index]
        cls_label = reg_label>0.4
        
        
        return torch.from_numpy(rbd_feat).float(), torch.from_numpy(hseq_feat).float(), torch.from_numpy(lseq_feat).float(), torch.tensor([cls_label]).float(), torch.tensor([reg_label]).float()
        #return torch.from_numpy(rbd_feat), torch.from_numpy(hseq_feat), torch.from_numpy(lseq_feat), torch.tensor([cls_label]), torch.tensor([reg_label])
        
        #return self.rbd_feats[index], self.hseq_feats[index], self.lseq_feats[index], self.cls_labels[index], self.reg_labels[index]
    def __len__(self):
        return len(self.labels)







if __name__ == '__main__':
    BATCH_SIZE = 100
    RBD_SEQ_LEN=201
    H_SEQ_LEN=136
    L_SEQ_LEN=136

    model_dir="<trained model path>/model_9.pth.tar"
    
    test_df = pd.read_csv('data/single_site_benchmark/escape/split/test.csv')

    Test_dataset = DataFrameDataset(test_df)
    test_loader_single = DataLoader(dataset=Test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SoluModel(RBD_SEQ_LEN,H_SEQ_LEN,L_SEQ_LEN)
    model.load_state_dict(torch.load(model_dir)['state_dict'])
    model = model.to(device)

    # testing
    model.eval()
    with torch.no_grad():
        y_cls = []
        y_reg = []
        val_cls_label = []
        val_reg_label = []
                
        for step, (rbd_feat_, hseq_feat_, lseq_feat_, cls_label_, reg_label_) in enumerate(test_loader_single):
            #label_ = label_.to(device) #.cuda()
            rbd_feat_=rbd_feat_.to(device)
            hseq_feat_=hseq_feat_.to(device)
            lseq_feat_=lseq_feat_.to(device)

            cls_output, reg_output = model(rbd_feat_,hseq_feat_,lseq_feat_)#.cpu().data.numpy().squeeze()
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