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
import math
import matplotlib.pyplot as plt

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
    def __init__(self, seq_len_rbd, seq_len_hseq, seq_len_lseq, in_dim=1024, sa_out=1024, conv_out=1024):
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

        return out
    
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
    def __init__(self,feats,cls_label, reg_label):
        self.feats = torch.from_numpy(feats).float()
        self.cls_labels = torch.from_numpy(cls_label).float()
        self.reg_labels = torch.from_numpy(reg_label).float()

    def __getitem__(self, index):
        return self.feats[index], self.cls_labels[index], self.reg_labels[index]
    def __len__(self):
        return len(self.reg_labels)






if __name__ == '__main__':
    BATCH_SIZE = 200
    RBD_SEQ_LEN=201
    H_SEQ_LEN=136
    L_SEQ_LEN=136
    thres=0.4

    data_dir = '<path to data>'
    res_dir = '<trained model path>'

    SCALE=True

    if SCALE:
        mean_rbd=np.load(data_dir + 'RBD_embedding_flatten_mean.npy')
        std_rbd=np.load(data_dir + 'RBD_embedding_flatten_std.npy')
        mean_hseq=np.load(data_dir + 'HSEQ_embedding_flatten_mean.npy')
        std_hseq=np.load(data_dir + 'HSEQ_embedding_flatten_std.npy')
        mean_lseq=np.load(data_dir + 'LSEQ_embedding_flatten_mean.npy')
        std_lseq=np.load(data_dir + 'LSEQ_embedding_flatten_std.npy')

    run_round=1

    for i_ in range(run_round):
        
        # testing
        with torch.no_grad():

            model = SoluModel(RBD_SEQ_LEN,H_SEQ_LEN,L_SEQ_LEN)
            model_path=os.path.join(res_dir, 'model_9.pth.tar')
            model.load_state_dict(torch.load(model_path)['state_dict'])
            model = model.to(device)
            model.eval()
            
            train_rbd_feats=np.load('<path to data>/RBD_embedding_train-{}.npy'.format(i_))
            train_hseq_feats=np.load('<path to data>/HSEQ_embedding_train-{}.npy'.format(i_))
            train_lseq_feats=np.load('<path to data>/LSEQ_embedding_train-{}.npy'.format(i_))
            train_reg_label=np.load('<path to data>/label_train-{}.npy'.format(i_)).reshape(-1,1)
            train_cls_label=(train_reg_label>thres)

            ######################################
            #standard scale
            if SCALE:
                print('begin scale')

                shape_=train_rbd_feats.shape
                train_rbd_feats=train_rbd_feats.reshape(shape_[0],-1)
                train_rbd_feats=((train_rbd_feats-mean_rbd)/std_rbd).reshape(shape_)

                shape_=train_hseq_feats.shape
                train_hseq_feats=train_hseq_feats.reshape(shape_[0],-1)
                train_hseq_feats=((train_hseq_feats-mean_hseq)/std_hseq).reshape(shape_)

                shape_=train_lseq_feats.shape
                train_lseq_feats=train_lseq_feats.reshape(shape_[0],-1)
                train_lseq_feats=((train_lseq_feats-mean_lseq)/std_lseq).reshape(shape_)
                print('scale done')
            #####################################

            Train_dataset = ScaleDataset(train_rbd_feats,train_hseq_feats,train_lseq_feats,train_cls_label,train_reg_label)
            train_loader_single = DataLoader(dataset=Train_dataset, batch_size=BATCH_SIZE, shuffle=False)


            feat_neck = []
            for step, (rbd_feat_, hseq_feat_, lseq_feat_, cls_label_, reg_label_) in enumerate(train_loader_single):
                rbd_feat_=rbd_feat_.to(device)
                hseq_feat_=hseq_feat_.to(device)
                lseq_feat_=lseq_feat_.to(device)
                    
                output = model(rbd_feat_,hseq_feat_,lseq_feat_)
                output=output.cpu().data.numpy().squeeze()
                feat_neck.append(output)
                
            feat_neck = np.concatenate(feat_neck,axis=0)
            train_reg_label=train_reg_label.flatten()

            print(feat_neck.shape)
            print(train_reg_label.shape)

            np.save('feat_an.npy',feat_neck)
            np.save('reg_label.npy',train_reg_label)