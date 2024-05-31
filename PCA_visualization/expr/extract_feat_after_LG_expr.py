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
    def __init__(self, seq_len ,in_dim=1024, sa_out=1024, conv_out=1024):
        super(SoluModel, self).__init__()
        
        #self.self_attention=SelfAttention(in_dim,4,sa_out,0.6) # input: [B,L,1024] output: [B,L,1024]
        self.contextpooling=ContextPooling(seq_len,in_dim)

        self.conv=nn.Sequential( #input: [B,1024,L] output: [B,1024,L]
            nn.Conv1d(in_dim,in_dim*2,3,stride=1,padding=1),
            nn.LayerNorm((in_dim*2,seq_len)),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim*2,in_dim*2,3,stride=1,padding=1),
            nn.LayerNorm((in_dim*2,seq_len)),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim*2,conv_out,3,stride=1,padding=1),
            nn.LayerNorm((conv_out,seq_len)),
            nn.LeakyReLU(True),
        )

        self.cls_dim=sa_out+conv_out

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

    def forward(self, feats):
        out_sa=self.contextpooling(feats)+feats

        out_conv=self.conv(feats.permute(0,2,1))
        out_conv=out_conv.permute(0,2,1)+feats

        out=torch.cat([out_sa,out_conv],dim=2)
        out=torch.max(out,dim=1)[0].squeeze()

        #cls_out = self.classifier(out)
        #reg_out = self.regressor(out)

        #print(cls_out)

        #return cls_out,reg_out
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

    BATCH_SIZE = 64
    SEQ_LEN=201
    model_dir="<trained model path>/model_{}.pth.tar"

    model = SoluModel(SEQ_LEN)
    model = model.to(device)

    if 1:
        # load train data
        train_feats=np.load('<path to data>/single_expr_customize_train_data.npy')
        train_reg_label=np.load('<path to data>/single_expr_customize_train_label.npy')
        train_cls_label=(train_reg_label>1)
        scaler=StandardScaler()
        
        shape_=train_feats.shape
        train_feats=train_feats.reshape(shape_[0],-1)
        train_feats=scaler.fit_transform(train_feats)
        train_feats=train_feats.reshape(shape_)

    Test_dataset = ScaleDataset(train_feats,train_cls_label,train_reg_label)
    test_loader = DataLoader(dataset=Test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model_n=1


    # testing
    model.eval()
    with torch.no_grad():
        for model_i in range(model_n):

            print('fold',model_i)

            model_path=model_dir.format(model_i)
            model.load_state_dict(torch.load(model_path)['state_dict'])

            feat_neck = []
            for step, (feat_, cls_label_, reg_label_) in enumerate(test_loader):
                feat_=feat_.to(device)
                output = model(feat_)#.cpu().data.numpy().squeeze()
                output=output.cpu().data.numpy().squeeze()
                feat_neck.append(output)
            
            feat_neck = np.concatenate(feat_neck,axis=0)
            train_reg_label=train_reg_label.flatten()

            print(feat_neck.shape)
            print(train_reg_label.shape)

            np.save('feat_after.npy',feat_neck)
            np.save('reg_label.npy',train_reg_label)