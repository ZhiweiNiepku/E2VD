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
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt


device=torch.device('cuda')

class CNNModel(nn.Module):
    def __init__(self, in_dim=20, conv_out=1024, seq_len=201):
        super(CNNModel, self).__init__()
        self.conv=nn.Sequential( #input: [B,1024,L] output: [B,1024,L]
            nn.Conv1d(in_dim,in_dim,3,stride=1,padding=1),
            nn.LayerNorm((in_dim,seq_len)),
            nn.MaxPool1d(2),
            nn.LeakyReLU(True),
            
            nn.Conv1d(in_dim,in_dim,3,stride=1,padding=1),
            nn.LayerNorm((in_dim,seq_len//2)),
            nn.MaxPool1d(2),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim,conv_out,3,stride=1,padding=1))

        self.regression=nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(conv_out, conv_out // 4),
            nn.LeakyReLU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Dropout(p=0.6),
            nn.Linear(conv_out // 4, conv_out // 4),
            nn.LeakyReLU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Linear(conv_out // 4, 1))
        
        self.classification=nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(conv_out, conv_out // 4),
            nn.LeakyReLU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Dropout(p=0.6),
            nn.Linear(conv_out // 4, conv_out // 4),
            nn.LeakyReLU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Linear(conv_out // 4, 1),
            nn.Sigmoid())
        
        self._initialize_weights()
        
    def forward(self, data):
        out=self.conv(data.permute(0,2,1)).permute(0,2,1)
        out=torch.max(out,dim=1)[0].squeeze()
        reg_out=self.regression(out)
        cls_out=self.classification(out)
        return cls_out, reg_out

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
    SEQ_LEN=504
    model_dir="<trained model path>/model_{}.pth.tar"

    model = CNNModel(seq_len=SEQ_LEN)
    model = model.to(device)

    if 1:
        # load test data
        test_feats=np.load('data/other_virus/zika/CNN/onehot_test_data.npy')
        test_reg_label=np.load('data/other_virus/zika/CNN/onehot_test_label.npy') / 10
        test_cls_label=(test_reg_label>0)

    Test_dataset = ScaleDataset(test_feats,test_cls_label,test_reg_label)
    test_loader = DataLoader(dataset=Test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model_n=5

    test_vc_all =[]
    test_precision_all=[]
    test_recall_all=[]
    test_f1_all=[]
    test_auc_all=[]
    test_mse_all=[]
    test_corr_all=[]

    # testing
    model.eval()
    with torch.no_grad():
        for model_i in range(model_n):

            print('fold',model_i)

            model_path=model_dir.format(model_i)
            model.load_state_dict(torch.load(model_path)['state_dict'])

            y_reg = []
            y_cls = []
            for step, (feat_, cls_label_, reg_label_) in enumerate(test_loader):
                feat_=feat_.to(device)

                cls_output, reg_output = model(feat_)#.cpu().data.numpy().squeeze()
                reg_output=reg_output.cpu().data.numpy().squeeze()
                cls_output=cls_output.cpu().data.numpy().squeeze()
                
                y_reg.append(reg_output.reshape(-1,1))
                y_cls.append(cls_output.reshape(-1,1))

            
            # metrics for classification
            pred_cls = np.concatenate(y_cls,axis=0).flatten()
            y_cls_bool=(pred_cls>0.5)
            test_cls_label=test_cls_label.flatten().astype(bool)
            cm_=confusion_matrix(test_cls_label,y_cls_bool)
            vc_=(cm_[0][0]+cm_[1][1])/(cm_[0][0]+cm_[0][1]+cm_[1][0]+cm_[1][1])
            precision_=precision_score(test_cls_label,y_cls_bool) #cm_[1][1]/(cm_[0][1]+cm_[1][1])
            recall_=recall_score(test_cls_label,y_cls_bool) #cm_[1][1]/(cm_[1][0]+cm_[1][1])
            f1_=f1_score(test_cls_label,y_cls_bool)
            auc_=roc_auc_score(test_cls_label,pred_cls)

            print('fold: {}, auc: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}\n'.format(model_i,auc_,vc_,precision_,recall_,f1_))
            
            # metrics for regression
            y_reg = np.concatenate(y_reg,axis=0).flatten()
            test_reg_label=test_reg_label.flatten()
            mse_=mean_squared_error(test_reg_label,y_reg)
            corr_=np.corrcoef(test_reg_label,y_reg)[0][1]

            print('fold: {}, mse: {:.4f}, corr: {:.4f}\n'.format(model_i,mse_,corr_))

            test_precision_all.append(precision_)
            test_recall_all.append(recall_)
            test_f1_all.append(f1_)
            test_auc_all.append(auc_)
            test_vc_all.append(vc_)
            test_mse_all.append(mse_)
            test_corr_all.append(corr_)
        

    vc_mean=np.mean(test_vc_all)
    mse_mean=np.mean(test_mse_all)
    corr_mean=np.mean(test_corr_all)
    precision_mean=np.mean(test_precision_all)
    recall_mean=np.mean(test_recall_all)
    f1_mean=np.mean(test_f1_all)
    auc_mean=np.mean(test_auc_all)

    print('total, auc: {:.4f}, vc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, mse: {:.4f}, corr: {:.4f}'.format(auc_mean,vc_mean,f1_mean,precision_mean,
                                                                                                                            recall_mean,mse_mean,corr_mean))
