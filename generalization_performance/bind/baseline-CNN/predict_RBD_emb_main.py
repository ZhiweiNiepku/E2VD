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

class PredictDataset(torch.utils.data.Dataset):
    def __init__(self,feats):
        self.feats = torch.from_numpy(feats).float()

    def __getitem__(self, index):
        return self.feats[index]
    def __len__(self):
        return len(self.feats)


if __name__ == '__main__':

    BATCH_SIZE = 64
    SEQ_LEN=201
    model_dir="model/model.pth.tar"

    test_data_path='<path to data>/multi_onehot_test_data.npy'
    add_='bind_onehot_test'

    model = CNNModel()
    model = model.to(device)

    # load test data
    test_feats=np.load(test_data_path)

    Test_dataset = PredictDataset(test_feats)
    test_loader = DataLoader(dataset=Test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # testing
    model.eval()
    with torch.no_grad():
        model.load_state_dict(torch.load(model_dir)['state_dict'])

        y_cls = []
        y_reg = []
        for step, feat_ in enumerate(test_loader):
            feat_=feat_.to(device)
            if device.type=='cpu':
                cls_output, reg_output = model(feat_)#.data.numpy().squeeze()
                cls_output=cls_output.data.numpy().squeeze()
                reg_output=reg_output.data.numpy().squeeze()
            else:
                cls_output, reg_output = model(feat_)#.cpu().data.numpy().squeeze()
                cls_output=cls_output.cpu().data.numpy().squeeze()
                reg_output=reg_output.cpu().data.numpy().squeeze()
            y_cls.append(cls_output.reshape(-1,1))
            y_reg.append(reg_output.reshape(-1,1))


        y_reg = np.concatenate(y_reg,axis=0).flatten() #回归预测结果

        y_cls = np.concatenate(y_cls,axis=0).flatten() #分类预测结果
        y_cls_bool=(y_cls>0.5)


    np.save('predict_reg_all_{}.npy'.format(add_),y_reg)
    np.save('predict_cls_all_{}.npy'.format(add_),y_cls)
    np.save('predict_cls_bool_all_{}.npy'.format(add_),y_cls_bool)
