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


class ANN(nn.Module):
    r"""An artificial neural network (ANN) for predicting 

    Parameters
    ----------
    in_features : int
        Number of features in input
    out_units : list
        List of layers with each element detailing number of neurons in each layer, e.g. two hidden layers: [16, 32]
    n_out : int
        Number of units in output layer
    p_dropout : float
        Probability of dropout, by default 0
    activation : nn.Activation
        A PyTorch activation function, by default nn.ReLU()

    Examples
    ----------
        >>> net = ANN(in_features=189, out_units=[16], n_out=1)
        >>> print(net)
            ANN(
                (fc): Sequential(
                    (fc0): Linear(in_features=189, out_features=16, bias=True)
                    (act0): ReLU()
                    (dropout): Dropout(p=0, inplace=False)
                )
                (fc_out): Linear(in_features=16, out_features=1, bias=True)
            )
    """

    def __init__(self, in_features = 1900, out_units = [16], p_dropout=0.14, activation=nn.ReLU()):
        super(ANN, self).__init__()

        # save args
        self.in_features = in_features
        self.out_units = out_units
        self.in_units = [self.in_features] + self.out_units[:-1]
        self.n_layers = len(self.out_units)

        # build the input and hidden layers
        self.fc = nn.Sequential()
        def add_linear(i):
            """Add n linear layers to the ANN"""
            self.fc.add_module('fc{}'.format(i), nn.Linear(self.in_units[i], self.out_units[i]))
            self.fc.add_module('{}{}'.format('act', i), activation)

        for i in range(self.n_layers):
            add_linear(i)
        
        # add dropout before final
        self.fc.add_module('dropout', nn.Dropout(p_dropout))

        # add final output layer
        self.fc_out_reg = nn.Linear(self.out_units[-1], 1)
        self.fc_out_cls = nn.Linear(self.out_units[-1], 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x): # (batch_size, in_units)

        o = self.fc(x) # (batch_size, out_units)
        o_cls = self.sigmoid(self.fc_out_cls(o)) # (batch_size, n_out)
        o_reg = self.fc_out_reg(o) # (batch_size, n_out) -> range now between 0 and 1

        return o_cls, o_reg

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
    model_dir='<model trained on all data>'

    test_data_path='<path to data>/1900_test_all_data.npy'
    add_='expr_1900'

    model = ANN()
    model = model.to(device)

    if 1:
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
            cls_output, reg_output = model(feat_)#.cpu().data.numpy().squeeze()
            cls_output=cls_output.cpu().data.numpy().squeeze()
            reg_output=reg_output.cpu().data.numpy().squeeze()
            y_cls.append(cls_output.reshape(-1,1))
            y_reg.append(reg_output.reshape(-1,1))


        y_reg = np.concatenate(y_reg,axis=0).flatten()

        y_cls = np.concatenate(y_cls,axis=0).flatten()
        y_cls_bool=(y_cls>0.5)


    np.save('predict_reg_all_{}.npy'.format(add_),y_reg)
    np.save('predict_cls_all_{}.npy'.format(add_),y_cls)
    np.save('predict_cls_bool_all_{}.npy'.format(add_),y_cls_bool)

