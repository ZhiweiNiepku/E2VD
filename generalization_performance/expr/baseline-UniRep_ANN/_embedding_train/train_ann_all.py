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

os.environ['CUDA_LAUNCH_BLOCKING']='1'

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



if __name__ == '__main__':
    BATCH_SIZE = 64
    EPOCH = 150
    SEQ_LEN=201
    REG_LOSS_K=1

    lr=2e-4

    cls_loss_func=nn.BCELoss()
    reg_loss_func=nn.MSELoss()

    res_dir = "result_{}_lr{}_regk{}_f1_prop_".format(BATCH_SIZE,lr,REG_LOSS_K)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    ##############
    
    #feats=np.array(pickle.load(open('../../data_process/expr_variant_data_train.txt_embedding.pickle','rb'))) #np.load('/code/lxd/85_backbone/addexp/LGM/T5/data/single_embedding_data_all.npy')
    #reg_label=np.load('../../data_process/expr_variant_label_train.npy').reshape(-1,1)
    
    feats = np.load('<path to data>/1900_all_data.npy')
    reg_label=np.load('<path to data>/1900_all_label.npy').reshape(-1,1)
    
    #mask_=(reg_label!=1).flatten()
    mask_=pd.isna(reg_label).flatten()
    assert np.sum(mask_) == 0

    # feats=feats[~mask_]
    # reg_label=reg_label[~mask_]
    cls_label=(reg_label>1).reshape(-1,1)

    print(feats.shape)
    print(reg_label.shape)


    ###############


    f = open(os.path.join(res_dir, 'result.txt'), 'a')
    f.write('begin time{}\n'.format(time.ctime()))
    f.close()

    model = ANN()
    optimizer = torch.optim.Adam(model.parameters(),lr)
    model = model.to(device) #.cuda()

    train_feats=feats
    train_reg_label=reg_label
    train_cls_label=cls_label

    Train_dataset = ScaleDataset(train_feats,train_cls_label,train_reg_label)
    train_loader_single = DataLoader(dataset=Train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    mse_best=20

    # training and testing
    for epoch in range(EPOCH):
        f = open(os.path.join(res_dir, 'result.txt'), 'a')
        total_loss = []
        pred_reg=[]
        pred_cls=[]
        label_reg=[]
        label_cls=[]

        # training
        model.train()
        for step, (feat_, cls_label_, reg_label_) in enumerate(train_loader_single):
            label_reg.append(reg_label_)
            label_cls.append(cls_label_)

            cls_label_ = cls_label_.to(device)
            reg_label_ = reg_label_.to(device)
            feat_=feat_.to(device)
            cls_output, reg_output = model(feat_)
            cls_output=cls_output.squeeze()
            reg_output=reg_output.squeeze()

            cls_loss = cls_loss_func(cls_output.flatten().float(), cls_label_.flatten().float())
            reg_loss = reg_loss_func(reg_output.flatten().float(), reg_label_.flatten().float())
            #print(cls_loss.item(),reg_loss.item())
            loss=cls_loss + reg_loss*REG_LOSS_K
            total_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            pred_reg.append(reg_output.flatten().cpu().data.reshape(-1,1))
            pred_cls.append(cls_output.flatten().cpu().data.reshape(-1,1))


            if (step+1) % 40 == 0:
                print(">>>>>loss:", sum(total_loss) / len(total_loss))

        epoch_loss_mean=np.mean(total_loss)

        print('Epoch: {}, loss: {:.3f}\n'.format(epoch,epoch_loss_mean))
        f.write('Epoch: {}, loss: {:.3f}\n'.format(epoch,epoch_loss_mean))

        pred_reg=np.concatenate(pred_reg,axis=0).flatten()
        pred_cls=np.concatenate(pred_cls,axis=0).flatten()
        label_reg=np.concatenate(label_reg,axis=0).flatten()
        label_cls=np.concatenate(label_cls,axis=0).flatten()

        # metrics for classification
        y_cls_bool=(pred_cls>0.5)
        val_cls_label=label_cls.flatten().astype(bool)
        cm_=confusion_matrix(val_cls_label,y_cls_bool)
        vc_=(cm_[0][0]+cm_[1][1])/(cm_[0][0]+cm_[0][1]+cm_[1][0]+cm_[1][1])
        precision_=precision_score(val_cls_label,y_cls_bool) #cm_[1][1]/(cm_[0][1]+cm_[1][1])
        recall_=recall_score(val_cls_label,y_cls_bool) #cm_[1][1]/(cm_[1][0]+cm_[1][1])
        f1_=f1_score(val_cls_label,y_cls_bool)
        auc_=roc_auc_score(val_cls_label,pred_cls)

        print('Epoch: {}, auc: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}\n'.format(epoch,auc_,vc_,precision_,recall_,f1_))
        f.write('Epoch: {}, auc: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}\n'.format(epoch,auc_,vc_,precision_,recall_,f1_))

        # metrics for regression
        mse_=mean_squared_error(label_reg,pred_reg)
        corr_=np.corrcoef(label_reg,pred_reg)[0][1]

        print('Epoch: {}, mse: {:.3f}, corr: {:.3f}\n'.format(epoch,mse_,corr_))
        f.write('Epoch: {}, mse: {:.3f}, corr: {:.3f}\n'.format(epoch,mse_,corr_))

        if epoch>99 or mse_<mse_best:
            mse_best=mse_

            # save model
            save_path = os.path.join(res_dir, 'model_{}.pth.tar'.format(epoch))
            torch.save({'state_dict': model.state_dict()}, save_path)
            print('Epoch {}, mse best: {}\n'.format(epoch,mse_best))
            f.write('Epoch {}, mse best: {}\n'.format(epoch,mse_best))

        f.close()


    f = open(os.path.join(res_dir, 'result.txt'), 'a')
    f.write('end time: {}'.format(time.ctime()))

    f.close()
