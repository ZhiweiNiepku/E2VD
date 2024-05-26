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
from sklearn.model_selection import KFold
import math
import time
import random

os.environ['CUDA_LAUNCH_BLOCKING']='1'

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
    def __init__(self,feats, cls_label, reg_label):
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



if __name__ == '__main__':
    BATCH_SIZE = 128
    EPOCH = 200
    SEQ_LEN=201
    REG_LOSS_K=1

    lr=1e-4

    reg_loss_func=nn.MSELoss()
    cls_loss_func=nn.BCELoss()

    k=5
    
    test_mse_all=[]
    test_corr_all=[]

    res_dir = "result_cnn_{}_lr{}_k{}_f1_prop_".format(BATCH_SIZE,lr,k)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    ##############
    # load data

    #feats=np.load('../new3_multi/data/single_embedding_data{}_T5-XL-UNI.npy'.format(add_))
    #reg_label=np.load('../new3_multi/data/single_embedding_label{}_T5-XL-UNI.npy'.format(add_)).reshape(-1,1)
    #cls_label=(reg_label>1).reshape(-1,1)

    feats=np.load('data_process/onehot_train_data.npy')
    reg_label=np.load('data_process/onehot_train_label.npy').reshape(-1,1)
    
    mask_=(reg_label!=1).flatten()
    
    feats = feats[mask_]
    reg_label=reg_label[mask_]
    
    cls_label=(reg_label>1)

    print(feats.shape)
    print(reg_label.shape)


    ###############

    kf=KFold(n_splits=k,shuffle=True,random_state=27)
    kf_idx=kf.split(feats,reg_label)
    #kf_idx=MyFold(cls_label,k,55)

    counter=0

    for i_idx in kf_idx:
    #for i_ in range(k):
    #    i_idx=kf_idx[i_]

        f = open(os.path.join(res_dir, 'result.txt'), 'a')
        print('fold',counter)
        f.write('fold {}, {}\n'.format(counter,time.ctime()))

        mse_best=20
        corr_best=0

        model = CNNModel()
        optimizer = torch.optim.Adam(model.parameters(),lr)
        model = model.to(device) #.cuda()

        train_feats=feats[i_idx[0]]
        train_reg_label=reg_label[i_idx[0]]
        train_cls_label=cls_label[i_idx[0]]

        val_feats=feats[i_idx[1]]
        val_reg_label=reg_label[i_idx[1]]
        val_cls_label=cls_label[i_idx[1]]


        Train_dataset = ScaleDataset(train_feats,train_cls_label,train_reg_label)
        train_loader_single = DataLoader(dataset=Train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        Val_dataset = ScaleDataset(val_feats,val_cls_label,val_reg_label)
        val_loader_single = DataLoader(dataset=Val_dataset, batch_size=16, shuffle=False)

        # training and testing
        for epoch in range(EPOCH):
            total_loss = []

            # training
            model.train()
            for step, (feat_, cls_label_, reg_label_) in enumerate(train_loader_single):
                reg_label_ = reg_label_.to(device)
                cls_label_ = cls_label_.to(device)
                feat_=feat_.to(device)
                cls_output, reg_output = model(feat_)
                cls_output=cls_output.squeeze()
                reg_output=reg_output.squeeze()

                reg_loss = reg_loss_func(reg_output.reshape(-1,1), reg_label_)
                cls_loss = cls_loss_func(cls_output.reshape(-1,1), cls_label_)
                #print(cls_loss.item(),reg_loss.item())
                loss=cls_loss + reg_loss*REG_LOSS_K
                total_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 40 == 0:
                    print(">>>>>loss:", sum(total_loss) / len(total_loss))

            # testing
            model.eval()
            with torch.no_grad():
                y_cls = []
                y_reg = []
                for step, (feat_, cls_label_, reg_label_) in enumerate(val_loader_single):
                    #label_ = label_.to(device) #.cuda()
                    feat_=feat_.to(device) #.cuda()

                    cls_output, reg_output = model(feat_)#.cpu().data.numpy().squeeze()
                    reg_output=reg_output.cpu().data.numpy().squeeze()
                    cls_output=cls_output.cpu().data.numpy().squeeze()
                    
                    y_reg.append(reg_output.reshape(-1,1))
                    y_cls.append(cls_output.reshape(-1,1))

                    
                # metrics for classification
                pred_cls = np.concatenate(y_cls,axis=0).flatten()
                y_cls_bool=(pred_cls>0.5)
                val_cls_label=val_cls_label.flatten().astype(bool)
                cm_=confusion_matrix(val_cls_label,y_cls_bool)
                vc_=(cm_[0][0]+cm_[1][1])/(cm_[0][0]+cm_[0][1]+cm_[1][0]+cm_[1][1])
                precision_=precision_score(val_cls_label,y_cls_bool) #cm_[1][1]/(cm_[0][1]+cm_[1][1])
                recall_=recall_score(val_cls_label,y_cls_bool) #cm_[1][1]/(cm_[1][0]+cm_[1][1])
                f1_=f1_score(val_cls_label,y_cls_bool)
                auc_=roc_auc_score(val_cls_label,pred_cls)

                print('Epoch: {}, auc: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}\n'.format(epoch,auc_,vc_,precision_,recall_,f1_))
                f.write('Epoch: {}, auc: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}\n'.format(epoch,auc_,vc_,precision_,recall_,f1_))
                
                # metrics for regression
                y_reg = np.concatenate(y_reg,axis=0).flatten()
                val_reg_label=val_reg_label.flatten()
                mse_=mean_squared_error(val_reg_label,y_reg)
                corr_=np.corrcoef(val_reg_label,y_reg)[0][1]

                print('Epoch: {}, mse: {:.3f}, corr: {:.3f}\n'.format(epoch,mse_,corr_))
                f.write('fold: {}, Epoch: {}, mse: {:.3f}, corr: {:.3f}\n'.format(counter,epoch,mse_,corr_))

                #if auc_ > auc_best:
                #if f1_ > f1_best:
                if mse_<mse_best:

                    mse_best=mse_
                    corr_best=corr_

                    # save model
                    save_path = os.path.join(res_dir, 'model_{}.pth.tar'.format(counter))
                    torch.save({'state_dict': model.state_dict()}, save_path)
                    print('Epoch {}, mse best: {}\n'.format(epoch,mse_best))
                    f.write('Epoch {}, mse best: {}\n'.format(epoch,mse_best))

        f.write('fold {}, mse: {:.4f}, corr: {:.4f}\n'.format(counter,mse_best,corr_best))

        test_mse_all.append(mse_best)
        test_corr_all.append(corr_best)
        f.close()

        counter+=1

    mse_mean=np.mean(test_mse_all)
    corr_mean=np.mean(test_corr_all)
    f = open(os.path.join(res_dir, 'result.txt'), 'a')
    f.write('total, mse: {:.4f}, corr: {:.4f}\n'.format(mse_mean,corr_mean))
    f.write('end time: {}'.format(time.ctime()))

    f.close()
