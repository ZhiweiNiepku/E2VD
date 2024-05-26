from transformers import BertTokenizer, BertModel, T5Tokenizer, T5EncoderModel
from torch.utils.data import DataLoader
import re
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class SoluModel(nn.Module):
    """
    Inputs:
    maxlen - maximum length of protein sequence, used for position embedding
    vocab_size - number of amino acids, used for token embedding
    embed_dim - dimension of embedding vectors
    num_heads - number of attention heads
    num_blocks - number of transformer blocks
    ff_dim - number of neurons in dense layers inside transformer block
    ff_dim? - number of neurons in dense layers 
    """
    def __init__(self,
                maxlen,
                vocab_size,
                embed_dim,
                num_heads,
                num_blocks,
                ff_dim,
                ff_dim2,
                ff_dim3,
                ff_dim4):
        super(SoluModel, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.ff_dim = ff_dim
        self.ff_dim2 = ff_dim2
        self.ff_dim3 = ff_dim3
        self.ff_dim4 = ff_dim4
        
        self.token_embedding = nn.Embedding(self.vocab_size,
                                            self.embed_dim,
                                            padding_idx=0)
        self.position_embedding = nn.Embedding(self.maxlen+1,
                                               self.embed_dim,
                                               padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(self.embed_dim,
                                                        self.num_heads,
                                                        dim_feedforward=self.ff_dim,
                                                        batch_first=True,
                                                        dropout=0.,
                                                        activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=self.num_blocks)
        
        self.linear1 = nn.Linear(self.embed_dim, self.ff_dim2)
        self.linear2 = nn.Linear(self.ff_dim2, 20)
        self.linear3 = nn.Linear(20*3, self.ff_dim3)
        self.linear4 = nn.Linear(self.ff_dim3, self.ff_dim4)
        #self.outlinear = nn.Linear(self.ff_dim4, 1)

        self.outlinear_cls = nn.Sequential(
            nn.Linear(self.ff_dim4, 1),
            nn.Sigmoid()
            )
        self.outlinear_reg = nn.Linear(self.ff_dim4, 1)
        
    def forward(self, heavys, heavyp, lights, lightp, rbds, rbdp, outtype='final'):
        """
        Execute neural network
        
        Inputs:
        heavys - ordinal encoding of heavy chain sequence (masked residues are 0)
        heavyp - position indices of heavy chain sequence (masked residues are 0)
        lights - ordinal encoding of light chain sequence (masked residues are 0)
        lightp - position indices of light chain sequence (masked residues are 0)
        rbds - ordinal encoding of rbd chain sequence (masked residues are 0)
        rbdp - position indices of rbd chain sequence (masked residues are 0)
        outtype - type of output. For example, if heavy then heavy chain embeddings are returned.
        
        Outputs:
        embeddings vectors or predicted log escape fraction
        """
        token_emb = self.token_embedding(heavys)
        pos_emb = self.position_embedding(heavyp)
        x = token_emb + pos_emb
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1) # global average pooling
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.leaky_relu(x, negative_slope=0.3)
        if outtype == 'heavy':
            return x

        token_emb2 = self.token_embedding(lights)
        pos_emb2 = self.position_embedding(lightp)
        x2 = token_emb2 + pos_emb2
        x2 = self.transformer_encoder(x2)
        x2 = torch.mean(x2, dim=1)
        x2 = self.linear1(x2)
        x2 = F.relu(x2)
        x2 = self.linear2(x2)
        x2 = F.leaky_relu(x2, negative_slope=0.3)
        if outtype == 'light':
            return x2

        token_emb3 = self.token_embedding(rbds)
        pos_emb3 = self.position_embedding(rbdp)
        x3 = token_emb3 + pos_emb3
        x3 = self.transformer_encoder(x3)
        x3 = torch.mean(x3, dim=1)
        x3 = self.linear1(x3)
        x3 = F.relu(x3)
        x3 = self.linear2(x3)
        x3 = F.leaky_relu(x3, negative_slope=0.3)
        if outtype == 'rbd':
            return x3

        x = torch.cat([x, x2, x3], dim=1)
        x = self.linear3(x)
        x = F.leaky_relu(x, negative_slope=0.3)
        x = self.linear4(x)
        x = F.leaky_relu(x, negative_slope=0.3)
        cls_output = self.outlinear_cls(x)
        reg_output = self.outlinear_reg(x)

        return cls_output, reg_output

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


df_RBD = pd.read_csv('data/single_site_benchmark/escape/RBD_sequences.csv').values
emb_dict_RBD = dict()
for i in range(len(df_RBD)):
    emb_dict_RBD[str(df_RBD[i][0])+df_RBD[i][1]] = df_RBD[i][2]

df_hseq = pd.read_csv('data/single_site_benchmark/escape/antibody_heavy_sequence.csv').values
emb_dict_hseq = dict()
for i in range(len(df_hseq)):
    emb_dict_hseq[df_hseq[i][0]] = df_hseq[i][1]
    
df_lseq = pd.read_csv('data/single_site_benchmark/escape/antibody_light_sequence.csv').values
emb_dict_lseq = dict()
for i in range(len(df_lseq)):
    emb_dict_lseq[df_lseq[i][0]] = df_lseq[i][1]

aa_map = {'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,
          'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,
          'X':0, '-':0}

RBD_SEQ_LEN=201
H_SEQ_LEN=136
L_SEQ_LEN=136

class DataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.antibodies = df['antibody'].values
        self.rbds = df.apply(lambda row: str(row['site']) + row['mutation'], axis=1).values
        self.labels = df['mut_escape'].values

    def __getitem__(self, index):
        antibody_ = self.antibodies[index]
        rbd_ = self.rbds[index]
        
        rbd_seq = emb_dict_RBD[rbd_]
        hseq_seq = emb_dict_hseq[antibody_][:H_SEQ_LEN]
        lseq_seq = emb_dict_lseq[antibody_][:L_SEQ_LEN]
        reg_label = self.labels[index]
        cls_label = reg_label>0.4

        rbd_feat = torch.from_numpy(np.array([aa_map[a] for a in rbd_seq]))
        hseq_feat = np.array([aa_map[a] for a in hseq_seq])
        lseq_feat = np.array([aa_map[a] for a in lseq_seq])

        hseq_feat = torch.from_numpy(np.pad(hseq_feat, (0, H_SEQ_LEN - len(hseq_feat)), 'constant', constant_values = 0))
        lseq_feat = torch.from_numpy(np.pad(lseq_feat, (0, L_SEQ_LEN - len(lseq_feat)), 'constant', constant_values = 0))
        
        rbd_pos = torch.where(rbd_feat > 0, torch.arange(RBD_SEQ_LEN), 0)
        hseq_pos = torch.where(hseq_feat > 0, torch.arange(L_SEQ_LEN), 0)
        lseq_pos = torch.where(lseq_feat > 0, torch.arange(H_SEQ_LEN), 0)

        return hseq_feat, hseq_pos, lseq_feat, lseq_pos, rbd_feat, rbd_pos, torch.tensor([cls_label]).float(), torch.tensor([reg_label]).float()

        #return torch.from_numpy(rbd_feat).float(), torch.from_numpy(hseq_feat).float(), torch.from_numpy(lseq_feat).float(), torch.tensor([cls_label]).float(), torch.tensor([reg_label]).float()
        #return torch.from_numpy(rbd_feat), torch.from_numpy(hseq_feat), torch.from_numpy(lseq_feat), torch.tensor([cls_label]), torch.tensor([reg_label])
        
        #return self.rbd_feats[index], self.hseq_feats[index], self.lseq_feats[index], self.cls_labels[index], self.reg_labels[index]
    def __len__(self):
        return len(self.labels)







if __name__ == '__main__':
    BATCH_SIZE = 200
    EPOCH = 70
    REG_LOSS_K=1.2
    thres=0.4
    
    SCALE=True

    lr=1e-4


    cls_loss_func = nn.BCELoss()
    reg_loss_func = nn.MSELoss()

    k=5
    run_round=10
    
    test_vc_all =[]
    test_precision_all=[]
    test_recall_all=[]
    test_f1_all=[]
    test_auc_all=[]
    test_mse_all=[]
    test_corr_all=[]

    res_dir = "result_{}_lr{}_regk{}_k{}_f1_prop_loss_".format(BATCH_SIZE,lr,REG_LOSS_K,k)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    counter=0
    
    val_df = pd.read_csv('data/single_site_benchmark/escape/split/val.csv')
    
    #for i_idx in kf_idx:
    for i_ in range(run_round):
        #i_idx=kf_idx[i_]

        f = open(os.path.join(res_dir, 'result.txt'), 'a')
        f_loss=open(os.path.join(res_dir, 'result_loss.txt'),'a')
        print('fold',counter)
        f.write('fold {}, {}\n'.format(counter,time.ctime()))
        f_loss.write('fold {}, {}\n'.format(counter,time.ctime()))
        
        vc_best=0
        precision_best=0
        recall_best=0
        f1_best=0
        auc_best=0
        cm_best=None
        mse_best=20

        
        corr_best=0

        model = SoluModel(201, 21, 36, 6, 1, 32, 64, 256, 32)
        if i_>0:
            model_path=os.path.join(res_dir, 'model_{}.pth.tar'.format(i_-1))
            model.load_state_dict(torch.load(model_path)['state_dict'])
        optimizer = torch.optim.Adam(model.parameters(),lr)
        model = model.to(device) #.cuda()
        
        train_df = pd.read_csv('data/single_site_benchmark/escape/split/train-{}.csv'.format(i_))

        Train_dataset = DataFrameDataset(train_df)
        train_loader_single = DataLoader(dataset=Train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        Val_dataset = DataFrameDataset(val_df)
        val_loader_single = DataLoader(dataset=Val_dataset, batch_size=200, shuffle=False)

        # training and testing
        for epoch in range(EPOCH):
            total_loss = []
            total_loss_cls = []
            total_loss_reg = []

            # training
            model.train()
            for step, (hseq_feat_, hseq_pos_, lseq_feat_, lseq_pos_, rbd_feat_, rbd_pos_, cls_label_, reg_label_) in enumerate(train_loader_single):

                cls_label_ = cls_label_.to(device)
                reg_label_ = reg_label_.to(device)
                
                rbd_feat_=rbd_feat_.to(device)
                hseq_feat_=hseq_feat_.to(device)
                lseq_feat_=lseq_feat_.to(device)
                rbd_pos_=rbd_pos_.to(device)
                hseq_pos_=hseq_pos_.to(device)
                lseq_pos_=lseq_pos_.to(device)
                
                cls_output, reg_output = model(hseq_feat_, hseq_pos_, lseq_feat_, lseq_pos_, rbd_feat_, rbd_pos_)
                cls_output=cls_output.squeeze()
                reg_output=reg_output.squeeze()


                cls_loss = cls_loss_func(cls_output.flatten().float(), cls_label_.flatten().float())
                reg_loss = reg_loss_func(reg_output.flatten().float(), reg_label_.flatten().float())
                #print(cls_loss.item(),reg_loss.item())
                loss=cls_loss + reg_loss*REG_LOSS_K

                f_loss.write('step loss: cls {}, reg {}\n'.format(cls_loss.item(),reg_loss.item()))

                total_loss.append(loss.item())
                total_loss_cls.append(cls_loss.item())
                total_loss_reg.append(reg_loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 50 == 0:
                    print(">>>>>loss:", sum(total_loss) / len(total_loss))

            f_loss.write('all loss epoch: {}\n'.format(sum(total_loss) / len(total_loss)))
            f_loss.write('cls loss epoch: {}\n'.format(sum(total_loss_cls) / len(total_loss_cls)))
            f_loss.write('reg loss epoch: {}\n'.format(sum(total_loss_reg) / len(total_loss_reg)))

            # testing
            model.eval()
            with torch.no_grad():
                y_cls = []
                y_reg = []
                val_cls_label = []
                val_reg_label = []
                
                for step, (hseq_feat_, hseq_pos_, lseq_feat_, lseq_pos_, rbd_feat_, rbd_pos_, cls_label_, reg_label_) in enumerate(val_loader_single):
                    #label_ = label_.to(device) #.cuda()
                    rbd_feat_=rbd_feat_.to(device)
                    hseq_feat_=hseq_feat_.to(device)
                    lseq_feat_=lseq_feat_.to(device)

                    rbd_pos_=rbd_pos_.to(device)
                    hseq_pos_=hseq_pos_.to(device)
                    lseq_pos_=lseq_pos_.to(device)
                    
                    cls_output, reg_output = model(hseq_feat_, hseq_pos_, lseq_feat_, lseq_pos_, rbd_feat_, rbd_pos_)#.cpu().data.numpy().squeeze()
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

                print('Epoch: {}, auc: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}\n'.format(epoch,auc_,vc_,precision_,recall_,f1_))
                f.write('fold: {}, Epoch: {}, auc: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}\n'.format(counter,epoch,auc_,vc_,precision_,recall_,f1_))

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
                    auc_best=auc_
                    vc_best=vc_
                    precision_best=precision_
                    recall_best=recall_
                    f1_best=f1_
                    cm_best=cm_

                    mse_best=mse_
                    corr_best=corr_

                    # save model
                    save_path = os.path.join(res_dir, 'model_{}.pth.tar'.format(counter))
                    torch.save({'state_dict': model.state_dict()}, save_path)
                    print('Epoch {}, mse best: {}\n'.format(epoch,mse_best))
                    f.write('Epoch {}, mse best: {}\n'.format(epoch,mse_best))

        f.write('\nfold {}, auc: {:.4f}, accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}\n'.format(counter,auc_best,vc_best,
                                                                                                                precision_best,recall_best,
                                                                                                                f1_best))
        f.write('fold {}, mse: {:.4f}, corr: {:.4f}\n'.format(counter,mse_best,corr_best))
        f.write('fold {}, \n[[{},{}],\n[{},{}]]\n\n\n'.format(counter,cm_best[0][0],cm_best[0][1],cm_best[1][0],cm_best[1][1]))

        test_precision_all.append(precision_best)
        test_recall_all.append(recall_best)
        test_f1_all.append(f1_best)
        test_auc_all.append(auc_best)
        test_vc_all.append(vc_best)
        test_mse_all.append(mse_best)
        test_corr_all.append(corr_best)
        f.close()

        counter+=1

    precision_mean=np.mean(test_precision_all)
    recall_mean=np.mean(test_recall_all)
    f1_mean=np.mean(test_f1_all)
    auc_mean=np.mean(test_auc_all)
    vc_mean=np.mean(test_vc_all)
    mse_mean=np.mean(test_mse_all)
    corr_mean=np.mean(test_corr_all)
    f = open(os.path.join(res_dir, 'result.txt'), 'a')
    f.write('total, auc: {:.4f}, accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}\n'.format(auc_mean,vc_mean,
                                                                                                            precision_mean,recall_mean,
                                                                                                            f1_mean))
    f.write('total, mse: {:.4f}, corr: {:.4f}\n'.format(mse_mean,corr_mean))
    f.write('end time: {}'.format(time.ctime()))

    f.close()
    f_loss.close()
