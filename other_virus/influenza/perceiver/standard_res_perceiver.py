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

from functools import wraps
from torch import einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Reduce, Rearrange
from torch.utils.data import Dataset, DataLoader

os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CUDA_LAUNCH_BLOCKING']='1'

device=torch.device('cuda')

# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

# helper classes Perceiver
class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x

class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)
        self.to_attn_logits = nn.Parameter(torch.eye(dim))

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = x[:,:,:-remainder]  # cut the last ones to be divisible for poolsize

        attn_logits = einsum('b d n, d e -> b e n', x, self.to_attn_logits)
        x = self.pool_fn(x)
        logits = self.pool_fn(attn_logits)

        attn = logits.softmax(dim = -1)
        return (x * attn).sum(dim = -1)

def ConvBlock(dim, dim_out = None, kernel_size = 1):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2)
    )

import math
def exponential_linspace_int(start, end, num, divisible_by = 1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]



class ProteinPerceiver_inner(nn.Module):
    def __init__(
            self,
            *,
            depth,
            dim,
            queries_dim,
            dropout=0.2,
            logits_dim=None,
            num_latents=512,
            latent_dim=512,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            weight_tie_layers=False,
            decoder_ff=False,
            self_per_cross_attn=1,
            final_perceiver_head=False,
            num_downsamples=3,
            att_pool_size=6,
            dim_divisible_by=2,
            num_tokens=21
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        latent_dim_head = latent_dim // latent_heads

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, dim, heads=cross_heads,
                                                               dim_head=cross_dim_head,dropout=dropout), context_dim=dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=dropout, mult=2))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head,
                                                                dropout=dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=dropout, mult=2))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff,
                                                                                      get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key=block_ind),
                    get_latent_ff(**cache_args, key=block_ind)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.final_perceiver_head = final_perceiver_head
        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, queries_dim)
        ) if final_perceiver_head else nn.Identity()

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads=cross_heads,
                                                                 dim_head=cross_dim_head), context_dim=latent_dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim, mult=2)) if decoder_ff else None

        half_dim = queries_dim // 2
        # create stem
        self.stem = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Conv1d(num_tokens, queries_dim, 15, padding=7),
            Residual(ConvBlock(queries_dim)),
            AttentionPool(queries_dim, pool_size=att_pool_size)
        )
        # create conv tower
        filter_list = exponential_linspace_int(queries_dim, half_dim, num=(num_downsamples - 1), divisible_by=dim_divisible_by)
        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size=3),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out, pool_size=att_pool_size//2)
            ))
        self.conv_tower = nn.Sequential(*conv_layers)

        self.flatten = nn.Flatten()
        self.dense1 = nn.LazyLinear(queries_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)

        out_scores = 1
        
        if final_perceiver_head:
            self.to_score_cls = nn.Sequential(
                nn.Linear(queries_dim * 2, out_scores),
                nn.Sigmoid()
            )
        else:
            self.to_score_cls = nn.Sequential(
                nn.Linear(queries_dim, out_scores),
                nn.Sigmoid()
            )
            
        self.to_score_reg = nn.Linear(queries_dim * 2, out_scores) \
            if final_perceiver_head else nn.Linear(queries_dim, out_scores)

    def forward(
            self,
            data,
            mask=None,
            queries=None
    ):
        b, *_, device = *data.shape, data.device

        x = repeat(self.latents, 'n d -> b n d', b=b)

        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context = data, mask = mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        queries = self.stem(queries)
        queries = self.conv_tower(queries)
        queries = self.flatten(queries)
        queries = self.dropout1(self.relu1(self.dense1(queries)))
        

        logits = self.decoder_cross_attn(queries.unsqueeze(1), context=x).squeeze(1) + queries

        if self.final_perceiver_head:
            x = self.to_logits(x)
            logits = torch.cat([logits, x], dim=-1)

        return self.to_score_cls(logits), self.to_score_reg(logits)


class ProteinPerceiver(nn.Module):
    def __init__(
            self,
            *,
            dim,
            num_tokens,
            max_seq_len,
            pos_emb,
            activation,
            dropout,
            depth,
            queries_dim,
            att_pool_size,
            **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.dropout_emb = nn.Dropout(p=dropout)

        # if pos_emb == 'sin_fix':
        #     self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_seq_len, dim, 0), freeze=True)
        # elif pos_emb == 'sin_learned':
        #     self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_seq_len, dim, 0), freeze=False)
        # else:
        #     self.pos_emb = nn.Embedding(max_seq_len, dim)
        
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.perceiver = ProteinPerceiver_inner(
                depth=depth,
                dim=dim,
                queries_dim=queries_dim,
                dropout=dropout,
                att_pool_size=att_pool_size,
                num_tokens=num_tokens,
                **kwargs
            )
        self.activation = activation
        self.max_len = max_seq_len

    def forward(self, x_seq, x_bool, mask=None):

        x_seq = x_seq[:, :self.max_len]
        x_bool = x_bool[:, :self.max_len, :]

        n, device = x_seq.shape[1], x_seq.device
        x = self.token_emb(x_seq)
        x = self.dropout_emb(x)

        pos_emb = self.pos_emb(torch.arange(n, device=device))
        pos_emb = rearrange(pos_emb, 'n d -> () n d')
        x = x + pos_emb

        mask = x_seq != 0

        score = self.perceiver(x, mask=mask, queries=x_bool.float()) # queries=x

        if self.activation == 'tanh':
            return torch.tanh(score[0].squeeze(1)), torch.tanh(score[1].squeeze(1))
        else:
            return score[0].squeeze(1), score[1].squeeze(1)


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


class ProteinSeqDataset(Dataset):
    def __init__(self, df):
        self.seqs = df[:,1]
        self.reg_labels = torch.from_numpy(df[:,2].astype(float))
        #print(torch.histogram(self.reg_labels))
        self.cls_labels = (self.reg_labels >= 0).float()
        
        self.map_dict = {'L': 0, 'A': 1, 'G': 2, 'V': 3, 'E': 4, 'S': 5, 'I': 6, 
                        'K': 7, 'R': 8, 'D': 9, 'T': 10, 'P': 11, 'N': 12, 'Q': 13,
                        'F': 14, 'Y': 15, 'M': 16, 'H': 17, 'C': 18, 'W': 19, 'X': 20}
        
        self.tokens = []
        for seq in self.seqs:
            self.tokens.append(torch.tensor([self.map_dict[c] for c in seq]))
        
    def __len__(self):
        return len(self.reg_labels)
    
    def __getitem__(self, idx):
        return self.tokens[idx], F.one_hot(self.tokens[idx], len(self.map_dict)), \
               self.cls_labels[idx], self.reg_labels[idx]







if __name__ == '__main__':
    BATCH_SIZE = 200
    EPOCH = 150
    backbone_name = 'perceiver' #"T5-XL-UNI" 'BERT' 'T5-XL-UNI'
    SEQ_LEN = 384
    REG_LOSS_K=1

    lr=2e-5

    cls_loss_func=nn.BCELoss() # ClsFocalLoss()
    reg_loss_func=nn.MSELoss() # RegFocalLoss()

    k=5
    
    test_vc_all =[]
    test_precision_all=[]
    test_recall_all=[]
    test_f1_all=[]
    test_auc_all=[]
    test_mse_all=[]
    test_corr_all=[]

    res_dir = "result_{}_lr{}_regk{}_k{}_f1_prop_".format(BATCH_SIZE,lr,REG_LOSS_K,k) + backbone_name
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    ##############
    # load data

    df_all=pd.read_csv('data/other_virus/influenza/data_train.csv').values
    print(df_all.shape)
    
    cls_label = df_all[:,2]>=0
    print(cls_label.shape)

    #kf=KFold(n_splits=k,shuffle=True,random_state=27)
    #kf_idx=kf.split(feats,reg_label)
    kf_idx=MyFold(cls_label,k,70)

    counter=0
    
    params = dict(
        letter_emb_size = 32,
        num_tokens = 21, 
        max_len = SEQ_LEN,
        depth = 1,
        num_latents = 128,
        latent_dim = 128,
        pos_emb = 'learned',
        activation = 'None',
        dropout = 0.2,
        queries_dim = 64,
        final_perceiver_head = True,
        att_pool_size = 10
    )

    #for i_idx in kf_idx:
    for i_ in range(k):
        i_idx=kf_idx[i_]

        f = open(os.path.join(res_dir, 'result.txt'), 'a')
        print('fold',counter)
        f.write('fold {}, {}\n'.format(counter,time.ctime()))
        
        vc_best=0
        precision_best=0
        recall_best=0
        f1_best=0
        auc_best=0
        cm_best=None

        mse_best=20
        corr_best=0

        model = ProteinPerceiver(dim=params['letter_emb_size'], num_tokens=params['num_tokens'], 
                                max_seq_len=params['max_len'], depth=params['depth'],
                                num_latents=params['num_latents'], latent_dim=params['latent_dim'], 
                                pos_emb=params['pos_emb'], activation=params['activation'], 
                                dropout=params['dropout'], queries_dim=params['queries_dim'],
                                final_perceiver_head=params['final_perceiver_head'], 
                                att_pool_size=params['att_pool_size'])
        optimizer = torch.optim.Adam(model.parameters(),lr)
        model = model.to(device)

        df_train=df_all[i_idx[0]]
        df_val=df_all[i_idx[1]]

        Train_dataset = ProteinSeqDataset(df_train)
        train_loader_single = DataLoader(dataset=Train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        Val_dataset = ProteinSeqDataset(df_val)
        val_loader_single = DataLoader(dataset=Val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # training and testing
        for epoch in range(EPOCH):
            total_loss = []

            # training
            model.train()
            for step, (feat_, bool_, cls_label_, reg_label_) in enumerate(train_loader_single):
                cls_label_ = cls_label_.to(device)
                reg_label_ = reg_label_.to(device)
                feat_ = feat_.to(device)
                bool_ = bool_.to(device)
                cls_output, reg_output = model(feat_, bool_)
                cls_output=cls_output.squeeze()
                reg_output=reg_output.squeeze()


                cls_loss = cls_loss_func(cls_output.flatten().float(), cls_label_.float())
                reg_loss = reg_loss_func(reg_output.flatten().float(), reg_label_.float())
                #print(cls_loss.item(),reg_loss.item())
                loss=cls_loss + reg_loss*REG_LOSS_K
                total_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                #accelerator.backward()
                optimizer.step()
                if step % 40 == 0:
                    #accelerator.print(">>>>>loss:", sum(total_loss) / len(total_loss))
                    print(">>>>>loss:", sum(total_loss) / len(total_loss))

            # testing
            model.eval()
            with torch.no_grad():
                y_cls = []
                y_reg = []
                
                val_cls_label = []
                val_reg_label = []
                
                for step, (feat_, bool_, cls_label_, reg_label_) in enumerate(val_loader_single):
                    #label_ = label_.to(device) #.cuda()
                    feat_=feat_.to(device) #.cuda()
                    bool_=bool_.to(device)

                    cls_output, reg_output = model(feat_, bool_)#.cpu().data.numpy().squeeze()
                    cls_output=cls_output.cpu().data.numpy().squeeze()
                    reg_output=reg_output.cpu().data.numpy().squeeze()
                    
                    y_cls.append(cls_output.reshape(-1,1))
                    y_reg.append(reg_output.reshape(-1,1))
                    
                    val_cls_label.append(cls_label_.cpu().data.numpy().reshape(-1,1))
                    val_reg_label.append(reg_label_.cpu().data.numpy().reshape(-1,1))

                val_cls_label = np.concatenate(val_cls_label, axis=0)
                val_reg_label = np.concatenate(val_reg_label, axis=0)

                # metrics for classification
                y_cls = np.concatenate(y_cls,axis=0).flatten()
                y_cls_bool=(y_cls>0.5)
                val_cls_label=val_cls_label.flatten().astype(bool)
                cm_=confusion_matrix(val_cls_label,y_cls_bool)
                print(cm_)
                vc_=(cm_[0][0]+cm_[1][1])/(cm_[0][0]+cm_[0][1]+cm_[1][0]+cm_[1][1])
                precision_=precision_score(val_cls_label,y_cls_bool) #cm_[1][1]/(cm_[0][1]+cm_[1][1])
                recall_=recall_score(val_cls_label,y_cls_bool) #cm_[1][1]/(cm_[1][0]+cm_[1][1])
                f1_=f1_score(val_cls_label,y_cls_bool)
                auc_=roc_auc_score(val_cls_label,y_cls)

                #accelerator.print('Epoch: {}, auc: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}\n'.format(epoch,auc_,vc_,precision_,recall_,f1_))
                print('Epoch: {}, auc: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}\n'.format(epoch,auc_,vc_,precision_,recall_,f1_))
                f.write('fold: {}, Epoch: {}, auc: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}\n'.format(counter,epoch,auc_,vc_,precision_,recall_,f1_))

                # metrics for regression
                y_reg = np.concatenate(y_reg,axis=0).flatten()
                val_reg_label=val_reg_label.flatten()
                mse_=mean_squared_error(val_reg_label,y_reg)
                corr_=np.corrcoef(val_reg_label,y_reg)[0][1]

                #accelerator.print('Epoch: {}, mse: {:.3f}, corr: {:.3f}\n'.format(epoch,mse_,corr_))
                print('Epoch: {}, mse: {:.3f}, corr: {:.3f}\n'.format(epoch,mse_,corr_))
                #if accelerator.is_local_main_process:
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
                    #accelerator.wait_for_everyone()
                    save_path = os.path.join(res_dir, 'model_{}.pth.tar'.format(counter))
                    #torch.save({'state_dict': accelerator.unwrap_model(model).state_dict()}, save_path)
                    torch.save({'state_dict': model.state_dict()}, save_path)
                    print('Epoch {}, mse best: {}\n'.format(epoch,mse_best))
                    #if accelerator.is_local_main_process:
                    f.write('Epoch {}, mse best: {}\n'.format(epoch,mse_best))

        if True: #accelerator.is_local_main_process:
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
