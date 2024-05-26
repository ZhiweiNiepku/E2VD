import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5EncoderModel
from torch.utils.data import DataLoader
import re
import argparse
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import DataLoader
import random
from sklearn import preprocessing
import pickle
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import os
import scipy
from sklearn.preprocessing import StandardScaler

def create_model(backbone_name="Rostlab/prot_bert"):
    path = "pretrained_model/"
    if backbone_name == 'T5-XL-UNI':
        pretrained_name = "Rostlab/prot_t5_xl_uniref50"
        pretrained_path = os.path.join(path, pretrained_name)
        print(pretrained_path)
        tokenizer = T5Tokenizer.from_pretrained(pretrained_path, do_lower_case=False)
        embedding_size = 1024 * 4
        try:
            embed_backbone = T5EncoderModel.from_pretrained(pretrained_path)
        except:
            embed_backbone = T5EncoderModel.from_pretrained(pretrained_name)
            embed_backbone.save_pretrained(pretrained_path)
    return embed_backbone, tokenizer, embedding_size


class ScaleDataset(torch.utils.data.Dataset):
    def __init__(self,ids,mask):
        self.ids = torch.from_numpy(ids)#.float()
        self.mask = torch.from_numpy(mask)#.float()

    def __getitem__(self, index):
        return self.ids[index], self.mask[index]
    def __len__(self):
        return len(self.ids)

class SelfDataset(torch.utils.data.Dataset):
    def __init__(self,feat,label):
        self.feat = torch.tensor(feat)
        self.labels = torch.from_numpy(label).float()

    def __getitem__(self, index):
        return self.feat[index], self.labels[index]
    def __len__(self):
        return len(self.labels)

def seq2token(data):
    data=data.upper()
    data = " ".join("".join(data.split()))
    data = re.sub(r"[UZOB]", "X", data)
    return data

if __name__=="__main__":

    BATCH_SIZE = 32
    backbone_name = 'T5-XL-UNI' #'BERT' #'T5-XL-UNI' #"T5-XL-UNI" 'BERT' 'T5-XL-UNI'
    device=torch.device('cuda')

    # load model
    embed_backbone, tokenizer, embedding_size = create_model(backbone_name=backbone_name)
    embed_backbone = embed_backbone.to(device) #.cuda()



    # train data
    train_data=pd.read_csv('data/single_site_benchmark/bind/data/data_train.csv')
    X_train=train_data['sequence'].apply(seq2token).values
    y_train=train_data['label'].values
    inputs=tokenizer.batch_encode_plus(X_train, add_special_tokens=True)
    
    Train_dataset=ScaleDataset(np.array(inputs['input_ids']),np.array(inputs['attention_mask']))
    train_loader = DataLoader(dataset=Train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    embedding_train=[]

    embed_backbone.eval()
    with torch.no_grad():
        for step, (ids, mask) in enumerate(train_loader):
            ids=ids.to(device) #.cuda()
            mask=mask.to(device)
            word_embeddings = embed_backbone(ids, mask)[0]
            word_embeddings=word_embeddings.cpu().data.numpy()
            embedding_train.append(word_embeddings)


    embedding_train=np.concatenate(embedding_train,axis=0)

    print(embedding_train.shape)

    np.save('<path to save data>/single_bind_t5_train_data.npy',embedding_train)
    np.save('<path to save data>/single_bind_t5_train_label.npy',y_train)


    # test data
    test_data=pd.read_csv('data/single_site_benchmark/bind/data/data_test.csv')
    X_test=test_data['sequence'].apply(seq2token).values
    y_test=test_data['label'].values
    inputs=tokenizer.batch_encode_plus(X_test, add_special_tokens=True)
    
    Test_dataset=ScaleDataset(np.array(inputs['input_ids']),np.array(inputs['attention_mask']))
    test_loader = DataLoader(dataset=Test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    embedding_test=[]

    embed_backbone.eval()
    with torch.no_grad():
        for step, (ids, mask) in enumerate(test_loader):
            ids=ids.to(device) #.cuda()
            mask=mask.to(device)
            word_embeddings = embed_backbone(ids, mask)[0]
            word_embeddings=word_embeddings.cpu().data.numpy()
            embedding_test.append(word_embeddings)


    embedding_test=np.concatenate(embedding_test,axis=0)

    print(embedding_test.shape)

    np.save('<path to save data>/single_bind_t5_test_data.npy',embedding_test)
    np.save('<path to save data>/single_bind_t5_test_label.npy',y_test)