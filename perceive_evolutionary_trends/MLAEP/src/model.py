import torch
from torch._C import device
import torch.nn as nn
import os
import sys
import esm
from transformers.file_utils import ModelOutput
from typing import Optional
import pandas as pd
import numpy as np
dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(dir) 
from utils.struct2vec_utils import StructEmbed
from utils.struct2vec_utils import StructureDataset
from utils.struct2vec_utils import featurize
import os


class output(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: tuple = None
    embedding: Optional[torch.FloatTensor] =None


class ClassPredictionHead(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 droupout: float=0.1) -> None:
        super().__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(in_dim,hid_dim,bias=True),
            nn.ReLU(),
            nn.Dropout(droupout,inplace=True),
            nn.Linear(hid_dim,out_dim)
            )
   
    def forward(self,pooled_output):
        value_pred = self.fc_layer(pooled_output)
        outputs = value_pred
        return(outputs)

class covid_prediction_model(nn.Module):
    def __init__(self,
                 jsonl_path: str=os.path.join(dir,'data/merged_all.jsonl'),
                 freeze_bert: bool=False,
                 seq_embedding_size: int=1280,
                 stru_embedding_size: int=5,
                 stru_seq_len:int=130,
                 dropout_prob: float=0.1,
                 pooling: str="first_token") -> None:
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.esm_model = esm.pretrained.esm1b_t33_650M_UR50S()[0].to(device)
        self.model_name ="esm1b_t33_650M_UR50S"
        self.model_layers = self.model_name.split("_")[1][1:]
        self.repr_layers = int(self.model_layers)
        self.predict_class = ClassPredictionHead(seq_embedding_size+stru_embedding_size*stru_seq_len*2,512,2,dropout_prob).to(device)
        self.pooling = pooling
        self.structEmbed = StructEmbed(node_features=128,edge_features=128,hidden_dim=128,out_dim=stru_embedding_size).to(device)
        self.antibody_data = StructureDataset(jsonl_file=jsonl_path, truncate=None, max_length=500)
        X, S, mask, lengths = featurize(self.antibody_data, device=device, shuffle_fraction=0)
        self.X=X
        self.S=S
        self.mask=mask
        self.lengths=lengths

        if freeze_bert:
            for p in self.esm_model.parameters():
                p.requires_grad = False
        
    def forward(self,input_ids,labels):
        outputs = self.esm_model(input_ids, repr_layers=[self.repr_layers])["representations"][self.repr_layers]
        b=outputs.shape[0]
        device=outputs.device
        if self.pooling == "first_token":
            outputs = outputs[:,0,:]
        elif self.pooling == "average" :
            outputs = outputs.mean(1)
        else:
            ValueError("Not implemented! Please choose the pooling methods from first_token/average")
        outputs = torch.repeat_interleave(outputs.unsqueeze(dim=1),repeats=9,dim=1)

        Struct_embedding = self.structEmbed(self.X,self.S,self.lengths,self.mask)
        Struct_embedding = torch.cat((Struct_embedding[0::2,:,:],Struct_embedding[1::2,:,:]),dim=1)
        Struct_embedding = torch.flatten(Struct_embedding,start_dim=1)

        Struct_embedding = torch.repeat_interleave(Struct_embedding.unsqueeze(dim=0),repeats=b,dim=0)

        combined_embedding = torch.cat((Struct_embedding,outputs),dim=2)

        class_logits = self.predict_class(combined_embedding)
        loss = None
        
        if labels is not None:
            pos_weight=torch.tensor([[0.08,0.92],[0.18,0.82],[0.06,0.94],[0.1,0.9],[0.08,0.92],[0.1,0.9],[0.05,0.95],[0.04,0.96],[0.17,0.83]],device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            if labels.shape[1]==10:
                labels=labels[:,0:9,:]
            loss = criterion(class_logits,labels)
        return(output(
        loss = loss,
        logits = class_logits,
        embedding = combined_embedding
        ))

    
class covid_prediction_model_without_GCN(nn.Module):
    def __init__(self,
                 freeze_bert: bool=False,
                 esm_path: str="./model",
                 seq_embedding_size: int=1280,
                 dropout_prob: float=0.1,
                 pos_weight:torch.tensor=torch.tensor([0.1,0.9]),
                 pooling: str="first_token") -> None:
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.esm_model = esm.pretrained.load_model_and_alphabet_local(esm_path)[0].to(device)
        self.model_name = os.path.basename(esm_path)
        self.model_layers = self.model_name.split("_")[1][1:]
        self.repr_layers = int(self.model_layers)
        self.predict_class = ClassPredictionHead(seq_embedding_size,512,2,dropout_prob).to(device)
        self.pooling = pooling
        self.pos_weight=pos_weight
        if freeze_bert:
            for p in self.esm_model.parameters():
                p.requires_grad = False
    def forward(self,input_ids,labels):
        outputs = self.esm_model(input_ids, repr_layers=[self.repr_layers])["representations"][self.repr_layers]
        b=outputs.shape[0]
        device=outputs.device
        if self.pooling == "first_token":
            outputs = outputs[:,0,:]
        elif self.pooling == "average" :
            outputs = outputs.mean(1)
        else:
            ValueError("Not implemented! Please choose the pooling methods from first_token/average")
        class_logits = self.predict_class(outputs) 
        loss = None
        if labels is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            loss = criterion(class_logits,labels)
        return(output(
        loss = loss,
        logits = class_logits,
        embedding = outputs
        ))


class covid_prediction_model_without_GCN_add_noise(nn.Module):
    def __init__(self,
                 freeze_bert: bool=False,
                 esm_path: str="./model",
                 seq_embedding_size: int=1280,
                 dropout_prob: float=0.1,
                 pooling: str="first_token") -> None:
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.esm_model = esm.pretrained.load_model_and_alphabet_local(esm_path)[0].to(device)
        self.model_name = os.path.basename(esm_path)
        self.model_layers = self.model_name.split("_")[1][1:]
        self.repr_layers = int(self.model_layers)
        self.predict_class = ClassPredictionHead(seq_embedding_size+1300,512,2,dropout_prob).to(device)
        self.pooling = pooling
        if freeze_bert:
            for p in self.esm_model.parameters():
                p.requires_grad = False
    def forward(self,input_ids,labels):
        outputs = self.esm_model(input_ids, repr_layers=[self.repr_layers])["representations"][self.repr_layers]
        b=outputs.shape[0]
        device=outputs.device
        if self.pooling == "first_token":
            outputs = outputs[:,0,:]
        elif self.pooling == "average" :
            outputs = outputs.mean(1)
        else:
            ValueError("Not implemented! Please choose the pooling methods from first_token/average")
        outputs = torch.repeat_interleave(outputs.unsqueeze(dim=1),repeats=9,dim=1)
        torch.manual_seed(3)
        noise=torch.randn((b,9,1300),device=device)
        combined_embedding = torch.cat((noise,outputs),dim=2)
        class_logits = self.predict_class(combined_embedding) 
        loss = None
        if labels is not None:
            pos_weight=torch.tensor([[0.08,0.92],[0.18,0.82],[0.06,0.94],[0.1,0.9],[0.08,0.92],[0.1,0.9],[0.05,0.95],[0.04,0.96],[0.17,0.83]],device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            if labels.shape[1]==10:
                labels=labels[:,0:9,:]
            loss = criterion(class_logits,labels)
        return(output(
        loss = loss,
        logits = class_logits,
        embedding = combined_embedding
        ))



