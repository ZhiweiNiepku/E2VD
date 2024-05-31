import utils.OpenAttack as oa 
import torch
import datasets
from model import covid_prediction_model
import esm
import numpy as np
import torch.nn.functional as F
import time
import os
from utils.OpenAttack.attack_assist.substitute.word import ProteinBlosum62Substitute
import pandas as pd
from collections import defaultdict
import multiprocessing as mp
import time
from dateutil.parser import parse as dparse

dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("cutoff", help="The cutoff of the score",type=float)
parser.add_argument("input_file", help="The path of input file",type=str)
parser.add_argument("success_file_out",help="The output file path",type=str)
parser.add_argument("failed_file_out",help="The output file path",type=str)
args = parser.parse_args()
cutoff=args.cutoff
input_file=args.input_file
success_file_out=args.success_file_out
failed_file_out=args.failed_file_out



#esm_path=os.path.join(dir,"trained_model/esm1b_t33_650M_UR50S.pt")
model_path=os.path.join(dir,"trained_model/pytorch_model.bin")

class tokenizer:
    def __init__(self) :
        self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()[1]
        self.batch_converter = self.alphabet.get_batch_converter()
        
    def tokenize(self,seq):
        def add_id(x):
            return ('1',x)
        data=[add_id(item) for item in seq]
        _, _, batch_tokens = self.batch_converter(data)
        return batch_tokens

class ProteinClassifier(oa.Classifier):
    def __init__(self,model_path,cutoff):
        self.model=covid_prediction_model(freeze_bert=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(model_path,map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer=tokenizer()
        self.alphabet=self.tokenizer.alphabet
        embedding_layer=self.model.esm_model.embed_tokens
        self.embedding_layer=embedding_layer

        self.curr_embedding=None
        #self.hook = embedding_layer.register_forward_hook(oa.HookCloser(self) )
        self.embedding_layer = embedding_layer
        self.word2id = self.alphabet.tok_to_idx

        self.embedding = embedding_layer.weight.detach().cpu().numpy()

        self.cutoff=cutoff

    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=-1)
    
    def get_prob(self,input_):
        f=0
        input_=[seq.upper() for seq in input_]
        token =self.tokenizer.tokenize(input_)
        #token=token.to(self.device)
        t_class_prob=None
        for batch_token in torch.split(token,10):
            batch_token=batch_token.to(self.device)
            logits=self.model(batch_token,labels=None).logits
            class_score=F.softmax(logits,dim=-1)[:,:,1]
            class_score=torch.mean(class_score,dim=-1)
            class_score=np.array(class_score.cpu().detach(), dtype=float)
            class_score=[0.01 if x<=0.3 else x-0.3 for x in class_score]
            class_score=np.asarray([[1-x,x] for x in class_score])
            if f==0:
                t_class_prob=class_score
            else:
                t_class_prob=np.append(t_class_prob,class_score,axis=0)
            f+=1   
        return t_class_prob

def main():
    #print(mp.cpu_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    init_seq=pd.read_csv(input_file,index_col=0)
    
    init_seq=init_seq["seq"].to_list()
    n_seq=len(init_seq)
    print("seed sequences number:", n_seq)
    dataset = datasets.Dataset.from_dict({
        "x": init_seq,
        "y": [0]*n_seq
    })
    substitute=ProteinBlosum62Substitute(k=20)
    attacker = oa.attackers.GeneticAttacker(lang="protein",substitute=substitute)
    #attacker = oa.attackers.FDAttacker(lang="protein",substitute=substitute)
    victim = ProteinClassifier(model_path,cutoff=cutoff)
    attack_eval = oa.AttackEval(attacker, victim)
    # time_start=time.time()
    attack_eval.eval(dataset, visualize=True, progress_bar=True,s_file=success_file_out,f_file=failed_file_out)
    print("dataset:",device)
    # time_end=time.time()
    # print('time cost:',time_end-time_start,'s')

if __name__ == "__main__":
    main()

# g=victim.get_grad([dataset["x"]],[0])
# print(g)