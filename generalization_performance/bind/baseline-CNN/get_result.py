import os
import re
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

targets=['Omicron_BA1','Omicron_BA2','Beta','Eta','Alpha','Delta']

target_seqs={
    'WT':          'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST',
    'Omicron_BA1': 'NITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNLAPFFTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVSGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLRSYSFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKST', 
    'Omicron_BA2': 'NITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLRSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKST', 
    'Beta':        'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVKGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKST', 
    'Eta':         'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVKGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST', 
    'Alpha':       'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKST', 
    'Delta':       'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYRYRLFRKSNLKPFERDISTEIYQAGSKPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'
    }

def replace(seq,rule):
    idx=int(re.findall(r'[A-Z]([0-9]*)[A-Z]',rule)[0])-331
    find=re.findall(r'([A-Z])([0-9]*)([A-Z])',rule)[0]
    ori,cha=find[0],find[2]
    if ori!=seq[idx]:
        raise ValueError
    seq=seq[:idx]+cha+seq[idx+1:]
    return seq

def cal_kd(delta):
    return math.exp(float(delta) * 4185.85 / ( 8.314 * 218) )



name='bind_onehot_test'

df=pd.read_csv('data/generalization_performance/bind/baseline-CNN/variant_data.csv')
reg_pred=np.load('predict_reg_all_{}.npy'.format(name))
cls_pred=np.load('predict_cls_all_{}.npy'.format(name))
cls_pred_bool=np.load('predict_cls_bool_all_{}.npy'.format(name))
df['predict_regression']=reg_pred.reshape(-1,1)
df['predict_classification']=cls_pred.reshape(-1,1)
df['predict_classification_bool']=cls_pred_bool.reshape(-1,1)

df.to_csv('variant_predict.csv',index=False)


df0=pd.read_csv('variant_predict.csv')
df0=df0.dropna(axis=0)

# OPP

# all
tp1=df0['KD_ratio'].values
tp2=df0['predict_regression'].values
tp3=df0['predict_classification'].values

tcounter=0
rcounter=0
ccounter=0
for i in tqdm(range(len(tp1))):
    for j in range(len(tp2)):
        if i<j:
            if (tp1[i]<tp1[j]) and (tp2[i]<tp2[j]):
                rcounter+=1
            elif (tp1[i]>=tp1[j]) and (tp2[i]>=tp2[j]):
                rcounter+=1

            if (tp1[i]<tp1[j]) and (tp3[i]<tp3[j]):
                ccounter+=1
            elif (tp1[i]>=tp1[j]) and (tp3[i]>=tp3[j]):
                ccounter+=1
            
            tcounter+=1

print('all, classification: {}, regression: {}'.format(ccounter/tcounter,rcounter/tcounter))

# lineages
targets=df0['target'].unique()

for target_ in targets:
    df0_=df0[df0['target']==target_]
    tp1=df0_['KD_ratio'].values
    tp2=df0_['predict_regression'].values
    tp3=df0_['predict_classification'].values

    tcounter=0
    rcounter=0
    ccounter=0
    for i in tqdm(range(len(tp1))):
        for j in range(len(tp2)):
            if i<j:
                if (tp1[i]<tp1[j]) and (tp2[i]<tp2[j]):
                    rcounter+=1
                elif (tp1[i]>=tp1[j]) and (tp2[i]>=tp2[j]):
                    rcounter+=1

                if (tp1[i]<tp1[j]) and (tp3[i]<tp3[j]):
                    ccounter+=1
                elif (tp1[i]>=tp1[j]) and (tp3[i]>=tp3[j]):
                    ccounter+=1
                
                tcounter+=1

    print('{}, classification: {}, regression: {}'.format(target_,ccounter/tcounter,rcounter/tcounter))

