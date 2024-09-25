import os
import numpy as np
import pandas as pd

import random

os.chdir(os.path.dirname(__file__))

# wt sequence
if 0:
    # elife-34420-fig4-data2-v3.csv is from *Mapping mutational effects along the evolutionary landscape of HIV envelope*
    # https://elifesciences.org/articles/34420
    res_all = pd.read_csv('elife-34420-fig4-data2-v3.csv')['wildtype'][29:699].values

    wt_seq = ''.join(res_all.tolist())

    print(wt_seq)
    print(len(wt_seq))

    with open('data/other_virus/HIV/wt_seq.txt', 'w') as f:
        f.write(wt_seq)

if 0:
    # elife-34420-fig4-data1-v3.csv is from *Mapping mutational effects along the evolutionary landscape of HIV envelope*
    # https://elifesciences.org/articles/34420
    df = pd.read_csv('elife-34420-fig4-data1-v3.csv')

    with open('data/other_virus/HIV/wt_seq.txt', 'r') as f:
        wt_seq = f.readline().strip()

    muts = []
    labels = []

    cols = list(df.columns)[1:]

    for i in range(len(df)):
        ori = df[wt_seq[i]].iloc[i]
        for j in range(len(cols)):
            muts.append('{}{}{}'.format(wt_seq[i], i+1, cols[j]))
            labels.append(df[cols[j]].iloc[i] / ori)

    df = pd.DataFrame()
    df['mutation'] = muts
    df['ratio'] = labels
    df['log2'] = np.log2(labels)

    df.to_csv('data/other_virus/HIV/raw_data.csv', index=False)


if 1:
    df = pd.read_csv('data/other_virus/HIV/raw_data.csv')

    labels = df['log2'].values
    pos_idx = random.sample(np.where(labels>0)[0].tolist(), 100)
    neg_idx = random.sample(np.where(labels<0)[0].tolist(), 100)
    test_idx = pos_idx+neg_idx
    train_idx = []
    for i in range(len(df)):
        if i not in test_idx:
            train_idx.append(i)

    np.save('data/other_virus/HIV/train_idx.npy', train_idx)
    np.save('data/other_virus/HIV/test_idx.npy', test_idx)

def mut2seq(wt, mut):
    ori, pos, mut = mut[0], int(mut[1:-1]), mut[-1]
    assert wt[pos-1] == ori
    seq = wt[:pos-1] + mut + wt[pos:]
    return seq

# get train data
if 1:
    df = pd.read_csv('data/other_virus/HIV/raw_data.csv')
    with open('data/other_virus/HIV/wt_seq.txt', 'r') as f:
        wt_seq = f.readline().strip()
    print(len(wt_seq))
    
    muts = df['mutation'].values
    labels = df['log2'].values
    seqs = [mut2seq(wt_seq, mut) for mut in muts]
    
    df = pd.DataFrame()
    df['mutation'] = muts
    df['sequence'] = seqs
    df['label'] = labels
    
    print(df.shape)
    df.to_csv('data/other_virus/HIV/data_all.csv', index=False)
    
    
    train_idx = np.load('data/other_virus/HIV/train_idx.npy')
    test_idx = np.load('data/other_virus/HIV/test_idx.npy')
    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx]
    print(df_train.shape)
    print(df_test.shape)
    df_train.to_csv('data/other_virus/HIV/data_train.csv', index=False)
    df_test.to_csv('data/other_virus/HIV/data_test.csv', index=False)