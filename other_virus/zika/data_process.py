import os
import numpy as np
import pandas as pd

import random

os.chdir(os.path.dirname(__file__))

if 0:
    # jvi.01291-19-sd003.xlsx is from the Supplemental Material of *Deep Mutational Scanning Comprehensively Maps How Zika Envelope Protein Mutations Affect Viral Growth and Antibody Escape*
    # https://journals.asm.org/doi/full/10.1128/jvi.01291-19
    df = pd.read_excel('jvi.01291-19-sd003.xlsx', sheet_name='mutational effects')
    df = df[df['log2effect'] != 0]

    print(df.head())
    print(df.shape)
    df.to_csv('data/other_virus/zika/raw_data.csv', index=False)

    df = pd.read_csv('data/other_virus/zika/raw_data.csv')

    labels = df['log2effect'].values
    pos_idx = random.sample(np.where(labels>0)[0].tolist(), 50)
    neg_idx = random.sample(np.where(labels<0)[0].tolist(), 50)
    test_idx = pos_idx+neg_idx
    train_idx = []
    for i in range(len(df)):
        if i not in test_idx:
            train_idx.append(i)

    np.save('data/other_virus/zika/train_idx.npy', train_idx)
    np.save('data/other_virus/zika/test_idx.npy', test_idx)

# get wt sequence
if 0:
    df = pd.read_csv('data/other_virus/zika/raw_data.csv')
    pos_all = range(1, 505)
    wt_seq = ''
    for pos in pos_all:
        muts = df[df['site'] == pos]['wildtype'].values
        res = muts[0]
        assert (muts==res).all()
        wt_seq += res
    
    with open('data/other_virus/zika/wt_seq.txt', 'w') as f:
        f.write(wt_seq)

def mut2seq(wt, mut):
    ori, pos, mut = mut[0], int(mut[1:-1]), mut[-1]
    assert wt[pos-1] == ori
    seq = wt[:pos-1] + mut + wt[pos:]
    return seq

# get train data
if 1:
    df = pd.read_csv('data/other_virus/zika/raw_data.csv')
    with open('data/other_virus/zika/wt_seq.txt', 'r') as f:
        wt_seq = f.readline().strip()
    print(len(wt_seq))
    
    muts = df['mutation'].values
    labels = df['log2effect'].values
    seqs = [mut2seq(wt_seq, mut) for mut in muts]
    
    df = pd.DataFrame()
    df['mutation'] = muts
    df['sequence'] = seqs
    df['label'] = labels
    
    print(df.shape)
    df.to_csv('data/other_virus/zika/data_all.csv', index=False)
    
    
    train_idx = np.load('data/other_virus/zika/train_idx.npy')
    test_idx = np.load('data/other_virus/zika/test_idx.npy')
    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx]
    print(df_train.shape)
    print(df_test.shape)
    df_train.to_csv('data/other_virus/zika/data_train.csv', index=False)
    df_test.to_csv('data/other_virus/zika/data_test.csv', index=False)