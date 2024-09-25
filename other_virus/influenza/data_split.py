import os
import numpy as np
import pandas as pd

import random

random.seed(27)

# only keep missense mutation
if 1:
    # mmc2.xlsx is from the Table S1 of *Mutational fitness landscape of human influenza H3N2 neuraminidase*
    # https://www.cell.com/cell-reports/fulltext/S2211-1247(22)01852-6
    df = pd.read_excel('mmc2.xlsx')
    df = df[df['mutation_type'] == 'missense']

    print(df.shape)
    df.to_csv('data/other_virus/influenza/raw_data.csv', index=False)

    df = pd.read_csv('data/other_virus/influenza/raw_data.csv')

    pos_index_all = np.where(df['Fitness'].values > 0)[0].tolist()
    pos_idx = random.sample(pos_index_all, 100)
    neg_index_all = np.where(df['Fitness'].values < 0)[0].tolist()
    neg_idx = random.sample(neg_index_all, 100)

    test_idx = pos_idx + neg_idx

    train_idx = []
    for i in range(len(df)):
        if i not in test_idx:
            train_idx.append(i)

    print(len(train_idx))
    print(len(test_idx))

    np.save('data/other_virus/influenza/train_idx.npy', train_idx)
    np.save('data/other_virus/influenza/test_idx.npy', test_idx)

# get wt sequence
if 0:
    df = pd.read_csv('data/other_virus/influenza/raw_data.csv')
    pos_all = range(82, 466)
    wt_seq = ''
    for pos in pos_all:
        muts = df[df['Position'] == pos]['Mutation'].values
        res = muts[0][0]
        for mut in muts:
            assert mut[0] == res
        wt_seq += res
    
    with open('data/other_virus/influenza/wt_seq.txt', 'w') as f:
        f.write(wt_seq)

def mut2seq(wt, mut):
    ori, pos, mut = mut[0], int(mut[1:-1]), mut[-1]
    assert wt[pos-82] == ori
    seq = wt[:pos-82] + mut + wt[pos-81:]
    return seq

# get train data
if 1:
    df = pd.read_csv('data/other_virus/influenza/raw_data.csv')
    with open('data/other_virus/influenza/wt_seq.txt', 'r') as f:
        wt_seq = f.readline().strip()
    print(len(wt_seq))
    
    muts = df['Mutation'].values
    labels = df['Fitness'].values
    seqs = [mut2seq(wt_seq, mut) for mut in muts]
    
    df = pd.DataFrame()
    df['mutation'] = muts
    df['sequence'] = seqs
    df['label'] = labels
    
    print(df.shape)
    df.to_csv('data/other_virus/influenza/data_all.csv', index=False)
    
    
    train_idx = np.load('data/other_virus/influenza/train_idx.npy')
    test_idx = np.load('data/other_virus/influenza/test_idx.npy')
    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx]
    print(df_train.shape)
    print(df_test.shape)
    df_train.to_csv('data/other_virus/influenza/data_train.csv', index=False)
    df_test.to_csv('data/other_virus/influenza/data_test.csv', index=False)
