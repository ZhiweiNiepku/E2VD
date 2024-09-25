import numpy as np
import pandas as pd


res_dict={'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
    
if 1:
    def seq2onehot(seq):
        ret=[]
        for res in seq:
            tp_=[0]*20
            tp_[res_dict[res]]=1
            ret.append(tp_)
            del tp_
        return np.array(ret)
    def df2data(df):
        seqs=df['sequence'].values
        label=df['label'].values
        
        datas=[]
        labels=[]
        
        for i in range(len(seqs)):
            datas.append(seq2onehot(seqs[i]))
            labels.append(label[i])
        return np.array(datas),np.array(labels)
    
    df=pd.read_csv('data/other_virus/influenza/data_train.csv')
    data,label=df2data(df)
    print(data.shape)
    print(label.shape)
    np.save('data/other_virus/influenza/CNN/onehot_train_data.npy',data)
    np.save('data/other_virus/influenza/CNN/onehot_train_label.npy',label)
    
    df=pd.read_csv('data/other_virus/influenza/data_test.csv')
    data,label=df2data(df)
    print(data.shape)
    print(label.shape)
    np.save('data/other_virus/influenza/CNN/onehot_test_data.npy',data)
    np.save('data/other_virus/influenza/CNN/onehot_test_label.npy',label)