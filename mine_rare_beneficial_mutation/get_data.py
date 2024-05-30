import pickle
import numpy as np
import pandas as pd

ori='NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'

test_sites=[339,346,368,371,373,375,376,405,408,417,440,445,446,460,477,478,484,486,490,498,501,505]

if 0:
    def get_muts(seq):
        len_=len(seq)
        ret=[]
        for i in range(len_):
            if ori[i]!=seq[i]:
                return [ori[i],331+i,seq[i]]
        return [None,None,None]

    df=pd.read_csv('data/single_site_benchmark/bind/data/data_all.csv')
    seqs=df['sequence'].values

    mutations=[]
    for seq in seqs:
        mutations.append(get_muts(seq))

    mutations=np.array(mutations)
    print(mutations.shape)

    df['origin']=mutations[:,0]
    df['site']=mutations[:,1]
    df['mutation']=mutations[:,2]

    df.to_csv('data/mine_rare_beneficial_mutation/seq_full_data.csv',index=False)

    
# test data
if 1:
    df=pd.read_csv('data/mine_rare_beneficial_mutation/seq_full_data.csv')
    seqs=df['sequence'].values
    labels=df['label'].values

    test_muts = ['G339C', 'R346A', 'S371A', 'S373A', 'D405A', 'N440C', 'V445C', 'N460C', 'S477C', 'T478C', 'E484A', 'F490C', 'Q498A', 'N501C', 'Y505A']
    test_muts += ['G339D', 'R346D', 'S371D', 'S373M', 'D405S', 'N440A', 'V445K', 'N460A', 'S477D', 'T478K', 'E484K', 'F490A', 'Q498F', 'N501F', 'Y505W']
    print(len(test_muts))
    print(test_muts)
    #print(num)
    
    def is_test_data(seq):
        len_=len(seq)
        ret=[]
        for i in range(len_):
            if ori[i]!=seq[i]:
                ret.append('{}{}{}'.format(ori[i],331+i,seq[i]))
        if len(ret)==0:
            return False
        else:
            return ret[0] in test_muts
        
    test_mask=[]
    for seq in seqs:
        test_mask.append(is_test_data(seq))
    test_mask=np.array(test_mask).astype(bool)
    
    train_mask=((1-test_mask)*(labels!=1)).astype(bool)
    
    print(test_mask.sum(),train_mask.sum(),test_mask.sum()+train_mask.sum())
    
    
    df_=df[test_mask]
    muts=[]
    for i in range(len(df_)):
        muts.append('{}{}{}'.format(df_.iloc[i]['origin'],df_.iloc[i]['site'],df_.iloc[i]['mutation']))
    print(muts)
    print(df_)
    
if 0:
    data=pickle.load(open('<path to data>/data_all_embedding.pickle','rb'))
    data=np.array(data)

    train_data=data[train_mask]
    print(train_data.shape)
    np.save('<path to data>/mine_rare_train_data.npy',train_data)
    del train_data
    
    test_data=data[test_mask]
    print(test_data.shape)
    np.save('<path to data>/mine_rare_test_data.npy',test_data)
    del test_data
    del data
    
    train_label=labels[train_mask]
    print(train_label.shape)
    np.save('<path to data>/mine_rare_train_label.npy',train_label)
    
    test_label=labels[test_mask]
    print(test_label.shape)
    np.save('<path to data>/mine_rare_test_label.npy',test_label)
