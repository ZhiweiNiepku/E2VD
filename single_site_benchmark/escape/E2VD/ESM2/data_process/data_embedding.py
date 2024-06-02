import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")

RBD = False
AB = False

if RBD:
    batch_converter = alphabet.get_batch_converter()
    model.eval().cuda()  # disables dropout for deterministic results

    # get data list

    df=pd.read_csv('data/single_site_benchmark/escape/RBD_sequences.csv')
    seqs=df['sequence'].values


    data=[]
    for i in range(len(seqs)):
        data.append(('seq{}'.format(i),seqs[i]))

    res=[]

    batch_size=8

    for idx in tqdm(range(0,len(seqs),batch_size)):
        data_=data[idx:idx+batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(data_)
        seq_lens = (batch_tokens != alphabet.padding_idx).sum(1)[0]
        #print(seq_lens)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens.cuda(), repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33][:,1:seq_lens-1,:].cpu().data.numpy()

        res.append(token_representations)
    res=np.concatenate(res,axis=0)

    print(res.shape)
    np.save('<path to data>/RBD_sequences_all_data.npy',res)


if AB:
    chain='heavy' # 'light'
    batch_converter = alphabet.get_batch_converter(padding_seq_length=136)
    model.eval().cuda()  # disables dropout for deterministic results

    # get data list

    df=pd.read_csv('data/single_site_benchmark/escape/antibody_{}_sequence.csv'.format(chain))
    seqs=df['sequence'].values


    data=[]
    for i in range(len(seqs)):
        data.append(('seq{}'.format(i),seqs[i]))

    res=[]

    batch_size=8

    for idx in tqdm(range(0,len(seqs),batch_size)):
        data_=data[idx:idx+batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(data_)
        seq_lens = (batch_tokens != alphabet.padding_idx).sum(1)[0]
        #print(seq_lens)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens.cuda(), repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33].cpu().data.numpy()

        res.append(token_representations)
    res=np.concatenate(res,axis=0)

    print(res.shape)
    np.save('<path to data>/antibody_{}_sequence_all_data.npy'.format(chain),res)