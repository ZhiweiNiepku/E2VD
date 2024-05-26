import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES']='0'

model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")


batch_converter = alphabet.get_batch_converter()
model.eval().cuda()  # disables dropout for deterministic results

# get data list

seq = 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'


data=[('seq',seq)]


batch_labels, batch_strs, batch_tokens = batch_converter(data)
seq_lens = (batch_tokens != alphabet.padding_idx).sum(1)[0]

# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(batch_tokens.cuda(), repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33][:,1:seq_lens-1,:].cpu().data.numpy()
#token_representations = results["representations"][33].cpu().data.numpy()

print(token_representations.shape)

np.save('RBD_ori_embedding.npy',token_representations)
