import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm


RBD_emb=np.load('<path to data>/RBD_embedding_all_data.npy').astype(np.float16)
HSEQ_emb=np.load('<path to data>/antibody_heavy_sequence_embedding_all_data.npy').astype(np.float16)
LSEQ_emb=np.load('<path to data>/antibody_light_sequence_embedding_all_data.npy').astype(np.float16)

shape_=RBD_emb.shape
RBD_emb=RBD_emb.reshape(shape_[0],-1)
rbd_mean=np.mean(RBD_emb,axis=0)
rbd_std=np.std(RBD_emb,axis=0)

shape_=HSEQ_emb.shape
HSEQ_emb=HSEQ_emb.reshape(shape_[0],-1)
hseq_mean=np.mean(HSEQ_emb,axis=0)
hseq_std=np.std(HSEQ_emb,axis=0)

shape_=LSEQ_emb.shape
LSEQ_emb=LSEQ_emb.reshape(shape_[0],-1)
lseq_mean=np.mean(LSEQ_emb,axis=0)
lseq_std=np.std(LSEQ_emb,axis=0)




print(rbd_mean.shape)
print(rbd_std.shape)
print(hseq_mean.shape)
print(hseq_std.shape)
print(lseq_mean.shape)
print(lseq_std.shape)


np.save('<path to data>/RBD_embedding_flatten_mean.npy',rbd_mean)
np.save('<path to data>/RBD_embedding_flatten_std.npy',rbd_std)

np.save('<path to data>/HSEQ_embedding_flatten_mean.npy',hseq_mean)
np.save('<path to data>/HSEQ_embedding_flatten_std.npy',hseq_std)

np.save('<path to data>/LSEQ_embedding_flatten_mean.npy',lseq_mean)
np.save('<path to data>/LSEQ_embedding_flatten_std.npy',lseq_std)
