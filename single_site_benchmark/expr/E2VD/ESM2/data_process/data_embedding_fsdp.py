import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap

import esm

os.environ['CUDA_LAUNCH_BLOCKING']='1'

url = "tcp://localhost:23456"
torch.distributed.init_process_group(backend="nccl", init_method=url, world_size=1, rank=0)


# initialize the model with FSDP wrapper
fsdp_params = dict(
    mixed_precision=True,
    flatten_parameters=True,
    state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
    cpu_offload=True,  # enable cpu offloading
)
with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
    model, vocab = torch.hub.load("facebookresearch/esm:main", "esm2_t48_15B_UR50D")
    batch_converter = vocab.get_batch_converter()
    model.eval()

    # Wrap each layer in FSDP separately
    for name, child in model.named_children():
        if name == "layers":
            for layer_name, layer in child.named_children():
                wrapped_layer = wrap(layer)
                setattr(child, layer_name, wrapped_layer)
    model = wrap(model)


model.eval()




# get data list
df=pd.read_csv('data/single_site_benchmark/expr/expr_4k_all.csv')
seqs=df['sequence'].values
labels=df['label'].values

data=[]
for i in range(len(seqs)):
    data.append(('seq{}'.format(i),seqs[i]))


res=[]

batch_size=8

for idx in tqdm(range(0,len(seqs),batch_size)):
    #print('idx',idx)
    data_=data[idx:idx+batch_size]
    #print('len',len(data_))
    batch_labels, batch_strs, batch_tokens = batch_converter(data_)
    seq_lens = (batch_tokens != vocab.padding_idx).sum(1)[0]
    #print(seq_lens)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens.cuda(), repr_layers=[48], return_contacts=True)
    token_representations = results["representations"][48][:,1:seq_lens-1,:].cpu().data.numpy()

    res.append(token_representations)

res=np.concatenate(res,axis=0)

print(res.shape)
np.save('<path to data>/single_expr_esm2_all_data.npy',res)
np.save('<path to data>/single_expr_esm2_all_label.npy',labels)