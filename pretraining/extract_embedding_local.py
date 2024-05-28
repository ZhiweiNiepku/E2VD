# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Bert evaluation script.
"""
import numpy as np
import mindspore as ms
from src.model_utils.device_adapter import get_device_id, get_device_num
import pickle
import argparse
from mindspore.common.tensor import Tensor
import os
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.model_utils.config import config as cfg, bert_net_cfg
from src.bert_model import BertModel
import tqdm
import glob
import pandas as pd
import numpy as np

map_dict = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'L': 5, 'A': 6, 'G': 7, 'V': 8,
            'E': 9, 'S': 10, 'I': 11, 'K': 12, 'R': 13, 'D': 14, 'T': 15, 'P': 16, 'N': 17, 'Q': 18,
            'F': 19, 'Y': 20, 'M': 21, 'H': 22, 'C': 23, 'W': 24, 'X': 25}


def get_embedding(seqs, batch_size):
    output = []
    batch = list()
    for i, seq in tqdm.tqdm(enumerate(seqs)):
        batch.append(seq)
        if len(batch) == batch_size or i == len(seqs) - 1:
            input_id, input_mask = get_ids(batch)
            print("Shape of input_id:", input_id.shape)
            sequence_output, encoder_output = net(input_id, input_mask)
            embedding = sequence_output.asnumpy()
            for i in range(embedding.shape[0]):
                # seq_len = np.count_nonzero(input_id[i].asnumpy())
                # output.append(embedding[i][1:seq_len - 1])
                output.append(embedding[i][1:-1])
            batch = list()
    return output



def get_ids(batch):
    lengths = [len(seq) for seq in batch]
    # max_length = max(lengths) + 2
    max_length = 203
    input_ids = []
    input_masks = []
    for seq in batch:
        seq_new = ''
        for i in range(len(seq)):
            if seq[i] not in map_dict.keys():
                seq_new += 'X'
            else:
                seq_new += seq[i]
        seq = seq_new
        seq = [map_dict['[CLS]']] + [map_dict[x] for x in seq] + [map_dict['[SEP]']]
        seq = np.array(seq)
        input_id = np.pad(seq.astype(np.int32), (0, max_length - len(seq)), constant_values=map_dict['[PAD]'])
        input_ids.append(input_id)
        input_mask = np.pad(np.ones(seq.shape, dtype=np.int32), (0, max_length - len(seq)),
                            constant_values=map_dict['[PAD]'])
        input_masks.append(input_mask)

    input_ids = np.array(input_ids)
    input_masks = np.array(input_masks)
    return Tensor(input_ids, ms.int32), Tensor(input_masks, ms.int32)


def bulid_model():
    '''
    Predict function
    '''
    net = BertModel(bert_net_cfg, is_training=False, use_one_hot_embeddings=False)
    net.set_train(False)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    return net


if __name__ == "__main__":
    cfg.device_id = get_device_id()
    cfg.device_num = get_device_num()
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=cfg.device_id)

    ms.context.set_context(variable_memory_max_size="30GB")
    ckpt_path = '<path to the pretrained model>/checkpoint.ckpt'

    data_path = '<path to the dir to extract features>/raw/'
    filelist = glob.glob(data_path+"*.txt")
    filelist.sort()
    result_path = "<path to the dir to save results>/result/"
    os.makedirs(result_path, exist_ok=True)

    net = bulid_model()
    for file in filelist:
        print(">>>>>>>>>>>>>>>>>>>>>>Process file:", file)
        f = open(file, "r")
        lines = f.readlines()
        f.close()
        seqs = [line.strip()[:40000] for line in lines]
        
        assert len(seqs) != 0

        # set different batch size for different sequence length
        # the length and batch size threshold can be adjusted according to the hardware
        length = [len(seq) for seq in seqs]
        batch_size = [64 if len(seq) < 300 else 8 for seq in seqs]
        df = pd.DataFrame()
        df["sequences"] = seqs
        df["length"] = length
        df["batch_size"] = batch_size
        df.sort_values("length", inplace=True, ascending=True)
        column = df['batch_size'].unique()
        embeddings = []
        sorted_index = []
        for col in column:
            new_df = df[df['batch_size'].isin([col])]
            sorted_index += new_df.index.tolist()
            embeddings += get_embedding(new_df["sequences"].tolist(), batch_size=col)
        sorted_embedding = [i for i in range(len(seqs))]
        for i in range(len(seqs)):
            sorted_embedding[sorted_index[i]] = embeddings[i]
        embedding_saved = sorted_embedding
        print(len(embedding_saved),embedding_saved[0].shape)
        embedding_out_path = os.path.join(result_path, os.path.basename(file) + "_embedding.pickle")
        print(f"extracted feature save in {embedding_out_path}")
        with open(embedding_out_path, "wb") as f:
            pickle.dump(embedding_saved, f)