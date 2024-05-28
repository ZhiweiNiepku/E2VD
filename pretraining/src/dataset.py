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
Data operations, will be used in run_pretrain.py
"""
import os
import math
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
from mindspore import log as logger
import glob
import random


class BucketDatasetGenerator:
    """
    Provide data distribution of different gears for the bert network.

    Args:
        dataset (Dataset): The training dataset.
        batch_size (Int): The training batchsize.
        bucket_list (List): List of different sentence lengths，such as [128, 256, 512]. Default: None.
    """

    def __init__(self, dataset, batch_size, bucket_list=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_list = bucket_list
        self.data_bucket = {bucket: [] for bucket in bucket_list}
        bucket_size = len(bucket_list)
        self.random_list = np.random.binomial(n=(bucket_size - 1), p=0.55, size=self.__len__())
        self.random_list = (self.random_list + 2) % bucket_size
        self.random_list = [bucket_list[i] for i in self.random_list]
        self.iter = 0

    def __next__(self):
        for item in self.iterator:
            for seq_length in self.bucket_list:
                if np.sum(item[1]) <= seq_length:
                    self.data_bucket[seq_length].append(item)
                    break
            for key in self.data_bucket.keys():
                data = self.data_bucket[key]
                if len(data) >= self.batch_size and self.random_list[self.iter] == key:
                    self.data_bucket[key] = self.data_bucket[key][self.batch_size:]
                    arr = data[0]
                    for i in range(1, self.batch_size):
                        current_data = data[i]
                        for j in range(len(current_data)):
                            arr[j] = np.concatenate((arr[j], current_data[j]))
                    res = ()
                    for label in arr:
                        newlabel = np.reshape(label, (self.batch_size, -1))
                        res += (newlabel,)
                    res += (np.array(key, np.int32),)
                    self.iter += 1
                    return res
        raise StopIteration

    def __iter__(self):
        self.iterator = self.dataset.create_tuple_iterator(output_numpy=True)
        return self

    def __len__(self):
        return (self.dataset.get_dataset_size() // self.batch_size) - 1


def create_bert_dataset(device_num=1, rank=0, do_shuffle="true", data_dir=None, schema_dir=None, batch_size=32,
                        bucket_list=None):
    """create train dataset"""
    # apply repeat operations
    # files = os.listdir(data_dir)
    # data_files = []
    # for file_name in files:
    #     # if "tfrecord" in file_name:
    #     #     data_files.append(os.path.join(data_dir, file_name))
    #     if not file_name.endswith(".db"):
    #         data_files.append(os.path.join(data_dir, file_name))


    data_files = glob.glob(data_dir + "/*.mindrecord")

    print("Train data:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(data_files)
    data_files.sort()

    # data_set = ds.TFRecordDataset(data_files, schema_dir if schema_dir != "" else None,
    #                               columns_list=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
    #                                             "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"],
    #                               shuffle=ds.Shuffle.FILES if do_shuffle == "true" else False,
    #                               num_shards=device_num, shard_id=rank, shard_equal_rows=True)

    data_set = ds.MindDataset(data_files,
                              columns_list=["input_ids", "input_mask", "masked_lm_positions", "masked_lm_ids",
                                            "masked_lm_weights"],
                              shuffle=True if do_shuffle == "true" else False,
                              num_shards=device_num, shard_id=rank)
    print('create_bert_dataset', data_set.get_dataset_size())

    if bucket_list:
        bucket_dataset = BucketDatasetGenerator(data_set, batch_size, bucket_list=bucket_list)
        data_set = ds.GeneratorDataset(bucket_dataset,
                                       column_names=["input_ids", "input_mask",
                                                     "masked_lm_positions", "masked_lm_ids", "masked_lm_weights",
                                                     "sentence_flag"],
                                       shuffle=False)
    else:
        data_set = data_set.batch(batch_size, drop_remainder=True)
    ori_dataset_size = data_set.get_dataset_size()
    print('origin dataset size: ', ori_dataset_size)
    type_cast_op = C.TypeCast(mstype.int32)
    data_set = data_set.map(operations=type_cast_op, input_columns="masked_lm_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="masked_lm_positions")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    # apply batch operations
    logger.info("data size: {}".format(data_set.get_dataset_size()))
    logger.info("repeat count: {}".format(data_set.get_repeat_count()))
    return data_set


def numpy_mask_tokens(inputs, mlm_probability=0.80, mask_token_id=9, token_length=5):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """

    labels = np.copy(inputs)
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = np.full(labels.shape, mlm_probability)
    for value in range(5):
        probability_matrix[inputs == value] = 0
    # Numpy doesn't have bernoulli, so we use a binomial with 1 trial
    masked_indices = np.random.binomial(1, probability_matrix, size=probability_matrix.shape).astype(bool)
    masked_lm_positions = np.where(masked_indices == True)[0]

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked_indices
    inputs[indices_replaced] = mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    indices_random = (
            np.random.binomial(1, 0.5, size=labels.shape).astype(bool) & masked_indices & ~indices_replaced)
    random_words = np.random.randint(
        low=0, high=token_length, size=np.count_nonzero(indices_random), dtype=np.int64
    )
    inputs[indices_random] = random_words
    masked_lm_ids = labels[masked_lm_positions]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, masked_lm_positions, masked_lm_ids


def get_lines(dataset_file):
    lines = []
    for txt in dataset_file:
        with open(txt, "r") as f:
            for line in f:
                if not len(line.strip()) <= 10:
                    lines.append(line.strip())
    return lines


def cut(seq, max_len):
    seqs = [seq[i:i + max_len] for i in range(0, len(seq), max_len)]
    index = random.randint(0, len(seqs) - 1)
    return seqs[index]


class RNADataset:
    """
    format of the dataset file:
    [start_token] + [A, C, G, T, X] ...

    start_token:
        S: the real start of the RNA sequence
        M: the middle start of the RNA sequence (when the sequence is too long, cut them into pieces)

    cv_schema_json = {"input_ids": {"type": "int32", "shape": [-1]},
                  "input_mask": {"type": "int32", "shape": [-1]},
                  "masked_lm_positions": {"type": "int32", "shape": [-1]},
                  "masked_lm_ids": {"type": "int32", "shape": [-1]},
                  "masked_lm_weights": {"type": "float32", "shape": [-1]}}
    """

    def __init__(self, dataset_file, seq_length, max_ml, mask_percent):
        self.lines = get_lines(dataset_file)
        self.num_samples = len(self.lines)
        self.max_len = seq_length
        self.actual_length = seq_length - 2
        self.mask_percent = mask_percent
        self.max_lm = max_ml
        self.map_dict = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'L': 5, 'A': 6, 'G': 7, 'V': 8,
                         'E': 9, 'S': 10, 'I': 11, 'K': 12, 'R': 13, 'D': 14, 'T': 15, 'P': 16, 'N': 17, 'Q': 18,
                         'F': 19, 'Y': 20, 'M': 21, 'H': 22, 'C': 23, 'W': 24, 'X': 25}

        self.count = 0

    def __getitem__(self, index):

        self.count += 1

        seq = self.lines[index].strip('\n')
        seq = seq.upper()
        seq = cut(seq, self.actual_length)

        # if self.count<=100:
        #     print("Original sequence is " + seq)

        seq_new = ''
        for i in range(len(seq)):
            if seq[i] not in self.map_dict.keys():
                seq_new += 'X'
            else:
                seq_new += seq[i]

        if self.count<=100:
            print("New sequence is " + seq_new)

        seq = seq_new
        seq = [self.map_dict['[CLS]']] + [self.map_dict[x] for x in seq] + [self.map_dict['[SEP]']]
        seq = np.array(seq)
        input_id, masked_lm_position, masked_lm_id = numpy_mask_tokens(inputs=seq, mlm_probability=self.mask_percent,
                                                                       mask_token_id=self.map_dict['[MASK]'],
                                                                       token_length=len(self.map_dict))
        input_id = np.pad(seq.astype(np.int32), (0, self.max_len - len(seq)), constant_values=self.map_dict['[PAD]'])
        input_mask = np.pad(np.ones(seq.shape, dtype=np.int32), (0, self.max_len - len(seq)),
                            constant_values=self.map_dict['[PAD]'])

        masked_lm_id = masked_lm_id[:self.max_lm]
        masked_lm_position = masked_lm_position[:self.max_lm]
        masked_lm_weight = np.pad(np.ones(masked_lm_id.shape, dtype=np.float32), (0, self.max_lm - len(masked_lm_id)),
                                  constant_values=self.map_dict['[PAD]'])
        masked_lm_position = np.pad(masked_lm_position.astype(np.int32), (0, self.max_lm - len(masked_lm_position)),
                                    constant_values=self.map_dict['[PAD]'])
        masked_lm_id = np.pad(masked_lm_id.astype(np.int32), (0, self.max_lm - len(masked_lm_id)),
                              constant_values=self.map_dict['[PAD]'])
        return input_id, input_mask, masked_lm_position, masked_lm_id, masked_lm_weight

    def __len__(self):
        return self.num_samples

    def __del__(self):
        del self.lines


def create_txt_dataset(device_num=1, rank=0, do_shuffle="true", data_dir=None, schema_dir=None, batch_size=2,
                       bucket_list=None):
    """
    Create dataset

    Inputs:
        batch_size: batch size
        data_path: path of your MindRecord files
        device_num: total device number
        rank: current rank id
        drop: whether drop remainder
        eod_reset: whether enable position reset and attention mask reset
        column_name: the column name of the mindrecord file. Default is input_ids
        epoch: The repeat times of the dataset
    Returns:
        dataset_restore: the dataset for training or evaluating
    """
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f[-1] == "t"]
    data_files.sort()
    dataset_generator = RNADataset(data_files, seq_length=256, max_ml=64, mask_percent=0.15)
    data_set = ds.GeneratorDataset(dataset_generator,
                                   column_names=["input_ids",
                                                 "input_mask",
                                                 "masked_lm_positions",
                                                 "masked_lm_ids",
                                                 "masked_lm_weights"],
                                   shuffle=True if do_shuffle == "true" else False,
                                   num_shards=device_num,
                                   shard_id=rank)
    print('create_bert_dataset', data_set.get_dataset_size())

    if bucket_list:
        bucket_dataset = BucketDatasetGenerator(data_set, batch_size, bucket_list=bucket_list)
        data_set = ds.GeneratorDataset(bucket_dataset,
                                       column_names=["input_ids", "input_mask",
                                                     "masked_lm_positions", "masked_lm_ids", "masked_lm_weights",
                                                     "sentence_flag"],
                                       shuffle=False)
    else:
        data_set = data_set.batch(batch_size, drop_remainder=True)
    ori_dataset_size = data_set.get_dataset_size()
    print('origin dataset size: ', ori_dataset_size)
    type_cast_op = C.TypeCast(mstype.int32)
    data_set = data_set.map(operations=type_cast_op, input_columns="masked_lm_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="masked_lm_positions")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    # apply batch operations
    logger.info("data size: {}".format(data_set.get_dataset_size()))
    logger.info("repeat count: {}".format(data_set.get_repeat_count()))
    return data_set


def create_ner_dataset(batch_size=1, assessment_method="accuracy", data_file_path=None,
                       dataset_format="mindrecord", schema_file_path=None, do_shuffle=True, drop_remainder=True):
    """create finetune or evaluation dataset"""
    type_cast_op = C.TypeCast(mstype.int32)
    if dataset_format == "mindrecord":
        dataset = ds.MindDataset([data_file_path],
                                 columns_list=["input_ids", "input_mask", "segment_ids", "label_ids"],
                                 shuffle=do_shuffle)
    else:
        dataset = ds.TFRecordDataset([data_file_path], schema_file_path if schema_file_path != "" else None,
                                     columns_list=["input_ids", "input_mask", "segment_ids", "label_ids"],
                                     shuffle=do_shuffle)
    if assessment_method == "Spearman_correlation":
        type_cast_op_float = C.TypeCast(mstype.float32)
        dataset = dataset.map(operations=type_cast_op_float, input_columns="label_ids")
    else:
        dataset = dataset.map(operations=type_cast_op, input_columns="label_ids")
    dataset = dataset.map(operations=type_cast_op, input_columns="segment_ids")
    dataset = dataset.map(operations=type_cast_op, input_columns="input_mask")
    dataset = dataset.map(operations=type_cast_op, input_columns="input_ids")
    # apply batch operations
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset


def create_classification_dataset(batch_size=1, assessment_method="accuracy",
                                  data_file_path=None, schema_file_path=None, do_shuffle=True):
    """create finetune or evaluation dataset"""
    type_cast_op = C.TypeCast(mstype.int32)
    data_set = ds.TFRecordDataset([data_file_path], schema_file_path if schema_file_path != "" else None,
                                  columns_list=["input_ids", "input_mask", "segment_ids", "label_ids"],
                                  shuffle=do_shuffle)
    if assessment_method == "Spearman_correlation":
        type_cast_op_float = C.TypeCast(mstype.float32)
        data_set = data_set.map(operations=type_cast_op_float, input_columns="label_ids")
    else:
        data_set = data_set.map(operations=type_cast_op, input_columns="label_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="segment_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set


def generator_squad(data_features):
    for feature in data_features:
        yield (feature.input_ids, feature.input_mask, feature.segment_ids, feature.unique_id)


def create_squad_dataset(batch_size=1, data_file_path=None, schema_file_path=None,
                         is_training=True, do_shuffle=True):
    """create finetune or evaluation dataset"""
    type_cast_op = C.TypeCast(mstype.int32)
    if is_training:
        data_set = ds.TFRecordDataset([data_file_path], schema_file_path if schema_file_path != "" else None,
                                      columns_list=["input_ids", "input_mask", "segment_ids", "start_positions",
                                                    "end_positions", "unique_ids", "is_impossible"],
                                      shuffle=do_shuffle)
        data_set = data_set.map(operations=type_cast_op, input_columns="start_positions")
        data_set = data_set.map(operations=type_cast_op, input_columns="end_positions")
    else:
        data_set = ds.GeneratorDataset(generator_squad(data_file_path), shuffle=do_shuffle,
                                       column_names=["input_ids", "input_mask", "segment_ids", "unique_ids"])
    data_set = data_set.map(operations=type_cast_op, input_columns="segment_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="unique_ids")
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set


def create_eval_dataset(batchsize=32, device_num=1, rank=0, data_dir=None, schema_dir=None):
    """create evaluation dataset"""
    data_files = []
    if os.path.isdir(data_dir):
        files = os.listdir(data_dir)
        for file_name in files:
            # if "tfrecord" in file_name:
            #     data_files.append(os.path.join(data_dir, file_name))
            if not file_name.endswith(".db"):
                data_files.append(os.path.join(data_dir, file_name))
    else:
        data_files.append(data_dir)

    data_files.sort()

    # data_set = ds.TFRecordDataset(data_files, schema_dir if schema_dir != "" else None,
    #                               columns_list=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
    #                                             "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"],
    #                               shard_equal_rows=True)
    data_set = ds.MindDataset(data_files,
                              columns_list=["input_ids", "input_mask", "masked_lm_positions", "masked_lm_ids",
                                            "masked_lm_weights"],
                              shuffle=False)
    print(data_set.get_dataset_size())

    ori_dataset_size = data_set.get_dataset_size()
    print("origin eval size: ", ori_dataset_size)
    dtypes = data_set.output_types()
    shapes = data_set.output_shapes()
    output_batches = math.ceil(ori_dataset_size / device_num / batchsize)
    padded_num = output_batches * device_num * batchsize - ori_dataset_size
    print("padded num: ", padded_num)
    # padded_num = 0
    if padded_num > 0:
        item = {"input_ids": np.zeros(shapes[0], dtypes[0]),
                "input_mask": np.zeros(shapes[1], dtypes[1]),
                "masked_lm_positions": np.zeros(shapes[2], dtypes[2]),
                "masked_lm_ids": np.zeros(shapes[3], dtypes[3]),
                "masked_lm_weights": np.zeros(shapes[4], dtypes[4])}
        padded_samples = [item for x in range(padded_num)]
        padded_ds = ds.PaddedDataset(padded_samples)
        eval_ds = data_set + padded_ds
        sampler = ds.DistributedSampler(num_shards=device_num, shard_id=rank, shuffle=False)
        eval_ds.use_sampler(sampler)
    else:
        # eval_ds = ds.TFRecordDataset(data_files, schema_dir if schema_dir != "" else None,
        #                              columns_list=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
        #                                            "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"],
        #                              num_shards=device_num, shard_id=rank, shard_equal_rows=True)
        eval_ds = ds.MindDataset(data_files,
                                 columns_list=["input_ids", "input_mask",
                                               "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"],
                                 shuffle=False,
                                 num_shards=device_num, shard_id=rank)

    type_cast_op = C.TypeCast(mstype.int32)
    eval_ds = eval_ds.map(input_columns="masked_lm_ids", operations=type_cast_op)
    eval_ds = eval_ds.map(input_columns="masked_lm_positions", operations=type_cast_op)
    eval_ds = eval_ds.map(input_columns="input_mask", operations=type_cast_op)
    eval_ds = eval_ds.map(input_columns="input_ids", operations=type_cast_op)

    eval_ds = eval_ds.batch(batchsize, drop_remainder=True)
    print("eval data size: {}".format(eval_ds.get_dataset_size()))
    print("eval repeat count: {}".format(eval_ds.get_repeat_count()))
    return eval_ds
