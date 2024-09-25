import pandas as pd
import numpy as np
import random

import pickle


feature_all=np.load('data/other_virus/zika/E2VD/embedding_all_data.npy')
label_all = pd.read_csv('data/other_virus/zika/data_all.csv')['label'].values
print(feature_all.shape)
print(label_all.shape)
train_idx=np.load('data/other_virus/zika/train_idx.npy')
test_idx=np.load('data/other_virus/zika/test_idx.npy')
train_feature=feature_all[train_idx]
train_label=label_all[train_idx]

test_feature=feature_all[test_idx]
test_label=label_all[test_idx]

print(train_feature.shape)
print(train_label.shape)
print(test_feature.shape)
print(test_label.shape)

np.save('data/other_virus/zika/E2VD/train_data.npy',train_feature)
np.save('data/other_virus/zika/E2VD/train_label.npy',train_label)
np.save('data/other_virus/zika/E2VD/test_data.npy',test_feature)
np.save('data/other_virus/zika/E2VD/test_label.npy',test_label)
