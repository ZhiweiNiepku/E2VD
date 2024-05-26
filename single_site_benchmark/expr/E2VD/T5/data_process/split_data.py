import pandas as pd
import numpy as np
import random


feature_all=np.load('<path to data>/single_expr_t5_all_data.npy')
label_all=np.load('<path to data>/single_expr_t5_all_label.npy').flatten()
print(feature_all.shape)
print(label_all.shape)
train_idx=np.load('data/single_site_benchmark/expr/expr_train_idx.npy')
test_idx=np.load('data/single_site_benchmark/expr/expr_test_idx.npy')
train_feature=feature_all[train_idx]
train_label=label_all[train_idx]

test_feature=feature_all[test_idx]
test_label=label_all[test_idx]

print(train_feature.shape)
print(train_label.shape)
print(test_feature.shape)
print(test_label.shape)

np.save('<path to data>/single_expr_t5_train_data.npy',train_feature)
np.save('<path to data>/single_expr_t5_train_label.npy',train_label)
np.save('<path to data>/single_expr_t5_test_data.npy',test_feature)
np.save('<path to data>/single_expr_t5_test_label.npy',test_label)
