import numpy as np
import pandas as pd


df=pd.read_csv('data/single_site_benchmark/expr/expr_4k_all.csv')
train_idx = np.load('data/single_site_benchmark/expr/expr_train_idx.npy')
test_idx = np.load('data/single_site_benchmark/expr/expr_test_idx.npy')

df_train = df.iloc[train_idx]
df_test = df.iloc[test_idx]

print(df_train.shape)
print(df_test.shape)

df_train.to_csv('data/single_site_benchmark/expr/expr_4k_train.csv',index=False)
df_test.to_csv('data/single_site_benchmark/expr/expr_4k_test.csv',index=False)