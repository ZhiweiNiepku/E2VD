# calculate the mean and std of training features
# this is for the prediction in predict_expr_emb_main_pickle.py
import numpy as np


feats1=np.load('<path to training data>/expr_train_data.npy')
feats2=np.load('<path to training data>/expr_test_data.npy')
label1=np.load('<path to training data>/expr_train_label.npy').reshape(-1,1)
label2=np.load('<path to training data>/expr_test_label.npy').reshape(-1,1)
feature=np.concatenate([feats1,feats2],axis=0)
label=np.concatenate([label1,label2],axis=0).flatten()

mask_=np.isnan(label)

feature=feature[~mask_]

shape_=feature.shape
print(shape_)
feature=feature.reshape(shape_[0],-1)

mean=np.mean(feature,axis=0)
std=np.std(feature,axis=0)

print(mean.shape)
print(std.shape)

np.save('<path to mean value of training data>/embedding_flatten_mean.npy',mean)
np.save('<path to std value of training data>/embedding_flatten_std.npy',std)
