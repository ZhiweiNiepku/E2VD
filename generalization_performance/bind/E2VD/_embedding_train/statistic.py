# calculate mean and std
import numpy as np


feats1=np.load('<path to embedding data>/single_bind_customize_train_data.npy')
feats2=np.load('<path to embedding data>/single_bind_customize_test_data.npy')
label1=np.load('<path to embedding data>/single_bind_customize_train_label.npy').reshape(-1,1)
label2=np.load('<path to embedding data>/single_bind_customize_test_label.npy').reshape(-1,1)
feature=np.concatenate([feats1,feats2],axis=0)
label=np.concatenate([label1,label2],axis=0)
feature=feature[(label!=1).flatten()]

shape_=feature.shape
print(shape_)
feature=feature.reshape(shape_[0],-1)

max_=np.max(feature,axis=0)
min_=np.min(feature,axis=0)

feature=(feature-min_)/(max_-min_)

mean=np.mean(feature,axis=0)
std=np.std(feature,axis=0)

print(max_.shape)
print(min_.shape)

print(mean.shape)
print(std.shape)

np.save('<path to data>/bind_all_flatten_mean.npy',mean)
np.save('<path to data>/bind_all_flatten_std.npy',std)
