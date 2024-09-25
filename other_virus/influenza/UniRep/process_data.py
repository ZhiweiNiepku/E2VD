import pickle
import numpy as np


data = pickle.load(open('_UniRep.pkl','rb'))

feat = []
label = []

for i in range(len(data)):
    feat.append(data[i][0])
    label.append(data[i][1])
    
feat = np.stack(feat, axis = 0)
label = np.array(label)

print(feat.shape)
print(label.shape)

np.save('data/other_virus/influenza/unirep/1900_all_data.npy', feat)
np.save('data/other_virus/influenza/unirep/1900_all_label.npy', label)

# data split
feature_all=np.load('data/other_virus/influenza/unirep/1900_all_data.npy')
label_all=np.load('data/other_virus/influenza/unirep/1900_all_label.npy').flatten()
print(feature_all.shape)
print(label_all.shape)
train_idx=np.load('data/other_virus/influenza/train_idx.npy')
test_idx=np.load('data/other_virus/influenza/test_idx.npy')
train_feature=feature_all[train_idx]
train_label=label_all[train_idx]

test_feature=feature_all[test_idx]
test_label=label_all[test_idx]

print(train_feature.shape)
print(train_label.shape)
print(test_feature.shape)
print(test_label.shape)

print(train_label)

np.save('data/other_virus/influenza/unirep/1900_train_data.npy',train_feature)
np.save('data/other_virus/influenza/unirep/1900_train_label.npy',train_label)
np.save('data/other_virus/influenza/unirep/1900_test_data.npy',test_feature)
np.save('data/other_virus/influenza/unirep/1900_test_label.npy',test_label)