# convert features from UniRep model to features for training
import pickle
import numpy as np


# _UniRep.pkl is the extracted features from UniRep model
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

np.save('data/1900_test_all_data.npy', feat)
np.save('data/1900_test_all_label.npy', label)