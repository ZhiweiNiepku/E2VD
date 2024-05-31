import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import os

feature_RBD_bn=np.load('<path to data>/RBD_embedding_train-0.npy').max(axis=1)
feature_HSEQ_bn=np.load('<path to data>/HSEQ_embedding_train-0.npy').max(axis=1)
feature_LSEQ_bn=np.load('<path to data>/LSEQ_embedding_train-0.npy').max(axis=1)
label_bn=np.load('<path to data>/label_train-0.npy')

feature_bn = np.concatenate([feature_RBD_bn, feature_HSEQ_bn, feature_LSEQ_bn], axis = 1)

print(feature_bn.shape)
print(label_bn.shape)

feature_an=np.load('feat_an.npy')
label_an=np.load('reg_label.npy')

print(feature_an.shape)
print(label_an.shape)


pca=PCA(n_components=2,random_state=27)

feature_bn_pca=pca.fit_transform(feature_bn)
feature_an_pca=pca.fit_transform(feature_an)
np.save('feature_before_pca.npy',feature_bn_pca)
np.save('feature_after_pca.npy',feature_an_pca)
