import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


feature_bn=np.load('<path to data>/single_expr_customize_train_data.npy')
label_bn=np.load('<path to data>/single_expr_customize_train_label.npy')

feature_bn=feature_bn.max(axis=1)

print(feature_bn.shape)
print(label_bn.shape)

feature_an=np.load('feat_after.npy')
label_an=np.load('reg_label.npy')

print(feature_an.shape)
print(label_an.shape)


pca=PCA(n_components=2,random_state=27)

feature_bn_pca=pca.fit_transform(feature_bn)
feature_an_pca=pca.fit_transform(feature_an)
np.save('feature_before_pca.npy',feature_bn_pca)
np.save('feature_after_pca.npy',feature_an_pca)

