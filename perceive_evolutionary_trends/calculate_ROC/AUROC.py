import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


name='BA5' #'XBB15'

df=pd.read_csv('data/perceive_evolutionary_trends/E2VD/{}_data.csv'.format(name)) # index, num, kind

kinds = df['kind']

## for E2VD, the column name is num
nums = df['num'].values
## for MLAEP, the column name is values
# nums = df['values'].values

label = (kinds == 'target').values
#label = (~pd.isna(kinds)).values


nums_p = nums[label == 1]
nums_n = nums[label == 0]

count = 0
score = 0

for i in range(len(nums_p)):
    for j in range(len(nums_n)):
        if nums_p[i] > nums_n[j]:
            score += 1
        elif nums_p[i] == nums_n[j]:
            score += 0.5
        count += 1

print('auc:',score/count)


tpr = []
fpr = []

thres = np.linspace(nums.min(), nums.max(), 100)

for thre in thres:
    tp = ((nums >= thre) * (label == 1)).sum()
    fp = ((nums >= thre) * (label == 0)).sum()
    tn = ((nums < thre) * (label == 0)).sum()
    fn = ((nums < thre) * (label == 1)).sum()
    
    
    tpr.append(tp/(tp+fn))
    fpr.append(fp/(fp+tn))
    
# roc
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],c='orange',linestyle='--')
plt.savefig('roc.png')
plt.show()
