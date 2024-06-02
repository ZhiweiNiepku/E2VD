import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


ORANGE_in= '#F9E2A4' #249,226,164 #'#EBD5B0'
ORANGE_edge= '#C6A53A' #198,165,58 #'#B7A175'
BLUE1_in= '#A6C8D9' #166,200,217 #'#BAD6D9'
BLUE1_edge= '#6C8BA6' #108,139,166 #'#7D9CA7'
GREEN1_in= '#C8E17E' #200,225,126
GREEN1_edge= '#8BAC4F' #139,172,79 #'#9BAD60'
GREEN2_in= '#B9E1A2' #185,225,162
GREEN2_edge= '#7CAC69' #124,172,105 #'#8EAD7A'
BLUE2_in= '#93A8DB' #147,168,219
BLUE2_edge= '#5E6AA7' #94,106,167 #'#7082A7'
BLUE3_in= '#A59FD8' #165,159,216
BLUE3_edge= '#7066A6' #112,102,166 #'#7A76A6'
PURPLE_in= '#D294AD' #210,148,173
PURPLE_edge= '#9E5F6E' #158,95,110 #'#A07086'
RED_in= '#C88C7A' #200,140,122
RED_edge= '#944C31' #148,76,49
ORANGE2_in= '#DFBD86' #223,189,134
ORANGE2_edge= '#AA8138' #170,129,56

draw_name = 'pos' # 'neg'

label=np.load('data/mine_rare_beneficial_mutation/MT/y_label.npy')

if draw_name == 'pos':
    mask_ = (label>1)
else:
    mask_ = (label<1)

label = label[mask_]
pred1=np.load('data/mine_rare_beneficial_mutation/MT/y_cls.npy')[mask_]
pred2=np.load('data/mine_rare_beneficial_mutation/BCE_MSE/y_cls.npy')[mask_]

label_res=[]
reg_res=[]

if draw_name == 'pos':
    muts=['G339D', 'R346D', 'S371D', 'S373M', 'D405S', 'N440A', 'V445K', 'N460A', 'S477D', 'T478K', 'E484K', 'F490A', 'Q498F', 'N501F', 'Y505W']
else:
    muts=['G339C', 'R346A', 'S371A', 'S373A', 'D405A', 'N440C', 'V445C', 'N460C', 'S477C', 'T478C', 'E484A', 'F490C', 'Q498A', 'N501C', 'Y505A']

SIZE=600

CORRECT=BLUE3_edge #GREEN2_edge
ERROR=RED_in

dpi=500

flg1,flg2,flg3,flg4=1,1,1,1
plt.figure(figsize=(26,7),dpi=dpi)
for i in range(len(muts)):
    if (label[i]>=1 and pred1[i]>=0.5) or (label[i]<1 and pred1[i]<0.5):
        if flg1:
            plt.scatter(muts[i],pred1[i],marker='*',s=SIZE,c=CORRECT,label='Multi-task Focal Loss Correct')
            flg1=0
        else:
            plt.scatter(muts[i],pred1[i],marker='*',s=SIZE,c=CORRECT)
    else:
        if flg2:
            plt.scatter(muts[i],pred1[i],marker='*',s=SIZE,c=ERROR,label='Multi-task Focal Loss Error')
            flg2=0
        else:
            plt.scatter(muts[i],pred1[i],marker='*',s=SIZE,c=ERROR)

label[0], label[10] = label[10], label[0]
pred2[0], pred2[10] = pred2[10], pred2[0]
muts[0], muts[10] = muts[10], muts[0]

for i in range(len(muts)):
    if (label[i]>=1 and pred2[i]>=0.5) or (label[i]<1 and pred2[i]<0.5):
        if flg3:
            plt.scatter(muts[i],pred2[i],marker='o',s=SIZE,c=CORRECT,label='BCE & MSE Correct')
            flg3=0
        else:
            plt.scatter(muts[i],pred2[i],marker='o',s=SIZE,c=CORRECT)
    else:
        if flg4:
            plt.scatter(muts[i],pred2[i],marker='o',s=SIZE,c=ERROR,label='BCE & MSE Error')
            flg4=0
        else:
            plt.scatter(muts[i],pred2[i],marker='o',s=SIZE,c=ERROR)

plt.plot([0,len(muts)-1],[0.5,0.5],c=BLUE1_edge,linestyle='--')



FONTSIZE=23
plt.legend(loc=0, prop={'family':'Arial','size':21})
plt.xticks(fontproperties='Arial',size=21,rotation=30)
plt.yticks(fontproperties='Arial',size=21)
plt.ylabel('Predicted probability',fontdict={'family':'Arial','size':FONTSIZE})
plt.ylim(top=1.19)
plt.xlim(left=-2)
#plt.ylim((0,1.5))
#plt.scatter(np.arange(len(df1['label'])),df1['label'].values,c='green')
#plt.show()
plt.savefig('scatter_{}.svg'.format(draw_name))