import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# choose variant to draw
name= 'BA5' #'XBB15'

def minmax(data):
    return (data-data.min())/(data.max()-data.min())

# mutation sites on variants
skip_dict={
    'BA5':[339,371,373,375,376,405,408,417,440,452,477,478,484,486,498,501,505],
    'XBB15':[339,346,368,371,373,375,376,405,408,417,440,445,446,460,477,478,484,486,490,498,501,505]
}

# mutation sites on target. These sites are just for figure labeling
target_dict={
    'BA5':['346','352','354','356','357','383','384','385','386','391','393','396','403','405','417','420','444','445','446',
           '449','450','452','453','455','456','460','468','482','483','484','485','490','493','505','514','518','519','524',
           '525','526'],
    'XBB15':['346','352','354','356','357','383','384','385','386','391','393','396','403','405','417','420','444','445','446',
             '449','450','452','453','455','456','460','468','470','482','483','484','485','490','493','505','514','516','518',
             '519','524','525','526']
}

targets=target_dict[name]

targets=np.unique(targets)

skip=skip_dict[name]

max_=520

# draw with matplotlib
if 0:
    bind_reg=np.load('data/perceive_evolutionary_trends/E2VD/{}/bind_result/predict_reg_{}.npy'.format(name, name))
    expr_reg=np.load('data/perceive_evolutionary_trends/E2VD/{}/expr_result/predict_reg_{}.npy'.format(name, name))
    escape_reg=np.load('data/perceive_evolutionary_trends/E2VD/{}/escape_result/predict_reg_{}.npy'.format(name, name))
    bind_cls=np.load('data/perceive_evolutionary_trends/E2VD/{}/bind_result/predict_cls_{}.npy'.format(name, name))
    expr_cls=np.load('data/perceive_evolutionary_trends/E2VD/{}/expr_result/predict_cls_{}.npy'.format(name, name))
    escape_cls=np.load('data/perceive_evolutionary_trends/E2VD/{}/escape_result/predict_cls_{}.npy'.format(name, name))
    
    if 1:
        bind_reg=minmax(bind_reg)
        expr_reg=minmax(expr_reg)
        escape_reg=minmax(escape_reg)
        bind_cls=minmax(bind_cls)
        expr_cls=minmax(expr_cls)
        escape_cls=minmax(escape_cls)

    sites=[]
    muts=[]

    subs='ACDEFGHIKLMNPQRSTVWY'

    for i in range(len(bind_reg)):
        sites.append(i//20+331)
        muts.append(subs[i%20])

    df=pd.DataFrame(np.array([sites,muts,bind_reg,expr_reg,escape_reg,bind_cls,expr_cls,escape_cls]).T,
                 columns=['site','mutation','bind_reg','expr_reg',
                          'escape_reg','bind_cls','expr_cls','escape_cls'])
    
    bind_cls=0.25
    expr_cls=0.7
    escape_cls=0.55
    
    bind_reg=0.0
    expr_reg=0.0
    escape_reg=0.0
    
    df=df[(df['bind_cls'].values).astype(float)>bind_cls]
    df=df[(df['expr_cls'].values).astype(float)>expr_cls]
    df=df[(df['escape_cls'].values).astype(float)>escape_cls]
    df=df[(df['bind_reg'].values).astype(float)>bind_reg]
    df=df[(df['expr_reg'].values).astype(float)>expr_reg]
    df=df[(df['escape_reg'].values).astype(float)>escape_reg]
    
    filtered_sites=df['site'].value_counts().index
    
    count=0
    for target in targets:
        if target in filtered_sites and df['site'].value_counts()[target]>3:
            count+=1
    print(filtered_sites)
    print(count,len(filtered_sites),count/len(filtered_sites))
    
    for i in range(len(filtered_sites)):
        if filtered_sites[i] in targets:
            print(i)
    
    if 1:
        plt.figure(figsize=(12,4))
        count_dict=df['site'].value_counts()
        count_dict_sorted = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
        
        nums=[]
        ticks=[]
        for i in range(331,532):
            if i in skip:
                nums.append(0)
                print('skip {}'.format(i))
            elif str(i) in count_dict.keys():
                nums.append(count_dict[str(i)])
            else:
                nums.append(0)
            if (i-331)%15==0:
                ticks.append(i)
            
        for target in targets:
            if int(target) in skip:
                continue
            if int(target) > max_:
                continue
            try:
                plt.text(int(target)-331,count_dict[target],target,c='red',fontsize=12)
            except:
                print('no {}'.format(target))
            
        for i in range(min(10,len(count_dict_sorted))):
            if int(count_dict_sorted[i][0]) in skip:
                continue
            elif int(count_dict_sorted[i][0])>=max_:
                continue
            plt.text(int(count_dict_sorted[i][0])-331,count_dict_sorted[i][1],count_dict_sorted[i][0])
        
        plt.plot(nums)
        plt.xticks(np.arange(0,len(nums),15),ticks)
        plt.xlim(-5,max_-331)
        plt.savefig('data/perceive_evolutionary_trends/{}/plot_{}_{}_{}.png'.format(name,bind_cls,expr_cls,escape_cls))

# prepare to draw with R
if 1:
    bind_reg=np.load('data/perceive_evolutionary_trends/E2VD/{}/bind_result/predict_reg_{}.npy'.format(name, name))
    expr_reg=np.load('data/perceive_evolutionary_trends/E2VD/{}/expr_result/predict_reg_{}.npy'.format(name, name))
    escape_reg=np.load('data/perceive_evolutionary_trends/E2VD/{}/escape_result/predict_reg_{}.npy'.format(name, name))
    bind_cls=np.load('data/perceive_evolutionary_trends/E2VD/{}/bind_result/predict_cls_{}.npy'.format(name, name))
    expr_cls=np.load('data/perceive_evolutionary_trends/E2VD/{}/expr_result/predict_cls_{}.npy'.format(name, name))
    escape_cls=np.load('data/perceive_evolutionary_trends/E2VD/{}/escape_result/predict_cls_{}.npy'.format(name, name))
    
    if 1:
        bind_reg=minmax(bind_reg)
        expr_reg=minmax(expr_reg)
        escape_reg=minmax(escape_reg)
        bind_cls=minmax(bind_cls)
        expr_cls=minmax(expr_cls)
        escape_cls=minmax(escape_cls)

    sites=[]
    muts=[]

    subs='ACDEFGHIKLMNPQRSTVWY'

    for i in range(len(bind_reg)):
        sites.append(i//20+331)
        muts.append(subs[i%20])

    df=pd.DataFrame(np.array([sites,muts,bind_reg,expr_reg,escape_reg,bind_cls,expr_cls,escape_cls]).T,
                 columns=['site','mutation','bind_reg','expr_reg',
                          'escape_reg','bind_cls','expr_cls','escape_cls'])
    
    bind_cls=0.25
    expr_cls=0.7
    escape_cls=0.55
    
    bind_reg=0.0
    expr_reg=0.0
    escape_reg=0.0
    
    df=df[(df['bind_cls'].values).astype(float)>bind_cls]
    df=df[(df['expr_cls'].values).astype(float)>expr_cls]
    df=df[(df['escape_cls'].values).astype(float)>escape_cls]
    df=df[(df['bind_reg'].values).astype(float)>bind_reg]
    df=df[(df['expr_reg'].values).astype(float)>expr_reg]
    df=df[(df['escape_reg'].values).astype(float)>escape_reg]
    
    filtered_sites=df['site'].value_counts().index
    
    count=0
    for target in targets:
        if target in filtered_sites and df['site'].value_counts()[target]>3:
            count+=1
    print(filtered_sites)
    print(count,len(filtered_sites),count/len(filtered_sites))
    
    for i in range(len(filtered_sites)):
        if filtered_sites[i] in targets:
            print(i)
    
    if 1:
        count_dict=df['site'].value_counts()
        count_dict_sorted = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
        
        data=[]
        for i in range(331,521):
            num_=0
            idx_=i
            kind_=None
            if str(i) in count_dict.keys():
                num_=count_dict[str(i)]
            if str(i) in targets:
                kind_='target'
            elif i in skip:
                kind_='mutation'
            data.append([idx_,num_,kind_])    
        data=pd.DataFrame(data,columns=['index','num','kind'])
        data.to_csv('data/perceive_evolutionary_trends/E2VD/{}_data.csv'.format(name),index=False)