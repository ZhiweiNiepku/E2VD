import os
import matplotlib.pyplot as plt
import numpy as np


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



SCATTER=0
MSIZE=70
BMSIZE=300
figsize=(8,6.5)
dpi=1000

LEGEND_SIZE=11
TICK_SIZE=17

if SCATTER:
    x =  ['AUC','Accuracy','F1-Score','Precision','Recall','Correlation'] #,'MSE']

    markers = ['o','v','>','D','<','s','^','*']
    labels  = ['NN_MM-GBSA','SeqVec','ProtTrans T5','ESM-1b','ESM-1v','ESM-2(650M)','ESM-2(15B)','ProtFound-BA']
    colors = [BLUE1_edge, PURPLE_edge, ORANGE_edge, GREEN1_edge, BLUE2_edge, GREEN2_edge, BLUE3_edge, RED_edge]
    sizes   = [MSIZE,MSIZE,MSIZE,MSIZE,MSIZE,MSIZE,MSIZE,BMSIZE]

    data1=[
        [None,80.41,None,None,None,79.00],
        #[61.22,58.89,51.07,58.07,49.63,25.50],
        [91.36,85.93,86.41,83.02,90.37,83.70],
        [92.81,87.41,87.46,86.31,88.89,81.80],
        [95.86,90.37,90.63,88.32,93.33,87.30],
        [93.90,89.11,89.27,88.06,90.81,85.30],
        [88.26,85.19,85.44,83.71,88.89,80.20],
        [93.83,90.37,90.66,87.69,94.07,86.30],
        [92.98,91.11,91.58,87.43,96.30,87.00]
    ]

    plt.figure(figsize=figsize)

    for i in range(len(data1)):
        plt.scatter(x, data1[i], marker=markers[i], label=labels[i], c=colors[i], s=sizes[i], zorder=1)

    for i in range(len(x)):
        plt.plot([i,i],[70,100],c='gray',alpha=0.3,zorder=0)
    
    plt.legend(loc=3, prop={'family':'Arial','size':LEGEND_SIZE})
    plt.tick_params(axis='x')
    #plt.ylim(70,100)
    plt.ylim(75,100)
    plt.xticks(family='Arial',size=TICK_SIZE,rotation=20)
    #plt.yticks((70,80,90,100),('70%','80%','90%','100%'),family='Arial',size=TICK_SIZE)
    plt.yticks((80,90,100),('80%','90%','100%'),family='Arial',size=TICK_SIZE)
    plt.savefig('scatter1_8.png',dpi=dpi)
    #plt.show()


if SCATTER:
    x =  ['AUC','Accuracy','F1-Score','Precision','Recall','Correlation','OPP'] #,'MSE']

    labels  = ['SeqVec','ProtTrans T5','ESM-1b','ESM-1v','ESM-2(650M)','ProtFound-BA']

    #colors  = ['c', 'b', 'g', 'y', 'orange', 'r', 'm', 'k', 'gray']
    #markers = ['o','v','>','D','<','*','s','^','h']
    #colors = [BLUE1_edge, PURPLE_edge, ORANGE_edge, GREEN1_edge, BLUE2_edge, RED_edge]

    markers= ['v','>','D','<','s','*']
    colors= [PURPLE_edge, ORANGE_edge, GREEN1_edge, BLUE2_edge, GREEN2_edge, RED_edge]


    sizes   = [MSIZE,MSIZE,MSIZE,MSIZE,MSIZE,BMSIZE,MSIZE,MSIZE,MSIZE]

    data1=[
        [86.68,79.50,80.51,76.86,84.60,89.80,73.15],
        [84.66,77.33,79.37,72.97,87.13,88.20,72.04],
        [84.93,77.17,77.54,76.37,78.93,93.70,73.27],
        [85.27,78.23,79.17,76.14,82.77,92.20,70.89],
        [83.25,75.83,76.65,74.21,79.33,90.30,66.42],
        [87.37,79.83,79.97,79.49,80.53,91.20,75.34]
    ]

    plt.figure(figsize=figsize)

    for i in range(len(data1)):
        plt.scatter(x, data1[i], marker=markers[i], label=labels[i], c=colors[i], s=sizes[i], zorder=1)

    for i in range(len(x)):
        plt.plot([i,i],[64,95],c='gray',alpha=0.3,zorder=0)
    
    plt.legend(loc=3, prop={'family':'Arial','size':LEGEND_SIZE})
    plt.tick_params(axis='x')
    #plt.ylim(64,95)
    plt.ylim(66,95)
    plt.xticks(family='Arial',size=TICK_SIZE,rotation=20)
    plt.yticks((65,75,85,95),('65%','75%','85%','95%'),family='Arial',size=TICK_SIZE)
    plt.savefig('scatter2.png',dpi=dpi)
    #plt.show()










RADAR2=0
RSIZE=31
dpi=500
figsize=(18,10)

LINEWIDTH=5

if RADAR2:
    data_labels = np.array(['Multi-task Focal Loss', 'BCE & MSE'])
    n = 6
    radar_labels = np.array(['       AUC', '             Accuracy', 'F1-Score              ',
                        'Precision              ', 'Recall            ', '                  Correlation'])
    data=np.array([[92.98, 84.06],
                [91.11, 57.41],
                [91.58, 26.33],
                [87.43, 97.14],
                [96.30, 15.56],
                [87.00, 87.10]])
    angles=np.linspace(0, 2*np.pi, n, endpoint=False)
    data=np.concatenate((data, [data[0]]))
    angles=np.concatenate((angles, [angles[0]]))

    #fig = plt.figure(facecolor='white', figsize=(15,8))
    #plt.subplot(111, polar=True)
    plt.figure(facecolor='white', figsize=figsize)
    plt.polar()
    plt.thetagrids(angles*180/np.pi, radar_labels, family='Arial',size=RSIZE)
    plt.plot(angles, data[:,0], 'o-', linewidth=LINEWIDTH, c=BLUE1_edge) #alpha0.2
    plt.fill(angles, data[:,0], c=BLUE1_in, alpha=0.8) #alpha 0.25
    plt.plot(angles, data[:,1], 'o-', linewidth=LINEWIDTH, c=ORANGE_edge) #alpha0.2
    plt.fill(angles, data[:,1], c=ORANGE_in, alpha=0.8) #alpha 0.25
    #plt.legend(data_labels, loc=(0.94,0.8), prop={'family':'Arial','size':16})
    plt.grid(True, linewidth=1.5)
    plt.rgrids([20,40,60,80,100],['20%','40%','60%','80%','100%'],size=25)
    #plt.show()
    ax=plt.gca()
    ax.spines['polar'].set_linewidth(2)
    plt.savefig('radar_1.svg', dpi=dpi)

if RADAR2:
    data_labels = np.array(['Multi-task Focal Loss', 'BCE & RegFocalLoss'])
    n = 6
    radar_labels = np.array(['       AUC', '             Accuracy', 'F1-Score              ',
                        'Precision              ', 'Recall            ', '                  Correlation'])
    data=np.array([[92.98, 88.59],
                [91.11, 58.52],
                [91.58, 28.93],
                [87.43, 100.0],
                [96.30, 17.04],
                [87.00, 87.30]])
    angles=np.linspace(0, 2*np.pi, n, endpoint=False)
    data=np.concatenate((data, [data[0]]))
    angles=np.concatenate((angles, [angles[0]]))

    #fig = plt.figure(facecolor='white', figsize=(15,8))
    #plt.subplot(111, polar=True)
    plt.figure(facecolor='white', figsize=figsize)
    plt.polar()
    plt.thetagrids(angles*180/np.pi, radar_labels, family='Arial',size=RSIZE)
    plt.plot(angles, data[:,0], 'o-', linewidth=LINEWIDTH, c=BLUE1_edge) #alpha0.2
    plt.fill(angles, data[:,0], c=BLUE1_in, alpha=0.8) #alpha 0.25
    plt.plot(angles, data[:,1], 'o-', linewidth=LINEWIDTH, c=PURPLE_edge) #alpha0.2
    plt.fill(angles, data[:,1], c=PURPLE_in, alpha=0.35) #alpha 0.25
    #plt.legend(data_labels, loc=(0.94,0.8), prop={'family':'Arial','size':16})
    plt.grid(True, linewidth=1.5)
    plt.rgrids([20,40,60,80,100],['20%','40%','60%','80%','100%'],size=25)
    #plt.show()
    ax=plt.gca()
    ax.spines['polar'].set_linewidth(2)
    plt.savefig('radar_2.svg', dpi=dpi)

if RADAR2:
    data_labels = np.array(['Multi-task Focal Loss', 'BCE & MSE', 'BCE & RegFocalLoss', 'MSE & ClsFocalLoss'])
    n = 6
    radar_labels = np.array(['       AUC', '             Accuracy', 'F1-Score              ',
                        'Precision              ', 'Recall            ', '                  Correlation'])
    data=np.array([[92.98, None, None, 93.69],
                [91.11, None, None, 85.56],
                [91.58, None, None, 84.92],
                [87.43, None, None, 88.28],
                [96.30, None, None, 82.22],
                [87.00, None, None, 87.90]])
    angles=np.linspace(0, 2*np.pi, n, endpoint=False)
    data=np.concatenate((data, [data[0]]))
    angles=np.concatenate((angles, [angles[0]]))

    #fig = plt.figure(facecolor='white', figsize=(15,8))
    #plt.subplot(111, polar=True)
    plt.figure(facecolor='white', figsize=figsize)
    plt.polar()
    plt.thetagrids(angles*180/np.pi, radar_labels, family='Arial',size=RSIZE)

    plt.plot(angles, data[:,0], 'o-', linewidth=LINEWIDTH, c=BLUE1_edge) #alpha0.2
    plt.fill(angles, data[:,0], c=BLUE1_in, alpha=0.8) #alpha 0.25

    plt.plot(angles, data[:,1], 'o-', linewidth=LINEWIDTH, c=ORANGE_edge, alpha=0.8) #alpha0.2
    plt.plot(angles, data[:,2], 'o-', linewidth=LINEWIDTH, c=PURPLE_edge, alpha=0.35) #alpha0.2
    
    plt.plot(angles, data[:,3], 'o-', linewidth=LINEWIDTH, c=GREEN1_edge) #alpha0.2
    plt.fill(angles, data[:,3], c=GREEN1_in, alpha=0.6) #alpha 0.25

    plt.legend(data_labels, 
               loc= (1.0,0.8), #(0.94,0.8), 
               prop={'family':'Arial','size':25})
    plt.grid(True, linewidth=1.5)
    plt.rgrids([20,40,60,80,100],['20%','40%','60%','80%','100%'],size=25)
    #plt.show()
    ax=plt.gca()
    ax.spines['polar'].set_linewidth(2)
    plt.savefig('radar_3_legend.svg', dpi=dpi)
