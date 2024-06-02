import os
import numpy as np


import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


add='pca'
name='PCA'

YELLOW12= '#DAB47F' #'#B1B1D7' #'#609E6F' 3
BLUE1= '#A0BBDA' #'#DAA6C3' #'#B45E5F' 1
RED2= '#DAA6C3' #'#B45E5F'
GREEN2= '#609E6F' #'#C6E0B4' #'#CA8963' 2

DPI=500
SIZE=3
NSTD=2.5
ALPHA= 0.2 #0.45 #0.3
LEGEND=7

RANGE=True
TICKS=False

feature_bn=np.load('data/PCA_visualization/bind/feature_before_{}.npy'.format(add))
feature_an=np.load('data/PCA_visualization/bind/feature_after_{}.npy'.format(add))
label_bn=np.load('data/PCA_visualization/bind/reg_label.npy')


def plot_point_cov(points, nstd=3, ax=None, **kwargs):
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)
 
def plot_cov_ellipse(cov, pos, nstd=3, ax=None, **kwargs):
    def eigsorted(cov):
        cov = np.array(cov)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]
 
    vals, vecs = eigsorted(cov)
 
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellip)




if 0:
    fig = plt.figure(num=2, figsize=(10,4),dpi=DPI)

    mask1=(label_bn>1)
    mask2=(label_bn<=1)

    ax = fig.add_subplot(1,2,1)

    pts=feature_bn[mask2,:]
    newx,newy=pts[:,0],pts[:,1]
    ax.set_title('ACE2 Binding: {} Before Feature Coupling'.format(name),fontdict={'size':12,'family':'Arial','weight':'bold'})
    plot_point_cov(pts, nstd=NSTD, ax=ax, alpha=ALPHA, color=BLUE1)
    ax.scatter(newx,newy,s=SIZE,c=BLUE1,label='Harmful mutations',zorder=1)


    pts=feature_bn[mask1,:]
    newx,newy=pts[:,0],pts[:,1]
    plot_point_cov(pts, nstd=NSTD, ax=ax, alpha=ALPHA, color=YELLOW12)
    ax.scatter(newx,newy,s=SIZE,c=YELLOW12,label='Beneficial mutations',zorder=2)

    if LEGEND:
        plt.xlim((-2.5,3))
        plt.ylim((-2,3))
    if not TICKS:
        plt.xticks((),fontproperties='Arial')
        plt.yticks((),fontproperties='Arial')
    ax.legend(prop={'family':'Arial','size':LEGEND,'weight':'bold'},loc=2)


    ax = fig.add_subplot(1,2,2)

    pts=feature_an[mask2,:]
    newx,newy=pts[:,0],pts[:,1]
    ax.set_title('ACE2 Binding: {} After Feature Coupling'.format(name),fontdict={'size':12,'family':'Arial','weight':'bold'})
    plot_point_cov(pts, nstd=NSTD, ax=ax, alpha=ALPHA, color=BLUE1)
    ax.scatter(newx,newy,s=SIZE,c=BLUE1,label='Harmful mutations',zorder=1)


    pts=feature_an[mask1,:]
    newx,newy=pts[:,0],pts[:,1]
    plot_point_cov(pts, nstd=NSTD, ax=ax, alpha=ALPHA, color=YELLOW12)
    ax.scatter(newx,newy,s=SIZE,c=YELLOW12,label='Beneficial mutations',zorder=2)

    if LEGEND:
        plt.xlim((-100,150))
        plt.ylim((-75,100))

    if not TICKS:
        plt.xticks((),fontproperties='Arial')
        plt.yticks((),fontproperties='Arial')
    ax.legend(prop={'family':'Arial','size':LEGEND,'weight':'bold'},loc=2)

    plt.show()
    #fig.savefig('4k_{}2.png'.format(add))
    fig.clear()


if 1:
    fig = plt.figure(num=2, figsize=(10,4),dpi=DPI)

    mask1=(label_bn>1)
    mask2=(label_bn>0.5)&(label_bn<=1)
    mask3=(label_bn<=0.5)

    ax = fig.add_subplot(1,2,1)

    pts=feature_bn[mask3,:]
    newx,newy=pts[:,0],pts[:,1]
    ax.set_title('ACE2 binding: PCA before dependence coupling',fontdict={'size':12,'family':'Arial','weight':'bold'})
    plot_point_cov(pts, nstd=NSTD, ax=ax, alpha=ALPHA, color=RED2)
    ax.scatter(newx,newy,s=SIZE,c=RED2,label='Risk-free (<0.5)',zorder=1)

    pts=feature_bn[mask2,:]
    newx,newy=pts[:,0],pts[:,1]
    plot_point_cov(pts, nstd=NSTD, ax=ax, alpha=ALPHA, color=GREEN2)
    ax.scatter(newx,newy,s=SIZE,c=GREEN2,label='Risk-free (0.5-1)',zorder=2)

    pts=feature_bn[mask1,:]
    newx,newy=pts[:,0],pts[:,1]
    plot_point_cov(pts, nstd=NSTD, ax=ax, alpha=ALPHA, color=YELLOW12)
    ax.scatter(newx,newy,s=SIZE,c=YELLOW12,label='Risky (>1)',zorder=3)

    if LEGEND:
        plt.xlim((-2.5,3))
        plt.ylim((-2,3))

    if not TICKS:
        plt.xticks((),fontproperties='Arial')
        plt.yticks((),fontproperties='Arial')
    ax.legend(prop={'family':'Arial','size':LEGEND,'weight':'bold'},loc=2)


    ax = fig.add_subplot(1,2,2)

    pts=feature_an[mask3,:]
    newx,newy=pts[:,0],pts[:,1]
    ax.set_title('ACE2 binding: PCA after dependence coupling',fontdict={'size':12,'family':'Arial','weight':'bold'})
    plot_point_cov(pts, nstd=NSTD, ax=ax, alpha=ALPHA, color=RED2)
    ax.scatter(newx,newy,s=SIZE,c=RED2,label='Risk-free (<0.5)',zorder=1)

    pts=feature_an[mask2,:]
    newx,newy=pts[:,0],pts[:,1]
    plot_point_cov(pts, nstd=NSTD, ax=ax, alpha=ALPHA, color=GREEN2)
    ax.scatter(newx,newy,s=SIZE,c=GREEN2,label='Risk-free (0.5-1)',zorder=2)

    pts=feature_an[mask1,:]
    newx,newy=pts[:,0],pts[:,1]
    plot_point_cov(pts, nstd=NSTD, ax=ax, alpha=ALPHA, color=YELLOW12)
    ax.scatter(newx,newy,s=SIZE,c=YELLOW12,label='Risky (>1)',zorder=3)

    if LEGEND:
        plt.xlim((-100,150))
        plt.ylim((-75,100))

    if not TICKS:
        plt.xticks((),fontproperties='Arial')
        plt.yticks((),fontproperties='Arial')
    ax.legend(prop={'family':'Arial','size':LEGEND,'weight':'bold'},loc=2)

    #plt.show()
    fig.savefig('bind_{}3.svg'.format(add))