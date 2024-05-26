import os
import random
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
import math
import time
import random


#######################################
# sampling
def list_remove_(input,remove):
    input_=input.copy()
    for i in remove:
        input_.remove(i)
    return input_

def MyFold(label,k=5,num=50):
    random.seed(27)
    ret=[]
    label=label.reshape(-1,)

    pos_idxs=np.where(label==1)[0]
    neg_idxs=np.where(label==0)[0]

    pos_chosen_idxs_all=random.sample(pos_idxs.tolist(),k*num)
    neg_chosen_idxs_all=random.sample(neg_idxs.tolist(),k*num)

    for i in range(k):
        pos_train_=list_remove_(pos_idxs.tolist(),pos_chosen_idxs_all[i*num:i*num+num])
        neg_train_=list_remove_(neg_idxs.tolist(),neg_chosen_idxs_all[i*num:i*num+num])
        train_=pos_train_+neg_train_
        val_=pos_chosen_idxs_all[i*num:i*num+num]+neg_chosen_idxs_all[i*num:i*num+num]
        ret.append([train_,val_])
    
    return ret

########################################



if __name__ == '__main__':
    SEQ_LEN=201

    k=5
    
    val_vc_all =[]
    val_precision_all=[]
    val_recall_all=[]
    val_f1_all=[]
    val_auc_all=[]
    val_mse_all=[]
    val_corr_all=[]
    
    
    test_vc_all =[]
    test_precision_all=[]
    test_recall_all=[]
    test_f1_all=[]
    test_auc_all=[]
    test_mse_all=[]
    test_corr_all=[]

    res_dir = "result_rf_k{}_".format(k)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    ##############
    # load data

    feats=np.load('data/1900_train_data.npy')
    reg_label=np.load('data/1900_train_label.npy').reshape(-1,1)
    cls_label=(reg_label>=1).reshape(-1,1)

    print(feats.shape)
    print(reg_label.shape)
    
    # load test data
    test_feats=np.load('data/1900_test_data.npy')
    test_reg_label=np.load('data/1900_test_label.npy')
    test_cls_label=(test_reg_label>1)

    print(test_feats.shape)
    print(test_reg_label.shape)

    #kf=KFold(n_splits=k,shuffle=True,random_state=27)
    #kf_idx=kf.split(feats,reg_label)
    kf_idx=MyFold(cls_label,k,60)

    counter=0

    #for i_idx in kf_idx:
    for i_ in range(k):
        i_idx=kf_idx[i_]

        f = open(os.path.join(res_dir, 'result.txt'), 'a')
        print('fold',counter)
        f.write('fold {}, {}\n'.format(counter,time.ctime()))
        
        vc_best=0
        precision_best=0
        recall_best=0
        f1_best=0
        auc_best=0
        cm_best=None

        mse_best=20
        corr_best=0

        model_cls = RandomForestClassifier(bootstrap=False, max_features='sqrt', min_samples_split=5, n_estimators=100)
        model_reg = RandomForestRegressor(bootstrap=False, max_features='sqrt', min_samples_split=5, n_estimators=100)

        train_feats=feats[i_idx[0]]
        train_reg_label=reg_label[i_idx[0]]
        train_cls_label=cls_label[i_idx[0]]

        val_feats=feats[i_idx[1]]
        val_reg_label=reg_label[i_idx[1]]
        val_cls_label=cls_label[i_idx[1]]

        model_cls.fit(train_feats, train_cls_label.flatten())
        model_reg.fit(train_feats, train_reg_label.flatten())

        y_cls = model_cls.predict(val_feats)
        y_reg = model_reg.predict(val_feats)
        
        # print(y_cls.shape)
        # print(y_reg.shape)
        # print(val_cls_label.shape)
        # print(val_reg_label.shape)

        # metrics for classification
        y_cls_bool=(y_cls>0.5)
        val_cls_label=val_cls_label.flatten().astype(bool)
        cm_best=confusion_matrix(val_cls_label,y_cls_bool)
        vc_best=(cm_best[0][0]+cm_best[1][1])/(cm_best[0][0]+cm_best[0][1]+cm_best[1][0]+cm_best[1][1])
        precision_best=precision_score(val_cls_label,y_cls_bool) #cm_[1][1]/(cm_[0][1]+cm_[1][1])
        recall_best=recall_score(val_cls_label,y_cls_bool) #cm_[1][1]/(cm_[1][0]+cm_[1][1])
        f1_best=f1_score(val_cls_label,y_cls_bool)
        auc_best=roc_auc_score(val_cls_label,y_cls)

        # metrics for regression
        val_reg_label=val_reg_label.flatten()
        mse_best=mean_squared_error(val_reg_label,y_reg)
        corr_best=np.corrcoef(val_reg_label,y_reg)[0][1]

        f.write('\nval fold {}, auc: {:.4f}, accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}\n'.format(counter,auc_best,vc_best,
                                                                                                                precision_best,recall_best,
                                                                                                                f1_best))
        f.write('val fold {}, mse: {:.4f}, corr: {:.4f}\n'.format(counter,mse_best,corr_best))
        f.write('val fold {}, \n[[{},{}],\n[{},{}]]\n\n\n'.format(counter,cm_best[0][0],cm_best[0][1],cm_best[1][0],cm_best[1][1]))

        val_precision_all.append(precision_best)
        val_recall_all.append(recall_best)
        val_f1_all.append(f1_best)
        val_auc_all.append(auc_best)
        val_vc_all.append(vc_best)
        val_mse_all.append(mse_best)
        val_corr_all.append(corr_best)
        
        
        
        y_cls = model_cls.predict(test_feats)
        y_reg = model_reg.predict(test_feats)
        
        y_cls_bool=(y_cls>0.5)
        # metrics for classification
        test_cls_label=test_cls_label.flatten().astype(bool)
        cm_test=confusion_matrix(test_cls_label,y_cls_bool)
        vc_test=(cm_test[0][0]+cm_test[1][1])/(cm_test[0][0]+cm_test[0][1]+cm_test[1][0]+cm_test[1][1])
        precision_test=precision_score(test_cls_label,y_cls_bool) #cm_[1][1]/(cm_[0][1]+cm_[1][1])
        recall_test=recall_score(test_cls_label,y_cls_bool) #cm_[1][1]/(cm_[1][0]+cm_[1][1])
        f1_test=f1_score(test_cls_label,y_cls_bool)
        auc_test=roc_auc_score(test_cls_label,y_cls)

        # metrics for regression
        test_reg_label=test_reg_label.flatten()
        mse_test=mean_squared_error(test_reg_label,y_reg)
        corr_test=np.corrcoef(test_reg_label,y_reg)[0][1]

        f.write('\ntest fold {}, auc: {:.4f}, accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}\n'.format(counter,auc_test,vc_test,
                                                                                                                precision_test,recall_test,
                                                                                                                f1_test))
        f.write('test fold {}, mse: {:.4f}, corr: {:.4f}\n'.format(counter,mse_test,corr_test))
        f.write('test fold {}, \n[[{},{}],\n[{},{}]]\n\n\n'.format(counter,cm_test[0][0],cm_test[0][1],cm_test[1][0],cm_test[1][1]))

        test_precision_all.append(precision_test)
        test_recall_all.append(recall_test)
        test_f1_all.append(f1_test)
        test_auc_all.append(auc_test)
        test_vc_all.append(vc_test)
        test_mse_all.append(mse_test)
        test_corr_all.append(corr_test)
        
        
        
        
        
        
        f.close()

        counter+=1


    precision_mean=np.mean(val_precision_all)
    recall_mean=np.mean(val_recall_all)
    f1_mean=np.mean(val_f1_all)
    auc_mean=np.mean(val_auc_all)
    vc_mean=np.mean(val_vc_all)
    mse_mean=np.mean(val_mse_all)
    corr_mean=np.mean(val_corr_all)
    f = open(os.path.join(res_dir, 'result.txt'), 'a')
    f.write('val total, auc: {:.4f}, accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}\n'.format(auc_mean,vc_mean,
                                                                                                            precision_mean,recall_mean,
                                                                                                            f1_mean))
    f.write('val total, mse: {:.4f}, corr: {:.4f}\n'.format(mse_mean,corr_mean))
    f.close()







    precision_mean=np.mean(test_precision_all)
    recall_mean=np.mean(test_recall_all)
    f1_mean=np.mean(test_f1_all)
    auc_mean=np.mean(test_auc_all)
    vc_mean=np.mean(test_vc_all)
    mse_mean=np.mean(test_mse_all)
    corr_mean=np.mean(test_corr_all)
    f = open(os.path.join(res_dir, 'result.txt'), 'a')
    f.write('test total, auc: {:.4f}, accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}\n'.format(auc_mean,vc_mean,
                                                                                                            precision_mean,recall_mean,
                                                                                                            f1_mean))
    f.write('test total, mse: {:.4f}, corr: {:.4f}\n'.format(mse_mean,corr_mean))
    f.write('end time: {}'.format(time.ctime()))

    f.close()
