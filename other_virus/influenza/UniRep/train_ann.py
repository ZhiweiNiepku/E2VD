from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, mean_squared_error
import os
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import KFold
import math
import time
import random


os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CUDA_LAUNCH_BLOCKING']='1'

device=torch.device('cuda')

class ANN(nn.Module):
    r"""An artificial neural network (ANN) for predicting 

    Parameters
    ----------
    in_features : int
        Number of features in input
    out_units : list
        List of layers with each element detailing number of neurons in each layer, e.g. two hidden layers: [16, 32]
    n_out : int
        Number of units in output layer
    p_dropout : float
        Probability of dropout, by default 0
    activation : nn.Activation
        A PyTorch activation function, by default nn.ReLU()

    Examples
    ----------
        >>> net = ANN(in_features=189, out_units=[16], n_out=1)
        >>> print(net)
            ANN(
                (fc): Sequential(
                    (fc0): Linear(in_features=189, out_features=16, bias=True)
                    (act0): ReLU()
                    (dropout): Dropout(p=0, inplace=False)
                )
                (fc_out): Linear(in_features=16, out_features=1, bias=True)
            )
    """

    def __init__(self, in_features = 1900, out_units = [16], p_dropout=0.14, activation=nn.ReLU()):
        super(ANN, self).__init__()

        # save args
        self.in_features = in_features
        self.out_units = out_units
        self.in_units = [self.in_features] + self.out_units[:-1]
        self.n_layers = len(self.out_units)

        # build the input and hidden layers
        self.fc = nn.Sequential()
        def add_linear(i):
            """Add n linear layers to the ANN"""
            self.fc.add_module('fc{}'.format(i), nn.Linear(self.in_units[i], self.out_units[i]))
            self.fc.add_module('{}{}'.format('act', i), activation)

        for i in range(self.n_layers):
            add_linear(i)
        
        # add dropout before final
        self.fc.add_module('dropout', nn.Dropout(p_dropout))

        # add final output layer
        self.fc_out_reg = nn.Linear(self.out_units[-1], 1)
        self.fc_out_cls = nn.Linear(self.out_units[-1], 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x): # (batch_size, in_units)

        o = self.fc(x) # (batch_size, out_units)
        o_cls = self.sigmoid(self.fc_out_cls(o)) # (batch_size, n_out)
        o_reg = self.fc_out_reg(o) # (batch_size, n_out) -> range now between 0 and 1

        return o_cls, o_reg

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class ScaleDataset(torch.utils.data.Dataset):
    def __init__(self, feats, cls_label, reg_label):
        self.feats = torch.from_numpy(feats).float()
        self.cls_labels = torch.from_numpy(cls_label).float()
        self.reg_labels = torch.from_numpy(reg_label).float()

    def __getitem__(self, index):
        return self.feats[index], self.cls_labels[index], self.reg_labels[index]
    def __len__(self):
        return len(self.reg_labels)


#######################################
# new loss define
from torch.nn.modules.loss import _Loss

class WeightMSELoss(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(WeightMSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target, k):
        input=input.reshape(1,-1)
        target=target.reshape(1,-1)
        mask=(target>=1)
        #print(input)
        return (torch.sum((input*mask-target*mask)**2)*k+torch.sum((input*(~mask)-target*(~mask))**2))/len(input)

class RegFocalLoss(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(RegFocalLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, pred, target, gamma=1):
        pred=pred.reshape(1,-1)
        target=target.reshape(1,-1)
        se_=torch.abs(pred-target)
        a_=torch.pow(se_,gamma).detach()
        a_sum=torch.sum(a_).detach()
        a_=(a_/a_sum).detach()
        return torch.sum(torch.pow(se_,2)*a_)

class ClsFocalLoss(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(ClsFocalLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, pred, target, gamma=1, alpha=0.5):
        assert alpha<1 and alpha>0

        epsilon=1e-7
        pred=pred.reshape(-1,)
        target=target.reshape(-1,)

        pt_0=1-pred[target==0]
        pt_1=pred[target==1]

        loss_0=(-torch.pow(1-pt_0,gamma)*torch.log(pt_0+epsilon)).sum()
        loss_1=(-torch.pow(1-pt_1,gamma)*torch.log(pt_1+epsilon)).sum()

        loss=(1-alpha)*loss_0+alpha*loss_1

        return loss/len(pred)

#######################################

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
    BATCH_SIZE = 32
    EPOCH = 150
    SEQ_LEN=384
    REG_LOSS_K=1

    lr=2e-4

    cls_loss_func= nn.BCELoss() #ClsFocalLoss()
    reg_loss_func= nn.MSELoss() #RegFocalLoss()

    k=5
    
    test_vc_all =[]
    test_precision_all=[]
    test_recall_all=[]
    test_f1_all=[]
    test_auc_all=[]
    test_mse_all=[]
    test_corr_all=[]

    res_dir = "result_{}_lr{}_regk{}_k{}_f1_prop".format(BATCH_SIZE,lr,REG_LOSS_K,k)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    ##############
    # load data

    feats=np.load('data/other_virus/influenza/unirep/1900_train_data.npy')
    reg_label=np.load('data/other_virus/influenza/unirep/1900_train_label.npy').reshape(-1,1)

    cls_label=(reg_label>=0).reshape(-1,1)

    print(feats.shape)
    print(reg_label.shape)
    

    #kf=KFold(n_splits=k,shuffle=True,random_state=27)
    #kf_idx=kf.split(feats,reg_label)
    kf_idx=MyFold(cls_label,k,70)

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

        model = ANN()
        optimizer = torch.optim.Adam(model.parameters(),lr)
        model = model.to(device) #.cuda()

        train_feats=feats[i_idx[0]]
        train_reg_label=reg_label[i_idx[0]]
        train_cls_label=cls_label[i_idx[0]]

        val_feats=feats[i_idx[1]]
        val_reg_label=reg_label[i_idx[1]]
        val_cls_label=cls_label[i_idx[1]]


        # scaler=StandardScaler()
        # shape_=train_feats.shape
        # train_feats=train_feats.reshape(shape_[0],-1)
        # scaler.fit(train_feats)
        # train_feats=scaler.transform(train_feats).reshape(shape_)
        # shape_=val_feats.shape
        # val_feats=val_feats.reshape(shape_[0],-1)
        # val_feats=scaler.transform(val_feats).reshape(shape_)

        Train_dataset = ScaleDataset(train_feats,train_cls_label,train_reg_label)
        train_loader_single = DataLoader(dataset=Train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        Val_dataset = ScaleDataset(val_feats,val_cls_label,val_reg_label)
        val_loader_single = DataLoader(dataset=Val_dataset, batch_size=16, shuffle=False)

        # training and testing
        for epoch in range(EPOCH):
            total_loss = []

            # training
            model.train()
            for step, (feat_, cls_label_, reg_label_) in enumerate(train_loader_single):
                cls_label_ = cls_label_.to(device)
                reg_label_ = reg_label_.to(device)
                feat_=feat_.to(device)
                cls_output, reg_output = model(feat_)
                cls_output=cls_output.squeeze()
                reg_output=reg_output.squeeze()


                cls_loss = cls_loss_func(cls_output.flatten().float(), cls_label_.flatten().float())
                reg_loss = reg_loss_func(reg_output.flatten().float(), reg_label_.flatten().float())
                #print(cls_loss.item(),reg_loss.item())
                loss=cls_loss + reg_loss*REG_LOSS_K
                total_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 40 == 0:
                    print(">>>>>loss:", sum(total_loss) / len(total_loss))

            # testing
            model.eval()
            with torch.no_grad():
                y_cls = []
                y_reg = []
                for step, (feat_, cls_label_, reg_label_) in enumerate(val_loader_single):
                    #label_ = label_.to(device) #.cuda()
                    feat_=feat_.to(device) #.cuda()
                    if device.type=='cpu':
                        cls_output, reg_output = model(feat_)#.data.numpy().squeeze()
                        cls_output=cls_output.data.numpy().squeeze()
                        reg_output=reg_output.data.numpy().squeeze()
                    else:
                        cls_output, reg_output = model(feat_)#.cpu().data.numpy().squeeze()
                        cls_output=cls_output.cpu().data.numpy().squeeze()
                        reg_output=reg_output.cpu().data.numpy().squeeze()
                    y_cls.append(cls_output.reshape(-1,1))
                    y_reg.append(reg_output.reshape(-1,1))

                # metrics for classification
                y_cls = np.concatenate(y_cls,axis=0).flatten()
                y_cls_bool=(y_cls>0.5)
                val_cls_label=val_cls_label.flatten().astype(bool)
                cm_=confusion_matrix(val_cls_label,y_cls_bool)
                vc_=(cm_[0][0]+cm_[1][1])/(cm_[0][0]+cm_[0][1]+cm_[1][0]+cm_[1][1])
                precision_=precision_score(val_cls_label,y_cls_bool) #cm_[1][1]/(cm_[0][1]+cm_[1][1])
                recall_=recall_score(val_cls_label,y_cls_bool) #cm_[1][1]/(cm_[1][0]+cm_[1][1])
                f1_=f1_score(val_cls_label,y_cls_bool)
                auc_=roc_auc_score(val_cls_label,y_cls)

                print('Epoch: {}, auc: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}\n'.format(epoch,auc_,vc_,precision_,recall_,f1_))
                f.write('fold: {}, Epoch: {}, auc: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}\n'.format(counter,epoch,auc_,vc_,precision_,recall_,f1_))

                # metrics for regression
                y_reg = np.concatenate(y_reg,axis=0).flatten()
                val_reg_label=val_reg_label.flatten()
                mse_=mean_squared_error(val_reg_label,y_reg)
                corr_=np.corrcoef(val_reg_label,y_reg)[0][1]

                print('Epoch: {}, mse: {:.3f}, corr: {:.3f}\n'.format(epoch,mse_,corr_))
                f.write('fold: {}, Epoch: {}, mse: {:.3f}, corr: {:.3f}\n'.format(counter,epoch,mse_,corr_))

                #if auc_ > auc_best:
                #if f1_ > f1_best:
                if mse_<mse_best:
                    auc_best=auc_
                    vc_best=vc_
                    precision_best=precision_
                    recall_best=recall_
                    f1_best=f1_
                    cm_best=cm_

                    mse_best=mse_
                    corr_best=corr_

                    # save model
                    save_path = os.path.join(res_dir, 'model_{}.pth.tar'.format(counter))
                    torch.save({'state_dict': model.state_dict()}, save_path)
                    print('Epoch {}, mse best: {}\n'.format(epoch,mse_best))
                    f.write('Epoch {}, mse best: {}\n'.format(epoch,mse_best))

        f.write('\nfold {}, auc: {:.4f}, accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}\n'.format(counter,auc_best,vc_best,
                                                                                                                precision_best,recall_best,
                                                                                                                f1_best))
        f.write('fold {}, mse: {:.4f}, corr: {:.4f}\n'.format(counter,mse_best,corr_best))
        f.write('fold {}, \n[[{},{}],\n[{},{}]]\n\n\n'.format(counter,cm_best[0][0],cm_best[0][1],cm_best[1][0],cm_best[1][1]))

        test_precision_all.append(precision_best)
        test_recall_all.append(recall_best)
        test_f1_all.append(f1_best)
        test_auc_all.append(auc_best)
        test_vc_all.append(vc_best)
        test_mse_all.append(mse_best)
        test_corr_all.append(corr_best)
        f.close()

        counter+=1

    precision_mean=np.mean(test_precision_all)
    recall_mean=np.mean(test_recall_all)
    f1_mean=np.mean(test_f1_all)
    auc_mean=np.mean(test_auc_all)
    vc_mean=np.mean(test_vc_all)
    mse_mean=np.mean(test_mse_all)
    corr_mean=np.mean(test_corr_all)
    f = open(os.path.join(res_dir, 'result.txt'), 'a')
    f.write('total, auc: {:.4f}, accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}\n'.format(auc_mean,vc_mean,
                                                                                                            precision_mean,recall_mean,
                                                                                                            f1_mean))
    f.write('total, mse: {:.4f}, corr: {:.4f}\n'.format(mse_mean,corr_mean))
    f.write('end time: {}'.format(time.ctime()))

    f.close()
