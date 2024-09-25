from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, mean_squared_error
import os
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import math
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES']='0'

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
    def __init__(self,feats,cls_label, reg_label):
        self.feats = torch.from_numpy(feats).float()
        self.cls_labels = torch.from_numpy(cls_label).float()
        self.reg_labels = torch.from_numpy(reg_label).float()

    def __getitem__(self, index):
        return self.feats[index], self.cls_labels[index], self.reg_labels[index]
    def __len__(self):
        return len(self.reg_labels)


if __name__ == '__main__':

    BATCH_SIZE = 32
    SEQ_LEN=504
    model_dir="<trained path>/model_{}.pth.tar"

    model = ANN()
    model = model.to(device)

    if 1:
        # load test data
        test_feats=np.load('data/other_virus/zika/uniref/1900_test_data.npy')
        test_reg_label=np.load('data/other_virus/zika/uniref/1900_test_label.npy') / 10
        test_cls_label=(test_reg_label>0)
        # shape_=test_feats.shape
        # test_feats=test_feats.reshape(shape_[0],-1)
        # test_feats=scaler.transform(test_feats).reshape(shape_)

    Test_dataset = ScaleDataset(test_feats,test_cls_label,test_reg_label)
    test_loader = DataLoader(dataset=Test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model_n=5

    test_vc_all=[]
    test_mse_all=[]
    test_corr_all=[]

    test_precision_all=[]
    test_recall_all=[]
    test_f1_all=[]
    test_auc_all=[]

    # testing
    model.eval()
    with torch.no_grad():
        for model_i in range(model_n):

            print('fold',model_i)

            model_path=model_dir.format(model_i)
            model.load_state_dict(torch.load(model_path)['state_dict'])

            y_cls = []
            y_reg = []
            for step, (feat_, cls_label_, reg_label_) in enumerate(test_loader):
                feat_=feat_.to(device)
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

            
            # metrics for regression
            y_reg = np.concatenate(y_reg,axis=0).flatten()
            test_reg_label=test_reg_label.flatten()
            mse_=mean_squared_error(test_reg_label,y_reg)
            corr_=np.corrcoef(test_reg_label,y_reg)[0][1]


            # metrics for classification
            y_cls = np.concatenate(y_cls,axis=0).flatten()
            y_cls_bool=(y_cls>0.5)
            test_cls_label=test_cls_label.flatten().astype(bool)
            cm_=confusion_matrix(test_cls_label,y_cls_bool)
            vc_=(cm_[0][0]+cm_[1][1])/(cm_[0][0]+cm_[0][1]+cm_[1][0]+cm_[1][1])
            precision_=precision_score(test_cls_label,y_cls_bool) #cm_[1][1]/(cm_[0][1]+cm_[1][1])
            recall_=recall_score(test_cls_label,y_cls_bool) #cm_[1][1]/(cm_[1][0]+cm_[1][1])
            f1_=f1_score(test_cls_label,y_cls_bool)
            auc_=roc_auc_score(test_cls_label,y_cls)

            print('fold: {}, mse: {:.4f}, corr: {:.4f}, vc: {:.4f}\n'.format(model_i,mse_,corr_,vc_))
            print('fold: {}, auc: {:.4f}, f1: {:.4f}, precision: {:.4f} recall: {:.4f}\n'.format(model_i,auc_,f1_,precision_,recall_))
            print('[[{},{}],\n[{},{}]]\n\n\n'.format(cm_[0][0],cm_[0][1],cm_[1][0],cm_[1][1]))

            test_vc_all.append(vc_)
            test_mse_all.append(mse_)
            test_corr_all.append(corr_)
            test_precision_all.append(precision_)
            test_recall_all.append(recall_)
            test_f1_all.append(f1_)
            test_auc_all.append(auc_)
        

    vc_mean=np.mean(test_vc_all)
    mse_mean=np.mean(test_mse_all)
    corr_mean=np.mean(test_corr_all)
    precision_mean=np.mean(test_precision_all)
    recall_mean=np.mean(test_recall_all)
    f1_mean=np.mean(test_f1_all)
    auc_mean=np.mean(test_auc_all)

    print('total, auc: {:.4f}, vc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, mse: {:.4f}, corr: {:.4f}'.format(auc_mean,vc_mean,f1_mean,precision_mean,
                                                                                                                            recall_mean,mse_mean,corr_mean))
