import numpy as np
import pandas as pd
import itertools
import random
import torch
from sklearn.metrics import roc_auc_score, matthews_corrcoef, roc_curve
from collections import defaultdict
import sys
import argparse

sys.path.append('../')
from train import _load, save_pickle, load_pickle, _load_freq
from model.ann import ANN
from funcs import train_cv, Logger, find_optimal_cutoff

def parse_args():
    parser = argparse.ArgumentParser(description = "Train ANN classifiers")

    parser.add_argument(
        '-s', '--source',
        type=str,
        required=True,
        help='Pickled UniRep formatted sequences',
        dest='unirep_source'
    )

    return parser.parse_args()

def train_seed(n, params, X_train, y_train, X_valid, y_valid, out_file, logger, epochs=200):
    """Seed training an ANN"""
    logger.write(str(params))
    results = {}
    for i in range(n):
        random.seed()
        
        _, _, net, _ = train_cv(
            X = X_train.values, 
            X_test = X_valid.values, 
            y = y_train.values, 
            y_test= y_valid.values, 
            epochs=epochs, 
            use_early_stopping=True,
            optimizer='Adam',
            **params)
        
        net.eval()
        # evaluate train performance
        y_pred = net(torch.from_numpy(X_train.values).float())
        train_auc = roc_auc_score(y_train.values, y_pred.detach().numpy())
        
        # evaluate valid performance
        y_pred = net(torch.from_numpy(X_valid.values).float())
        valid_auc = roc_auc_score(y_valid.values, y_pred.detach().numpy())
     
        results[i] = {
            'model': net.state_dict(),
            'train_auc': train_auc,
            'valid_auc': valid_auc
        }
        
    try:
        train_aucs = np.array([v['train_auc'] for v in results.values()])
        valid_aucs = np.array([v['valid_auc'] for v in results.values()])

        train_avg_auc, train_std_auc = np.mean(train_aucs), np.std(train_aucs)
        valid_avg_auc, valid_std_auc = np.mean(valid_aucs), np.std(valid_aucs)

        logger.write("Train AUC: {:.3f} (+/- {:.3f})".format(train_avg_auc, train_std_auc))
        logger.write("Valid AUC: {:.3f} (+/- {:.3f})".format(valid_avg_auc, valid_std_auc))
    except:
        pass
    
    results['params'] = params
    save_pickle(results, out_file)


if __name__ == "__main__":

    args = parse_args()
    
    param_grids = {
        'out_units': [8, 16, 32, 64],
        'p_dropout': [0, 0.1, 0.14, 0.2],
        'lr': [0.01, 0.001, 0.0001]
    }
    k, v = zip(*param_grids.items())
    params = [dict(zip(k,v)) for v in itertools.product(*v)]
    
    #######################################################
    # UniRep ANNs
    X_train, X_valid, y_train, y_valid = _load(args.source)
    unit_cols = np.arange(0, 1900, 1)

    X_train = X_train[unit_cols]
    X_valid = X_valid[unit_cols]

    logger = Logger('unirep_ann.log')
    for param in params:
        
        param_str = "_".join([
            f"{k}_{v}" for k, v in param.items()]
        )
        out_file = f'data/seed_training/UniRep_ANN_{param_str}.pkl'
        
        train_seed(
            n=10,
            X_train = X_train,
            y_train = y_train,
            X_valid = X_valid,
            y_valid = y_valid,
            out_file=out_file,
            logger=logger,
            params=param
        )
    logger.close()

    #######################################################
    ## AAFreq ANNs
    # load data
    train = pd.read_csv('data/Copy of train3.csv', index_col=0)
    valid = pd.read_csv('data/Copy of valid3.csv', index_col=0)
    test = pd.read_csv('data/Copy of test3.csv', index_col=0)

    y_col = 'is_expression_successful'

    # get cols
    valid_aas = 'ARNDCQEGHILKMFPSTWYV'
    cols = ['aa_' + aa for aa in valid_aas]

    X_train = train[cols].copy()
    X_valid = valid[cols].copy()

    y_train, y_valid = train[y_col], valid[y_col]
    X_test, y_test = test[cols], test[y_col]
    logger = Logger('aafreq_ann.log')
    for param in params:
        
        param_str = "_".join([f"{k}_{v}" for k, v in param.items()])
        out_file = f'data/seed_training/AAFreq_ANN_{param_str}.pkl'
        
        train_seed(
            n=10,
            X_train = X_train,
            y_train = y_train,
            X_valid = X_valid,
            y_valid = y_valid,
            out_file=out_file,
            logger=logger,
            params=param
        )
    logger.close()
    
    #######################################################
    # Evaluate on test set
    best_unirep_ann = load_pickle('data/seed_training/UniRep_ANN_out_units_64_p_dropout_0.1_lr_0.001.pkl')
    test_aucs, test_mccs, cutoffs = [], [], []
    out_units = best_unirep_ann['params']['out_units']
    p_dropout = best_unirep_ann['params']['p_dropout']
    for i in range(10):
        net = ANN(in_features=X_test.shape[1], n_out=1, out_units=[out_units], p_dropout=p_dropout)
        net.load_state_dict(best_unirep_ann[i]['model'])
        net.eval()
        
        y_score = net(X_test).detach().numpy()
        test_auc = roc_auc_score(y_test.values, y_score)
        
        test_aucs.append(test_auc)
        
        threshold = find_optimal_cutoff(y_score, y_test.values)
        cutoffs.append(threshold)
        y_pred = (y_score > threshold).astype('int')
        test_mccs.append(
            matthews_corrcoef(y_test, y_pred)
        )

    print("File:", best_unirep_ann)
    print("Test AUC: {:.3f} (+/- {:.3f})".format(
        np.mean(test_aucs), 
        np.std(test_aucs))
    )
    print("Test MCC: {:.3f} (+/- {:.3f})".format(
        np.mean(test_mccs), 
        np.std(test_mccs))
    )
    print(f"Cutoff: {np.mean(cutoffs):.2f}")
    
    best_aafreq_ann = load_pickle('data/seed_training/AAFreq_ANN_out_units_64_p_dropout_0.2_lr_0.01.pkl')
    test_aucs, test_mccs, cutoffs = [], [], []
    out_units = best_aafreq_ann['params']['out_units']
    p_dropout = best_aafreq_ann['params']['p_dropout']
    for i in range(10):
        net = ANN(in_features=X_test.shape[1], n_out=1, out_units=[out_units], p_dropout=p_dropout)
        net.load_state_dict(best_aafreq_ann[i]['model'])
        net.eval()
        y_score = net(torch.from_numpy(X_test.values).float()).detach().numpy()
        test_auc = roc_auc_score(y_test.values, y_score)
        test_aucs.append(test_auc)
        threshold = find_optimal_cutoff(y_score, y_test.values)
        cutoffs.append(threshold)
        y_pred = (y_score > threshold).astype('int')
        test_mccs.append(
            matthews_corrcoef(y_test, y_pred)
        )
    print("File:", best_aafreq_ann)
    print("Test AUC: {:.3f} (+/- {:.3f})".format(np.mean(test_aucs), np.std(test_aucs)))
    print("Test MCC: {:.3f} (+/- {:.3f})".format(np.mean(test_mccs), np.std(test_mccs)))
    print(f"Cutoff: {np.mean(cutoffs):.2f}")
