from collections import defaultdict
import pandas as pd
import os
import argparse
import pickle
import re
import numpy as np
import glob
import torch
import sys
sys.path.append(
    os.path.join(os.path.dirname(__file__), '..')
)

from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve, auc, accuracy_score
from funcs import find_optimal_cutoff
from model.baselines import CreateBaselines
from model.ann import ANN

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d', '--datadir',
        type=str,
        required=True,
        help='Directory with data files',
        dest='datadir'
    )

    parser.add_argument(
        '-m', '--modeldir',
        type=str,
        required=True,
        help='Directory with pickled models',
        dest='modeldir'
    )

    parser.add_argument(
        '-o', '--outcsv',
        type=str,
        required=True,
        help='Destination file containing results',
        dest='outcsv'
    )

    return parser.parse_args()


def load_nz_data(datadir, verbose=True):   

    # load
    train = pd.read_csv(os.path.join(datadir, 'Copy of train3.csv'))
    train['set'] = 'train'
    
    valid = pd.read_csv(os.path.join(datadir, 'Copy of valid3.csv'))
    valid['set'] = 'valid'

    test = pd.read_csv(os.path.join(datadir,  'Copy of test3.csv'))
    test['set'] = 'test'
    

    # load unirep formatted sequences
    with open(os.path.join(datadir, 'expr_unirep_may11.pkl'), 'rb') as source:
        train_unirep = pickle.load(source)
        valid_unirep = pickle.load(source)
        test_unirep = pickle.load(source)

    # add to frames
    unirep_cols = np.arange(0, 1900)
    train = train.merge(train_unirep[['protein', 'unirep']], on='protein')
    train[unirep_cols] = train.unirep.apply(pd.Series)

    valid = valid.merge(valid_unirep[['protein', 'unirep']], on='protein')
    valid[unirep_cols] = valid.unirep.apply(pd.Series)

    test = test.merge(test_unirep[['protein', 'unirep']], on='protein')
    test[unirep_cols] = test.unirep.apply(pd.Series)

    # concat
    data = pd.concat([train, valid, test], axis=0)
    if verbose:
        print("Data shapes:")
        print(" train:", train.shape)
        print(" valid:", valid.shape)
        print(" test:", test.shape)
        print("Total:", data.shape)
        print()
    
    # load preds
    protein_sol = pd.read_csv(os.path.join(datadir,  'nz_protein-sol_preds.csv') )
    sodope = pd.read_csv(os.path.join(datadir,  'sodope_swi.tsv'), sep='\t', usecols=[0, 1, 3])
    skade = pd.read_csv(os.path.join(datadir,  'expr_all.fasta.skadePredictions.tsv'), sep='\t')
    #unirep_rf = pd.read_csv(os.path.join(datadir,  'nz_unirep-rf_preds.csv'))
    #unirep_rf.rename(columns={'y_score': 'unirep_rf_score'}, inplace=True)
    
    if verbose:
        print("nz_protein-sol_preds.csv shape:", protein_sol.shape)
        print("sodope_swi.tsv shape:", sodope.shape)
        print("expr_all.fasta.skadePredictions.tsv shape:", skade.shape)
        print()
    
    # Protein-Sol
    data = data.merge(
        protein_sol[['ID', 'scaled-sol']],
        left_on='protein', right_on='ID',
        how='left'
    )
    if verbose:
        print("Merged protein_sol, total shape:", data.shape)
    
    # SoDoPe
    data =data.merge(
        sodope,
        left_on=['protein'], right_on='Accession',
        how='left'
    )
    if verbose:
        print("Merged sodope, total shape:", data.shape)
    
    # SKADE
    data = data.merge(
        skade,
        left_on='protein', right_on='Protein_ID',
        how='left'
    )
    if verbose:
        print("Merged SKADE, total shape:", data.shape)
        
    #.merge(
    #    unirep_rf,
    # on='protein'
    # )
    
    return data


def roc_auc(y_true, y_score):

    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score)
    return auc(fpr, tpr)


def evaluator(X_train, y_train, X_valid, y_valid, X_test, y_test, predictor):
    
    # calculate probabilities
    try:
        y_train_score = predictor.predict_proba(X_train)[:, 1]
        y_valid_score = predictor.predict_proba(X_valid)[:, 1]
        y_test_score = predictor.predict_proba(X_test)[:, 1]
    except:
        y_train_score = predictor(torch.from_numpy(X_train.values).float()).detach().numpy()
        y_valid_score = predictor(torch.from_numpy(X_valid.values).float()).detach().numpy()
        y_test_score = predictor(torch.from_numpy(X_test.values).float()).detach().numpy()

    # calculate tau on train set
    tau = find_optimal_cutoff(y_true=y_valid, y_score=y_valid_score)

    # get y_pred for the sets
    y_train_pred = np.asarray(y_train_score > tau, dtype='int')
    y_valid_pred = np.asarray(y_valid_score > tau, dtype='int')
    y_test_pred = np.asarray(y_test_score > tau, dtype='int')

    # calculate AUCs
    train_auc = roc_auc(y_true=y_train, y_score=y_train_score)
    valid_auc = roc_auc(y_true=y_valid, y_score=y_valid_score)
    test_auc = roc_auc(y_true=y_test, y_score=y_test_score)

    # calculate MCCs
    train_mcc = matthews_corrcoef(y_true=y_train, y_pred=y_train_pred)
    valid_mcc = matthews_corrcoef(y_true=y_valid, y_pred=y_valid_pred)
    test_mcc = matthews_corrcoef(y_true=y_test, y_pred=y_test_pred)

    return tau, train_auc, valid_auc, test_auc, train_mcc, valid_mcc, test_mcc

def load_sklearn(model_file, testdata, res, y_true='is_expression_successful'):

    p = re.compile(r'(RF|SVM|LR)')
    model2name = {'RF': 'baseline_rf', 'SVM': 'baseline_linear', 'LR': 'baseline_log'}

    # open pickled models
    with open(model_file, 'rb') as source:
        model = pickle.load(source)

    taus = []
    train_aucs, valid_aucs, test_aucs = [], [], []
    train_mccs, valid_mccs, test_mccs = [], [], []

    model_type = p.findall(model_file)[0]

    # loop through them
    for k, v in model.items():
        if k in ['input_cols', 'params']:
            continue

        # get predictor
        predictor = getattr(v['model'], model2name[model_type])

        # define X_test and y_test
        train_columns = v['model'].X_train.columns.tolist()
        try: 
            X_test = testdata[train_columns]
        except:
            continue
        y_test = testdata[y_true]

        # calculate
        tau, train_auc, valid_auc, test_auc, train_mcc, valid_mcc, test_mcc = evaluator(
            X_train = v['model'].X_train,
            y_train = v['model'].y_train,
            X_valid = v['model'].X_test,
            y_valid = v['model'].y_test,
            X_test = X_test,
            y_test = y_test,
            predictor=predictor
        )

        taus.append(tau)
        train_aucs.append(train_auc)
        valid_aucs.append(valid_auc)
        test_aucs.append(test_auc)
        train_mccs.append(train_mcc)
        valid_mccs.append(valid_mcc)
        test_mccs.append(test_mcc)

    res[os.path.basename(model_file)]['Train AUC'] = {
            'mean': np.mean(train_aucs),
            'std': np.std(train_aucs)
    }

    res[os.path.basename(model_file)]['Valid AUC'] = {
            'mean': np.mean(valid_aucs),
            'std': np.std(valid_aucs),
        }
    
    res[os.path.basename(model_file)]['Test AUC'] = {
            'mean': np.mean(test_aucs),
            'std': np.std(test_aucs),
        }
        
    res[os.path.basename(model_file)]['Train MCC'] = {
        'mean': np.mean(train_mccs),
        'std': np.mean(train_mccs)
    }

    res[os.path.basename(model_file)]['Valid MCC'] = {
        'mean': np.mean(valid_mccs),
        'std': np.mean(valid_mccs)
    }

    res[os.path.basename(model_file)]['Test MCC'] = {
        'mean': np.mean(test_mccs),
        'std': np.std(test_mccs),
    }
    res[os.path.basename(model_file)][r'\tau'] = {
        'mean': np.mean(taus),
        'std': np.std(taus)
    }

    return res


def load_ann(modeldir, data, res, y_true='is_expression_successful'):

    train = data.query("set == 'train'")
    valid = data.query("set == 'valid'")
    test = data.query("set == 'test'")
    
    # load aafreq ann
    with open(os.path.join(modeldir, 'AAFreq_ANN_out_units_64_p_dropout_0.2_lr_0.01.pkl'), 'rb') as source:
        best_aafreq_ann = pickle.load(source)
    
    # define 
    valid_aas = 'ARNDCQEGHILKMFPSTWYV'
    aa_cols = ['aa_' + aa for aa in valid_aas]
    aafreq = defaultdict(list)

    # evaluate
    for i in range(10):
        net = ANN(
            in_features=test[aa_cols].shape[1], 
            n_out=1,
            out_units=[best_aafreq_ann['params']['out_units']],
            p_dropout=best_aafreq_ann['params']['p_dropout']
        )
        net.load_state_dict(best_aafreq_ann[i]['model'])
        net.eval()

        tau, train_auc, valid_auc, test_auc, train_mcc, valid_mcc, test_mcc = evaluator(
            X_train=train[aa_cols], y_train=train[y_true],
            X_valid=valid[aa_cols], y_valid=valid[y_true],
            X_test=test[aa_cols], y_test=test[y_true],
            predictor=net
        )

        aafreq['tau'].append(tau)
        aafreq['train_auc'].append(train_auc)
        aafreq['valid_auc'].append(valid_auc)
        aafreq['test_auc'].append(test_auc)
        aafreq['train_mcc'].append(train_mcc)
        aafreq['valid_mcc'].append(valid_mcc)
        aafreq['test_mcc'].append(test_mcc)

    #load unirep ann
    with open(os.path.join(modeldir, 'UniRep_ANN_out_units_64_p_dropout_0.1_lr_0.001.pkl'), 'rb') as source:
        best_unirep_ann = pickle.load(source)
    
    # define
    unirep_cols = np.arange(0, 1900)
    unirep=defaultdict(list)
    
    # evaluate
    for i in range(10):
        net = ANN(
            in_features=test[unirep_cols].shape[1],
            n_out=1,
            out_units=[best_unirep_ann['params']['out_units']],
            p_dropout=best_unirep_ann['params']['p_dropout']
        )
        net.load_state_dict(best_unirep_ann[i]['model'])
        net.eval()

        tau, train_auc, valid_auc, test_auc, train_mcc, valid_mcc, test_mcc = evaluator(
            X_train=train[unirep_cols], y_train=train[y_true],
            X_valid=valid[unirep_cols], y_valid=valid[y_true],
            X_test=test[unirep_cols], y_test=test[y_true],
            predictor=net
        )

        unirep[r'\tau'].append(tau)
        unirep['train_auc'].append(train_auc)
        unirep['valid_auc'].append(valid_auc)
        unirep['test_auc'].append(test_auc)
        unirep['train_mcc'].append(train_mcc)
        unirep['valid_mcc'].append(valid_mcc)
        unirep['test_mcc'].append(test_mcc)
    
    
    # get mean and std
    for k in unirep.keys():
        ks=k.split('_')
        if len(ks) == 2:
            kstr=" ".join([ks[0].title(), ks[1].upper()])
        else:
            kstr = k

        res['AAFreqANN'][kstr] = {
            'mean': np.mean(aafreq[k]),
            'std': np.std(aafreq[k])
        }

        res['UniRepANN'][kstr] = {
            'mean': np.mean(unirep[k]),
            'std': np.std(unirep[k])
        }
    
    return res

def load_models(modeldir, data, outcsv=None, y='is_expression_successful'):

    res = defaultdict(dict)

    testdata = data.query("set == 'test'")

    # load best anns
    res = load_ann(
        modeldir=modeldir,
        data=data,
        res=res,
        y_true=y
    )

    pickled_models = glob.glob(os.path.join(modeldir, '*.pkl'))
    for pm in pickled_models:
        if 'Enzyme' in pm or 'Top' in pm or 'ANN' in pm:
            continue
        res = load_sklearn(
                model_file=pm,
                testdata=testdata,
                y_true=y,
                res = res
            )
        

    res_dfs, res_models = [], []
    for k, v in res.items():
        res_dfs.append(pd.DataFrame.from_dict(v))
        res_models.append(k)
    res_df=pd.concat(res_dfs, keys=res_models, axis=0)
    
    if outcsv is not None:
        res_df.to_csv(outcsv)

    return res_df

def evaluate_predictors(data, score, true, method, setcol='set'):
    
    data = data.copy()
    data.dropna(subset=[score], inplace=True)
    data[score] = data[score].astype('float')
        
    res = dict()  

    # Get tau from train
    tau = find_optimal_cutoff(
        y_true = data.loc[data[setcol] == 'valid', true],
        y_score = data.loc[data[setcol] == 'valid', score]
    )
    res[r'\tau'] = tau

    for g, gdata in data.groupby(setcol):
        y_true = gdata[true]
        y_score = gdata[score]
    
        # calculate auc
        fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score)
        res_auc = auc(fpr, tpr)
        
        # calculate mcc
        y_pred = np.asarray(y_score > tau, dtype='int')
        res_mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
        
        res[g.title() + ' AUC'] = res_auc
        res[g.title() + ' MCC'] = res_mcc
        
    res = pd.DataFrame.from_dict(res, orient='index').T
    res.index = pd.MultiIndex.from_tuples([(method, 'value')])
    
    return res

if __name__ == "__main__":
    args = parse_args()
    
    nz_data = load_nz_data(args.datadir)
    y='is_expression_successful'

    res = load_models(
        modeldir=args.modeldir, 
        data=nz_data,
        y=y
    )

    # evaluate other methods than what we have trained
    other_predictors = ['SoDoPe', 'SKADE', 'Protein-Sol']
    other_columns = ['SWI', 'SKADE_PREDICTION', 'scaled-sol']
    for op, oc in zip(other_predictors, other_columns):
        res_other = evaluate_predictors(
            data=nz_data,
            score=oc,
            true=y,
            method=op
        )
        res = pd.concat([res, res_other], axis = 0)
    
    res.to_csv(args.outcsv)

    # print
    print(res.dropna(how='all').round(2))