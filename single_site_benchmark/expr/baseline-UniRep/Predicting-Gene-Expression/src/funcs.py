import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, matthews_corrcoef, roc_curve
import sys

# sys.path.append('..')
# from model.ann import ANN

actname2func = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(),
    'tanh': nn.Tanh()
}   

class Logger(object):
    def __init__(self, logfile):
        self.log = open(logfile, "w")

    def write(self, message):
        print(message)
        self.log.write(message + '\n')
    def flush(self):
            pass
    def close(self):
        self.log.close()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def find_optimal_cutoff(y_score, y_true):
    """Find the optimal cutoff value"""

    fpr, tpr, threshold = roc_curve(y_true, y_score)

    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    threshold = list(roc_t['threshold'])[0]
    return threshold

# def _train(net, train_loader, eval_loader, epochs, optimizer='SGD', lr=0.01, momentum=0, l2_reg=0, use_cuda=False, n_patience=10, use_early_stopping=False, verbose=False):
#     """Train a neural network

#     Parameters
#     ----------
#     net : nn.Module
#         The neural network
#     train_loader : [type]
#         iterator for training data
#     eval_loader : [type]
#         iterator for Evalation data
#     epochs : int
#         Number of epochs (iterations) to train the network
#     optimizer : optim.Optimizer
#         Approach for minimizing loss function
#     lr : float
#         Learning rate, by default 0.01
#     momentum : float
#         Momentum factor, by default 0
#     l2_reg : float
#         Weight decay (L2 penalty) for optimizer
#     use_cuda : bool, optional
#         Train network on CPU or GPU, by default False
#     n_patience : int, optional
#         Patience for early stopping: epochs // n_patience, by default 10
#     use_early_stopping : bool, optional
#         If True, use early stopping. By default False
#     verbose : bool, optional
#         Print statements if True, by default False

#     Returns
#     ----------
#     net : nn.Module 
#         Trained network
#     train_loss : list
#         Training loss per epoch
#     eval_loss : 
#         Evalation loss per epoch
#     """

#     # switch to GPU if given
#     if use_cuda:
#         net.cuda()

#     # initializer optimizer
#     if optimizer == 'SGD':
#         optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=l2_reg, momentum=momentum)
#     elif optimizer == 'Adam':
#         optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=l2_reg)
    
#     # loss function
#     criterion = nn.MSELoss()

#     train_loss, eval_loss = [], []

#     # early stopping function
#     patience = epochs // n_patience
#     early_stopping = EarlyStopping(patience=patience, verbose=verbose)

#     if verbose and use_early_stopping:
#         print("Using early stopping with patience:", patience)

#     # start main loop
#     for epoch in range(epochs):

#         # train
#         net.train()

#         running_train_loss = 0
#         for X, y in train_loader:
#             pred = net(X)
#             loss = criterion(pred, y)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             running_train_loss += loss.data
        
#         train_loss.append(running_train_loss / len(train_loader))

#         # Evalation
#         net.eval()
        
#         running_eval_loss = 0
#         for x_Eval, y_Eval in eval_loader:
#             pred = net(x_Eval)
#             loss = criterion(pred, y_Eval)
#             running_eval_loss += loss.data
        
#         eval_loss.append(running_eval_loss / len(eval_loader))

#         if verbose:
#             print("Epoch {}: Train loss: {:.5f} Eval loss: {:.5f}".format(epoch+1, train_loss[epoch], eval_loss[epoch]))

#         if invoke(early_stopping, eval_loss[-1], net, implement=use_early_stopping):
#             net.load_state_dict(torch.load('checkpoint.pt'))
#             break

#     return net, optimizer, train_loss, eval_loss, epoch

# def train_cv(X, X_test, y, y_test, batch_size=16, epochs=100, out_units=[16], optimizer='SGD', p_dropout=0, activation_function='relu', lr=0.01, momentum=0, l2_reg=0, use_cuda=False, n_patience=10, verbose=False, use_early_stopping=False):
#     """Wrapper function to use in nested CV"""

#     if not isinstance(out_units, list):
#         out_units = [out_units]

#     # convert to PyTorch datasets 
#     train_loader, eval_loader = data_loader(X, X_test, y, y_test, batch_size=batch_size)

#     # define neural network
#     in_features = X.shape[1]
#     net = ANN(in_features=in_features, out_units=out_units, n_out=1, p_dropout=p_dropout, activation=actname2func[activation_function])
#     if verbose:
#         print(net)


#     net, optimizer, train_loss, eval_loss, epoch = _train(
#         net=net,
#         train_loader=train_loader,
#         eval_loader=eval_loader,
#         epochs=epochs,
#         optimizer=optimizer,
#         lr=lr,
#         momentum=momentum,
#         l2_reg=l2_reg,
#         n_patience=n_patience,
#         use_early_stopping=use_early_stopping,
#         verbose=verbose,
#         use_cuda=torch.cuda.is_available()
#     )

#     final_train_loss = train_loss[epoch] # [-1] also works instead of indexing by epoch
#     final_eval_loss = eval_loss[epoch]

#     y_pred = net(torch.from_numpy(X_test.astype('float32'))).data.numpy()

#     return final_train_loss, final_eval_loss, net, y_pred

def data_loader(X, X_Eval, y, y_Eval, batch_size=64):
    """Convert numpy arrays into iterable dataloaders with tensors"""

    # convert to tensors
    X = torch.from_numpy(X.astype('float32'))
    X_Eval = torch.from_numpy(X_Eval.astype('float32'))
    y = torch.from_numpy(y.astype('float32')).view(-1, 1)
    y_Eval = torch.from_numpy(y_Eval.astype('float32')).view(-1, 1)

    # define loaders
    train_loader = DataLoader(
        dataset=TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=True
    )
    eval_loader = DataLoader(
        dataset=TensorDataset(X_Eval, y_Eval),
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, eval_loader

def invoke(early_stopping, loss, model, implement=False, verbose=False):
    if implement == False:
        return False
    else:
        early_stopping(loss, model)
        if early_stopping.early_stop:
            if verbose:
                logger.write("Early stopping")
            return True

def read_fasta(fastafile):
    """Parse a file with sequences in FASTA format and store in a dict"""
    with open(fastafile, 'r') as f:
        content = [l.strip() for l in f.readlines()]

    res = {}
    seq, seq_id = '', None
    for line in content:
        if line.startswith('>'):
            
            if len(seq) > 0:
                res[seq_id] = seq
            
            seq_id = line.replace('>', '')
            seq = ''
        else:
            seq += line

    return res

def read_parsnip(datafile):
    with open(datafile, 'r') as f:
        content = [l.strip() for l in f.readlines()]
    res = {}
    for i, line in enumerate(content):
        res[i] = line.split(' ')[0]
    return res