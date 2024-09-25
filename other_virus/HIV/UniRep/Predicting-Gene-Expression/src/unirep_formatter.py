import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
import subprocess
import argparse
import sys
sys.path.append('UniRep/')

def compute_UniRep(sequences, y, destination_name, model_1900=True, download=False):
    """
    Computes the UniRep representations of the input sequences. 
    It requires that the GitHub repo is in path and this function is adatoped from their tutorial notebook.
    GitHub repository: https://github.com/churchlab/UniRep/
    
    Arguments:
        sequences: list of sequences in string format
        y: label column
        destination_name: file name to pickle the formatted the sequences to
        model_1900: use full representation (dim=1900) or the smaller one with dim=64
        download: if True, download the model weights from aws s3 bucket
    """
    
    if model_1900:
        if download:
            subprocess.run(['aws', 's3' ,'sync', '--no-sign-request', '--quiet', 's3://unirep-public/1900_weights/', '1900_weights/'])
    
        # Import the mLSTM babbler model
        from unirep import babbler1900 as babbler

        # Where model weights are stored.
        MODEL_WEIGHT_PATH = "./1900_weights"
    else:
         # Sync relevant weight files
        if download:
            subprocess.run(['aws', 's3' ,'sync', '--no-sign-request', '--quiet', 's3://unirep-public/64_weights/', '64_weights/'])

        # Import the mLSTM babbler model
        from unirep import babbler64 as babbler

        # Where model weights are stored.
        MODEL_WEIGHT_PATH = "./64_weights"
    
    batch_size = 12
    b = babbler(batch_size=batch_size, model_path=MODEL_WEIGHT_PATH)
    
    UniRep_sequences = []
    fusion_sequences = []
    N = len(sequences)
    for i, seq in enumerate(sequences):
        print("Formatting sequence {}/{}".format(i+1, N), end='\r')
        
        avg_hidden, final_hidden, final_cell = b.get_rep(seq)
        
        # save average hidden state as this is the UniRep representation
        UniRep_sequences.append([avg_hidden, y[i]])
        
        # concate to get UniRep-fusion representation
        fusion = np.stack((avg_hidden, final_hidden, final_cell), axis=1)
        fusion_sequences.append([fusion, y[i]])
    
    # create two file names for the two different representations
    split_name = destination_name.split('.')
    fusion_name = "{}_fusion.pkl".format("".join(split_name[:-1]))
    unirep_name = "{}_UniRep.pkl".format("".join(split_name[:-1]))
    
    # dump the lists
    with open(unirep_name, 'wb') as destination:
        pickle.dump(UniRep_sequences, destination)
    with open(fusion_name, 'wb') as destination:
        pickle.dump(fusion_sequences, destination)

def parse_args():
    parser = argparse.ArgumentParser(description = "use UniRep to format sequences")

    parser.add_argument('--data', type=str, required=True, help="csv file with data")
    parser.add_argument('--column', type=str, required=True, help="column with sequences")
    parser.add_argument('--destination', type=str, required=True, help="file to put formatted sequences")
    parser.add_argument('--label', type=str, required=True, help="column with labels")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    #data = pd.read_csv(args.data, index_col=0)
    data = pd.read_csv(args.data)

    if args.column in data.columns:
        sequences = data[args.column].copy()
        y = data[args.label].copy()
        compute_UniRep(sequences, y, destination_name=args.destination)
    else:
        raise Exception("Column {} not in data".format(args.column))