#!/usr/bin/python3

import argparse
import pandas as pd
from jax_unirep import get_reps
from tqdm import tqdm
import gc
import os
import pickle
from funcs import read_parsnip, read_parsnip

def parse_args():
    parser = argparse.ArgumentParser("Format sequences with jax-UniRep")

    parser.add_argument(
        '-f', '--fasta',
        type=str,
        required=True,
        help="File with fasta sequences to be formatted",
        dest='fasta'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Name to store converted sequences in pickle format',
        dest='output_file'
    )

    parser.add_argument(
        '-parsnip',
        action='store_true',
        help='If given, the data is from parsnip and not soluprot',
        dest='parsnip'
    )

    parser.add_argument(
        '-pickled',
        action='store_true',
        help='Load input file as pickle',
        dest='input_pickled'
    )

    return parser.parse_args()
    
def convert(sequence):
    """Format sequences with jax-UniRep"""
    h_avg, _, _ = get_reps(sequence)
    return h_avg

def to_pickle(d, destfile):
    with open(destfile, 'wb') as dest:
        pickle.dump(d, dest)

def load_pickle(sourcefile):
    with open(sourcefile, 'rb') as source:
        d = pickle.load(source)
    return d

if __name__ == "__main__":
    args = parse_args()

    # load sequences from FASTA file
    if args.input_pickled:
        seqs = load_pickle(args.fasta)
    elif args.parsnip:
        seqs = read_parsnip(args.fasta)
    else:
        seqs = read_fasta(args.fasta)

    if os.path.isfile(args.output_file) and os.path.getsize(args.output_file) > 0:
        unirep_seqs = load_pickle(args.output_file)
        for sid in unirep_seqs.keys():
            try:
                seqs.pop(sid)
            except KeyError:
                pass
    else:
        unirep_seqs = {}

    for sid, sequence in tqdm(seqs.items(), desc=args.fasta):
        h_avg = convert(sequence)
        unirep_seqs[sid] = h_avg

        to_pickle(
            d = unirep_seqs, 
            destfile=args.output_file
        )
        
        del h_avg
        gc.collect()