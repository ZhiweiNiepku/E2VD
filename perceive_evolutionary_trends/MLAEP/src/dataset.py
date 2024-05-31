import torch
import pandas as pd
import string
from typing import List, Tuple
import itertools
from Bio import SeqIO
import numpy as np
import os
dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
from sklearn.preprocessing import OneHotEncoder

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class DMS_data(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 msa=False,
                 msa_dir="None",
                 msa_num=64,
                 seq_path=os.path.join(dir,'data/Covid19_RBD_seq.txt')) -> None:
        super().__init__()
        self.data_path=data_path
        self.msa=msa
        self.seq_path=seq_path
        self.msa_dir=msa_dir
        self.msa_num=msa_num
        with open(seq_path, "r") as f:  
            seq = f.read() 
        self.seq=seq
        
        dms_data = pd.read_csv(data_path,index_col=0)
        if dms_data.isnull().values.any() !=False:
            ValueError("There is nan in the data.")
        
        self.dms_data=dms_data
        
        num_data=dms_data.shape[0]
        r=OneHotEncoder(sparse=False).fit_transform(dms_data.iloc[:,2].to_numpy().reshape(num_data,1)).reshape(num_data,1,2)
        for i in range(3,11):
            n=OneHotEncoder(sparse=False).fit_transform(dms_data.iloc[:,i].to_numpy().reshape(num_data,1)).reshape(num_data,1,2)
            r=np.append(r,n,axis=1)
        # r [batch_size,9,2]
        self.class_label=r

    def get_data(self):
        if self.msa==False:
            data=[]
            for index, row in self.dms_data.iterrows():
                data.append(("1",row["seq"]))
            return(data)
        else:
            mutation=[]
            for index, row in self.dms_data.iterrows():
                mutation.append(row["mutation"])
            deletekeys = dict.fromkeys(string.ascii_lowercase)
            deletekeys["."] = None
            deletekeys["*"] = None
            translation = str.maketrans(deletekeys)
            
            def remove_insertions(sequence: str) -> str:
                """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
                return sequence.translate(translation)
            
            def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
                """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
                return [(record.description, remove_insertions(str(record.seq)))
                        for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]
            
            msa_data=[]
            for each in mutation:
                msa_data.append(read_msa(self.msa_dir+'seq_'+each,self.msa_num))
            return(msa_data)
        
class DMS_dataset_pl(torch.utils.data.Dataset):
    def __init__(self, encodings, class_label):

        self.encodings = encodings
        self.class_label = class_label

    def __getitem__(self, idx):
        item = {"input_ids":self.encodings[idx]}
        item["labels"] = (torch.tensor(self.class_label[idx]))
        return item

    def __len__(self):
        return len(self.class_label)