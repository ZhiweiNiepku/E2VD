from .base import WordSubstitute
from ....tags import *
from ....exceptions import WordNotInDictionaryException
from typing import Optional
import json

#protein_dict={"L": ["L", "I", "M", "V", "F", "A", "C", "X", "T", "Y", "R", "Q", "K", "S", "W", "N", "E", "H", "P", "Z", "D", "G", "B"], "A": ["A", "S", "V", "C", "G", "T", "X", "Q", "L", "M", "E", "R", "K", "Z", "P", "I", "N", "Y", "F", "D", "B", "H", "W"], "G": ["G", "N", "A", "S", "D", "B", "X", "R", "Q", "E", "K", "Z", "H", "P", "W", "T", "C", "Y", "V", "F", "M", "I", "L"], "V": ["V", "I", "M", "L", "T", "A", "F", "Y", "C", "X", "E", "S", "Q", "K", "P", "Z", "D", "N", "R", "G", "W", "H", "B"], "S": ["S", "N", "A", "T", "D", "G", "K", "Q", "E", "B", "X", "Z", "P", "H", "C", "R", "M", "L", "F", "I", "V", "Y", "W"], "E": ["E", "Z", "D", "Q", "K", "B", "R", "N", "S", "H", "A", "T", "P", "X", "Y", "V", "G", "M", "F", "W", "L", "I", "C"], "R": ["R", "K", "Q", "N", "H", "E", "Z", "A", "M", "T", "S", "X", "B", "L", "G", "D", "Y", "P", "W", "V", "F", "I", "C"], "T": ["T", "S", "N", "A", "V", "X", "R", "M", "I", "P", "E", "Q", "L", "D", "K", "C", "B", "Z", "F", "H", "G", "W", "Y"], "I": ["I", "V", "L", "M", "F", "C", "A", "Y", "T", "X", "S", "H", "D", "R", "N", "Q", "E", "Z", "W", "B", "K", "P", "G"], "D": ["D", "B", "E", "N", "Z", "S", "Q", "H", "P", "X", "K", "G", "T", "R", "A", "V", "I", "C", "M", "Y", "F", "W", "L"], "P": ["P", "D", "K", "Q", "E", "A", "S", "Z", "T", "H", "G", "R", "N", "M", "B", "X", "V", "L", "C", "I", "Y", "F", "W"], "K": ["K", "R", "Q", "E", "Z", "N", "B", "S", "H", "D", "A", "P", "X", "T", "M", "G", "L", "Y", "V", "I", "C", "F", "W"], "Q": ["Q", "Z", "E", "R", "K", "N", "D", "M", "B", "S", "H", "A", "Y", "T", "P", "X", "V", "L", "W", "G", "C", "F", "I"], "N": ["N", "B", "D", "H", "S", "R", "G", "T", "Q", "E", "K", "Z", "X", "A", "M", "Y", "P", "L", "V", "F", "C", "I", "W"], "F": ["F", "Y", "W", "I", "M", "L", "H", "V", "X", "A", "C", "T", "S", "Q", "E", "R", "N", "G", "K", "D", "Z", "B", "P"], "Y": ["Y", "F", "W", "H", "M", "I", "Q", "L", "V", "X", "E", "A", "N", "R", "K", "C", "S", "T", "Z", "G", "D", "P", "B"], "M": ["M", "L", "I", "V", "Q", "F", "R", "A", "C", "K", "Y", "T", "W", "Z", "S", "X", "N", "E", "H", "P", "D", "G", "B"], "H": ["H", "Y", "N", "R", "Q", "E", "Z", "B", "D", "S", "X", "K", "F", "G", "A", "P", "W", "T", "M", "C", "I", "L", "V"], "W": ["W", "Y", "F", "M", "Q", "L", "H", "T", "G", "C", "X", "R", "A", "E", "I", "K", "S", "Z", "V", "D", "P", "N", "B"], "C": ["C", "A", "S", "I", "V", "L", "T", "M", "X", "Y", "F", "W", "D", "N", "R", "Z", "B", "Q", "H", "P", "K", "G", "E"], "X": ["T", "S", "A", "L", "H", "D", "X", "K", "G", "N", "F", "B", "Z", "V", "R", "M", "I", "E", "Y", "Q", "C", "W", "P"], "B": ["D", "B", "N", "E", "Z", "S", "K", "H", "Q", "G", "T", "R", "X", "P", "A", "Y", "C", "I", "V", "M", "F", "W", "L"], "Z": ["Z", "E", "Q", "K", "D", "B", "H", "S", "N", "R", "P", "A", "M", "T", "X", "Y", "G", "V", "C", "W", "I", "L", "F"]}
protein_dict={"L": ["I", "M", "V", "F", "A", "C", "T", "Y", "R", "Q", "K", "S", "W", "N", "E", "H", "P", "D", "G"], "A": ["S", "V", "C", "G", "T", "Q", "L", "M", "E", "R", "K", "P", "I", "N", "Y", "F", "D", "H", "W"], "G": ["N", "A", "S", "D", "R", "Q", "E", "K", "H", "P", "W", "T", "C", "Y", "V", "F", "M", "I", "L"], "V": ["I", "M", "L", "T", "A", "F", "Y", "C", "E", "S", "Q", "K", "P", "D", "N", "R", "G", "W", "H"], "S": ["N", "A", "T", "D", "G", "K", "Q", "E", "P", "H", "C", "R", "M", "L", "F", "I", "V", "Y", "W"], "E": ["D", "Q", "K", "R", "N", "S", "H", "A", "T", "P", "Y", "V", "G", "M", "F", "W", "L", "I", "C"], "R": ["K", "Q", "N", "H", "E", "A", "M", "T", "S", "L", "G", "D", "Y", "P", "W", "V", "F", "I", "C"], "T": ["S", "N", "A", "V", "R", "M", "I", "P", "E", "Q", "L", "D", "K", "C", "F", "H", "G", "W", "Y"], "I": ["V", "L", "M", "F", "C", "A", "Y", "T", "S", "H", "D", "R", "N", "Q", "E", "W", "K", "P", "G"], "D": ["E", "N", "S", "Q", "H", "P", "K", "G", "T", "R", "A", "V", "I", "C", "M", "Y", "F", "W", "L"], "P": ["D", "K", "Q", "E", "A", "S", "T", "H", "G", "R", "N", "M", "V", "L", "C", "I", "Y", "F", "W"], "K": ["R", "Q", "E", "N", "S", "H", "D", "A", "P", "T", "M", "G", "L", "Y", "V", "I", "C", "F", "W"], "Q": ["E", "R", "K", "N", "D", "M", "S", "H", "A", "Y", "T", "P", "V", "L", "W", "G", "C", "F", "I"], "N": ["D", "H", "S", "R", "G", "T", "Q", "E", "K", "A", "M", "Y", "P", "L", "V", "F", "C", "I", "W"], "F": ["Y", "W", "I", "M", "L", "H", "V", "A", "C", "T", "S", "Q", "E", "R", "N", "G", "K", "D", "P"], "Y": ["F", "W", "H", "M", "I", "Q", "L", "V", "E", "A", "N", "R", "K", "C", "S", "T", "G", "D", "P"], "M": ["L", "I", "V", "Q", "F", "R", "A", "C", "K", "Y", "T", "W", "S", "N", "E", "H", "P", "D", "G"], "H": ["Y", "N", "R", "Q", "E", "D", "S", "K", "F", "G", "A", "P", "W", "T", "M", "C", "I", "L", "V"], "W": ["Y", "F", "M", "Q", "L", "H", "T", "G", "C", "R", "A", "E", "I", "K", "S", "V", "D", "P", "N"], "C": ["A", "S", "I", "V", "L", "T", "M", "Y", "F", "W", "D", "N", "R", "Q", "H", "P", "K", "G", "E"]}

class ProteinBlosum62Substitute(WordSubstitute):
    TAGS = { TAG_Protein }

    def __init__(self, k : Optional[int] = None):
        """
        Protein amino acid substitute based on Blosum62.
        Args:
            k: Top-k results to return. If k is `None`, all results will be returned.
        
        :Language: protein
        
        """

        self.k = k
        #self.protein_dict = json.load(open('/Users/janie/Desktop/pre-training/AttackProtein/attack_protein/dict_protein.json',"r"))
        self.protein_dict = protein_dict

    def substitute(self, word, pos_tag):
        word=word.upper()
        if word not in self.protein_dict:
            raise WordNotInDictionaryException()
        sym_words = self.protein_dict[word]
        
        ret = []
        for sym_word in sym_words:
            ret.append((sym_word, 1))

        if self.k is not None:
            ret = ret[:self.k]
        return ret