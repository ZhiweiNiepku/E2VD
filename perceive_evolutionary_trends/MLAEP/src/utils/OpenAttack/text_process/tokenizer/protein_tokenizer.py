from .base import Tokenizer
from ...tags import *
#from ...data import aas_pos

aas_pos={'L':("L","a"),'A':('A','a'),'G':('G','a'),'V':('V','a'),'S':('S','a'),'E':('E','a'),'R':('R','a'),'T':('T','a'),'I':('I','a'),'D':('D','a'),'P':('P','a'),'K':('K','a'),
'Q':('Q','a'),'N':('N','a'),'F':('F','a'),'Y':('Y','a'),'M':('M','a'),'H':('H','a'),'W':('W','a'),'C':('C','a'),'X':('X','a'),'B':('B','a'),'Z':('Z','a')}

def tokenize_pos(seq):
    ret=[]
    for aa in seq.upper():
        ret.append(aas_pos[aa])
    return(ret)
class ProteinTokenizer(Tokenizer):
    """
    Tokenizer based on single amino acid
    :Language: protein
    """

    TAGS = { TAG_Protein }

    def __init__(self) -> None:
        self.__tokenize = tokenize_pos
    
    def do_tokenize(self, x, pos_tagging):
        if pos_tagging:
            ret = self.__tokenize(x)
            return ret
        else:
            return list(x)
    
    def do_detokenize(self, x):
        return "".join(x)