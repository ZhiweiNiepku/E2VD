
from .protein_blosum62 import ProteinBlosum62Substitute
from .base import WordSubstitute

def get_default_substitute(lang):
    from ....tags import TAG_Protein
    if lang == TAG_Protein:
        return ProteinBlosum62Substitute()