from .protein import PROTEIN_FILTER_WORDS

def get_default_filter_words(lang):
    from ...tags import TAG_Protein
    if lang == TAG_Protein:
        return PROTEIN_FILTER_WORDS