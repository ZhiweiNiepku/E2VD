c2b = {
    '0 very high positive corrrelation': [0.9, 1],
    '1 high positive corrrelation': [.7, .9],
    '2 moderate positive correlation': [.5, .7],
    '3 low positive correlation': [.3, .5],
    '4 negligible correlation': [-.3, .3],
    '5 low negative correlation': [-.5, -.3],
    '6 moderate negative correlation':[-.7, -.5],
    '7 high negative correlation': [-.9, -.7],
    '8 very high negative correlation': [-1, -.9],
}

c2b_abs = {
    '0 very high correlation': [0,9],
    '1 high correlation': [.7, .9],
    '2 moderate correlation': [.5, .7],
    '3 low correlation': [.3, .5],
    '4 negligible correlation': [0, .3]
}

c2l = {
    '0 very high positive corrrelation': 'very high positive corrrelation [0.9, 1]',
    '1 high positive corrrelation': 'high positive corrrelation [0.7, 0.9]',
    '2 moderate positive correlation': 'moderate positive correlation [0.5, 0.7]',
    '3 low positive correlation': 'low positive correlation [0.3, 0.5]',
    '4 negligible correlation': 'negligible correlation [-0.3, 0.3]',
    '5 low negative correlation': 'low negative correlation [-0.3, -0.5]',
    '6 moderate negative correlation': 'moderate negative correlation [-0.5, -0.7]',
    '7 high negative correlation': 'high negative correlation [-0.7, -0.9]' ,
    '8 very high negative correlation': 'very high negative correlation [-0.9, -1]',
}

c2l_abs = {
    '0 very high correlation': 'very high correlation',
    '1 high correlation': 'high correlation',
    '2 moderate correlation': 'moderate correlation',
    '3 low correlation': 'low correlation',
    '4 negligible correlation': 'negligible correlation'
}

# spectral colormap used to get the colors:
# from matplotlib import cm
# norm = matplotlib.colors.Normalize(vmin=0, vmax=9)
# rgba_color = [cm.Spectral(norm(i)) for i in range(9)]
c2c = {
    '0 very high positive corrrelation': (0.6196078431372549, 0.00392156862745098, 0.25882352941176473, 1.0),
    '1 high positive corrrelation': (0.8472126105344099, 0.2612072279892349, 0.30519031141868513, 1.0),
    '2 moderate positive correlation': (0.9637831603229527, 0.47743175701653207, 0.28581314878892733, 1.0),
    '3 low positive correlation': (0.9934640522875817, 0.7477124183006535, 0.4352941176470587, 1.0),
    '4 negligible correlation': (0.9977700884275279, 0.930872741253364, 0.6330642060745867, 1.0),
    '5 low negative correlation': (0.944252210688197, 0.9777008842752788, 0.6620530565167244, 1.0),
    '6 moderate negative correlation': (0.7477124183006538, 0.8980392156862746, 0.6274509803921569, 1.0),
    '7 high negative correlation': (0.4530565167243369, 0.7815455594002307, 0.6462898885044214, 1.0),
    '8 very high negative correlation':  (0.21607074202229912, 0.5556324490580546, 0.7319492502883507, 1.0),
}

c2c_abs = {
    '0 very high correlation': (0.6196078431372549, 0.00392156862745098, 0.25882352941176473, 1.0),
    '1 high correlation': (0.9568627450980393, 0.42745098039215684, 0.2627450980392157, 1.0),
    '2 moderate correlation': (0.996078431372549, 0.8784313725490196, 0.5450980392156862, 1.0),
    '3 low correlation': (0.9019607843137256, 0.9607843137254902, 0.5960784313725491, 1.0),
    '4 negligible correlation': (0.4, 0.7607843137254902, 0.6470588235294118, 1.0)
}

aa2codons = {
    'A': ['GCU', 'GCC', 'GCA','GCG'],
    'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'N': ['AAU', 'AAC'],
    'D': ['GAU', 'GAC'],
    'C': ['UGU', 'UGC'],
    'Q': ['CAA', 'CAG'],
    'E': ['cid', 'GAA', 'GAG'],
    'G': ['GGU', 'GGC', 'GGA', 'GGG'],
    'H': ['CAU', 'CAC'],
    'I': ['AUU', 'AUC', 'AUA'],
    'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'],
    'K': ['AAA', 'AAG'],
    'M': ['AUG'],
    'F': ['UUU UUC'],
    'P': ['CCU', 'CCC', 'CCA', 'CCG'],
    'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'],
    'T': ['ACU', 'ACC', 'ACA', 'ACG'],
    'W': ['UGG'],
    'Y': ['UAU', 'UAC'],
    'V': ['GUU', 'GUC', 'GUA', 'GUG'],
    'B': ['AAU', 'AAC', 'GAU', 'GAC'],
    'Z': ['AAU', 'AAC', 'GAU', 'GAC'],
    'J': ['AUU', 'AUC', 'AUA', 'UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'],
}

valid_codons = ['codon_' + y for x in aa2codons.values() for y in x]