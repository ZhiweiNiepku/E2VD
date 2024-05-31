import os
import Bio.SeqIO
#import dmslogo
#import dmslogo.colorschemes
#from IPython.display import display, HTML
#import matplotlib.cm
#import matplotlib.colors
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
import pandas as pd
#import yaml
import logomaker as lm


site_color_schemes = pd.read_csv('data/site_class.csv',index_col=0)

site_color_schemes["color"]=site_color_schemes["class"].map({"n":"#808080","1":"#044B7F","2":"#E3B505","3":"#610345","1+2":"#044B7F","4":"#107E7D"})
site_color_schemes=site_color_schemes.rename(columns={"site":"pos"})
site_color_schemes.head()

with open('data/Covid19_RBD_seq.txt','r') as f:
    wt_seq=f.read()
wt=[str(x)+str(y) for x,y in zip(list(wt_seq),list(range(331,532)))]
wt_dict=pd.DataFrame({"pos":list(range(331,532)),"wt":wt}).set_index("pos").to_dict()
wt_dict=wt_dict["wt"]


def generate_psedo_count_matrix(pcm_1, pcm_2, len_1, len_2, pseudo_count=0.01):
    # one thing we need to make sure is that the zero counts do not influence our results 
    # As we only focus on the true "mutational" change found by our model
    # so set different pseudo counts according to the size of the two datasets
    pcm_1 = pcm_1 + pseudo_count
    pcm_2 = pcm_2 + pseudo_count*len_2/len_1
    return pcm_1, pcm_2

def get_frequency_matrix(pcm_1, pcm_2):
    # for consistency and avoiding misleading, compute the frequency matrix only from count matrix 
    pfm_1 = pcm_1/np.sum(pcm_1, axis=1)[0]
    pfm_2 = pcm_2/np.sum(pcm_2, axis=1)[0]
    return pfm_1, pfm_2


def kl_logo_matrix(pfm_1, pfm_2, mode="prob_weight_KL"):
    log_pfm_2 = np.log(pfm_2)
    log_pfm_1 = np.log(pfm_1)
    if mode == "prob_weight_KL":
        KL_site = np.sum(pfm_2*np.log(pfm_2) - pfm_2*np.log(pfm_1), axis=1)
        #print(np.sum(pfm_2 * np.abs(np.log(pfm_2/pfm_1)), axis=1).shape)
        #print(np.sum(pfm_2 * np.abs(np.log(pfm_2/pfm_1)), axis=1))
        site_norm = np.sum(pfm_2 * np.abs(np.log(pfm_2/pfm_1)), axis=1)#[:, np.newaxis]
        #kl_logo = KL_site[:, np.newaxis] * pfm_2 * np.log(pfm_2/pfm_1) / site_norm
        kl_logo = KL_site * pfm_2 * np.log(pfm_2/pfm_1) / site_norm
        return kl_logo
    elif mode=="KL_site":
        KL_site = np.sum(pfm_2*np.log(pfm_2) - pfm_2*np.log(pfm_1), axis=1)
        return KL_site
    
def get_top_df (gisaid_attack,top_n,attack2_init,add):
    gisaid_attack=gisaid_attack.sort_values(by="mean_score",ascending=False)[:top_n]
    counts_mat_init=lm.alignment_to_matrix(attack2_init["seq"])
    counts_mat_attack=lm.alignment_to_matrix(gisaid_attack["seq"])
    counts_mat_init,counts_mat_attack=generate_psedo_count_matrix(counts_mat_init,counts_mat_attack,attack2_init.shape[0],gisaid_attack.shape[0],add)
    pfm_init,pfm_attack2=get_frequency_matrix(counts_mat_init,counts_mat_attack)
    KL=kl_logo_matrix(pfm_init,pfm_attack2,mode="prob_weight_KL")
    KL["pos"]=list(range(331,532))
    df=pd.melt(KL,id_vars="pos")
    df=pd.merge(df,site_color_schemes[["pos","color","ACE2"]],on="pos")
    df["wt_label"]=df.pos.map(wt_dict)
    df["shade_alpha"]=0.1
    df=df.rename(columns={"value":"prob_weight_KL"})
    return df

# def save_figure(df,add,top,measure,KL_site,n):
#     fig, ax = dmslogo.draw_logo(data=df[df["pos"].isin(KL_site.KL_site.sort_values(ascending=False)[:n].index)],
#                         x_col='pos',
#                         letter_col='variable',
#                         letter_height_col=measure,
#                         heatmap_overlays=["Relative Entropy"],
#                         color_col="color",
#                         xtick_col="wt_label",
#                         xlabel="Site",
#                         ylabel="Bits",
#                         shade_color_col='ACE2',
#                         shade_alpha_col='shade_alpha',
#                         axisfontscale= 1.2,
#                         letterheightscale=1,
#                         # fontfamily="Arial"
#                         )

#     fig.savefig('figures/%s_%s_%s_seqlogo_add%s.png'%(str(n),"top"+str(top),measure,str(add)),bbox_inches='tight',dpi=500)

def get_KL_site(add,top,gisaid_attack,attack2_init):
    gisaid_attack=gisaid_attack.sort_values(by="mean_score",ascending=False)[:top]
    counts_mat_init=lm.alignment_to_matrix(attack2_init["seq"])
    counts_mat_attack=lm.alignment_to_matrix(gisaid_attack["seq"])
    counts_mat_init,counts_mat_attack=generate_psedo_count_matrix(counts_mat_init,counts_mat_attack,attack2_init.shape[0],gisaid_attack.shape[0],add)
    pfm_init,pfm_attack2=get_frequency_matrix(counts_mat_init,counts_mat_attack)
    KL_site=kl_logo_matrix(pfm_init,pfm_attack2,"KL_site")
    KL_site.index=KL_site.index+331
    return(KL_site)

def draw(synthetic_init, synthetic):
    ## set the pseudo counts and the sequence number selected
    add=0.1
    top=synthetic.shape[0]

    KL_site=get_KL_site(add,top,synthetic,synthetic_init)
    KL_site=KL_site.to_frame(name="KL_site")
    
    ## get the dataframe contain all the information
    df=get_top_df(synthetic,top,synthetic_init,add)
    
    df=df.merge(KL_site,on="pos")
    
    sites=list(range(331,532))
    values=[]
    for site in sites:
        values.append(df[df['pos']==site]['KL_site'].values[0])

    df1=pd.DataFrame()
    df1['index']=sites
    df1['values']=values
    
    df1.to_csv('BA5_data.csv', index=False)
    
    print(df1.sort_values('values',ascending=False))
    print(df1.sort_values('values',ascending=False).head(10))
    
    plt.plot(sites,values)
    plt.show()


name='BA.5'
df=pd.read_csv('data/success_BA5.txt'.format(name),sep='\t',names=['seq_ori','col1','col2','seq','col3','mean_score'])

print(df.shape)
print(df.head())
df[['seq','mean_score']].to_csv('data/synthesize_{}.csv'.format(name))





synthetic_init=pd.read_csv('data/BA.5_origin.csv',index_col=0)
synthetic=pd.read_csv('data/synthesize_BA.5.csv')
draw(synthetic_init, synthetic)