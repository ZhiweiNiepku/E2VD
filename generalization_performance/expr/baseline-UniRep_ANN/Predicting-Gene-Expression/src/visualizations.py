import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from nxviz.utils import infer_data_type, num_discrete_groups, cmaps, n_group_colorpallet, is_data_diverging
from matplotlib.cm import get_cmap
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch, Rectangle
import networkx as nx
from nxviz.plots import CircosPlot
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_auc_score, roc_curve, auc
from collections import defaultdict
import seaborn as sns

from .extra import c2b, c2l, c2c, c2b_abs, c2l_abs, c2c_abs


def corr_pval(data, col1, col2, alpha=0.05, p_fmt=':.8f', c_fmt=':.3f'):
    """Calculate pairwise correlations, p-values and adjusted p-values with Bonferroni"""

    # compute pairwise correlations and p-values
    out = {c: {} for c in col1}
    for c1 in col1:
        for c2 in col2:
            x = data[c1].values
            y = data[c2].values
            
            nas = np.logical_or(np.isnan(x), np.isnan(y))
            
            try:
                c, p = spearmanr(x[~nas], y[~nas])
            except:
                continue
            
            out[c1][c2] = {'corr': c, 'p-value': p}
    
    # convert dictionary to a pandas dataframe
    out = {(i,j): out[i][j] for i in out.keys() for j in out[i].keys() }
    mux = pd.MultiIndex.from_tuples(out.keys(), names=['x', 'y'])
    df = pd.DataFrame(list(out.values()), index=mux)

    df['reject'] = df['p-value'] < alpha

    # calculate adjusted p-values with Bonferroni method
    rejected, pvals_adj, _,_ = multipletests(df['p-value'], method='bonferroni', alpha=alpha)
    df['ms_reject'] = rejected
    df['ms_adj_pval'] = pvals_adj
    
    # make text annotations (for plotting)
    anno_str = 'corr {' + c_fmt + '}\np-value:{' + p_fmt + '}'
    df['annotation'] = df.apply(
        lambda x: anno_str.format(x['corr'], x['p-value']),
        axis=1
    )

    df['annotation_ms'] = df.apply(
        lambda x: anno_str.format(x['corr'], x['ms_adj_pval']),
        axis=1
    )
    
    return df

def compute_edge_colors(G):
    """ 
    Compute the edge colors. 
    NOTE: This function is copied from the nxviz package! 
    """
    data = [G.graph.edges[n][G.edge_color] for n in G.edges]
    data_reduced = sorted(list(set(data)))
    dtype = infer_data_type(data)
    n_grps = num_discrete_groups(data)

    if dtype == "categorical" or dtype == "ordinal":
        if n_grps <= 8:
            cmap = get_cmap(
                cmaps["Accent_{0}".format(n_grps)].mpl_colormap
            )
        else:
            cmap = n_group_colorpallet(n_grps)
    elif dtype == "continuous" and not is_data_diverging(data):
        cmap = get_cmap(cmaps["weights"])

    for d in data:
        idx = data_reduced.index(d) / n_grps
        G.edge_colors.append(cmap(idx))
    # Add colorbar if required.
    if len(data_reduced) > 1 and dtype == "continuous":
        G.sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(
                vmin=min(data_reduced),
                vmax=max(data_reduced),  
            ),
        )
        G.sm._A = []
    return cmap


def circos_cmap2legend(c, data, alpha):
    """Create custom legend to add to nxviz circos plot"""
    cmap = compute_edge_colors(c)
    
    i2cmap = {i: cmap(i) for i in data['inter_idx'].unique()}
    bounds = sorted(list(i2cmap.keys()))
    i2label = data[['inter_idx', 'inter2']].drop_duplicates().set_index('inter_idx').to_dict()['inter2']
    patches = [mpatches.Patch(color=i2cmap[i], label=i2label[i], alpha=alpha) for i in bounds]
    return patches

def custom_legend(data, title, length, label='inter2', color='edge_color', sort_by='interpretation', alpha=.5):
    legend_elements = [Rectangle((0,0), 0, 0, color='w')]
    
    #title = '$\textbf{{{0}}}$'.format(title)
    labels = [title]

    if isinstance(data, pd.DataFrame):
        data_records = data.sort_values(by=sort_by)[[label, color]].to_records(index=False)
        seen = [] 
        for k in data_records:
            if k not in seen:
                legend_elements.append(
                    mpatches.Patch(color=k[1], alpha=alpha)
                )
                labels.append(k[0])
                seen.append(k)
    elif isinstance(data, dict):
        for k, v in data.items():
            legend_elements.append(
                mpatches.Patch(color=v, alpha=alpha)
            )
            labels.append(k)
        
    if len(legend_elements) < length:
        diff = length - len(legend_elements)
        for _ in range(diff):
            legend_elements.append(Rectangle((0,0), 0, 0, color='w'))
            labels.append('')

    return legend_elements, labels

def interpret_corr(data, col='corr'):
    """Wrapper function to help creating circos legend"""

    if 'abs' in col:
        corr_indexer = c2b_abs
        indexer_label = c2l_abs
        indexer_color = c2c_abs
    else:
        corr_indexer = c2b
        indexer_label = c2l
        indexer_color = c2c

    for l, bs in corr_indexer.items():
        data.loc[data[col].between(bs[0], bs[1], inclusive=True), 'interpretation'] = l
    
    data['inter2'] = data['interpretation'].map(indexer_label)
    label_reduced = sorted(list(data['interpretation'].unique()))
    r2i = {l: i for i, l in enumerate(label_reduced)}
    data['inter_idx'] = data['interpretation'].map(r2i) / len(label_reduced)
    data['edge_color'] = data['interpretation'].map(indexer_color)
    
    return data


def create_circos(data, source, target, edge_attr, node_classes, node_grouping, edge_width, edge_color, legend_label, legend_color, legend_sort, node2color, figsize=(15,15), alpha=.5, title=None, group_label_offset=10, legend_length=9, **kwargs):
    """Create a circos plot with correlations"""

    # create nodes and edges
    G = nx.from_pandas_edgelist(data, source=source, target=target, edge_attr=edge_attr)
    
    # classify a node
    for n, _ in G.nodes(data=True):
        G.nodes[n]["class"] = node_classes[n]
    
    # initialize Circos plot
    c = CircosPlot(
        G,
        node_grouping=node_grouping,
        edge_width=edge_width,
        edge_color=edge_color,
        figsize=figsize,
        node_labels=True,
        node_label_layout="rotation",
        group_label_position=None,
        group_label_color=True,
        group_label_offset=group_label_offset,
        **kwargs
    )
    
    # set alpha of edges 
    c.edgeprops['alpha'] = alpha

    # set node colors
    node_colors = [] 
    for l in c.nodes:
        lc = l.split('(')[0].strip()
        co = node2color[lc]
        c_rgba = to_rgba(co)
        node_colors.append(c_rgba)
    c.node_colors = node_colors
    c.node_size = .5

    # set edge colors
    edge_colors = [] 
    for e in c.edges:
        try:
            ec = data.loc[
                (data[source] == e[0]) & 
                (data[target] == e[1]),
                legend_color
            ]
            edge_color=ec.item()
        except ValueError:
            ec = data.loc[
                (data[source] == e[1]) & 
                (data[target] == e[0]),
                legend_color
            ]
            edge_color=ec.item()
        except Exception as err:   
            print(e)
            print(ec)
            raise err

        edge_colors.append(edge_color)
    c.edge_colors = edge_colors

    # actually draw the Circos plot
    c.draw()

    # get matplotlib.pyplot figure to add legend and title
    fig = c.figure
    
    # create legend for edge colors
    edge_legends, edge_labels = custom_legend(
        data=data, 
        title='Edge colors', 
        length=legend_length, 
        label=legend_label, 
        color=legend_color, 
        sort_by=legend_sort, 
        alpha=alpha
    ) 
    # create legend for node colors
    class2color = {}
    for k, v in node_classes.items():
        if k.startswith('UR_'): k = k.split('(')[0].strip()
        class2color[v] = node2color[k]

    node_legends, node_labels = custom_legend(
        data=class2color, 
        length=legend_length, 
        title='Node colors', 
        alpha=alpha)
    
    # add legend
    fig.legend(
        handles = edge_legends + node_legends, labels=edge_labels + node_labels,
        ncol=2,
        loc='lower right'
    )
    
    if title is not None:
        fig.suptitle(title, fontsize=16)
    
    plt.close(fig)
    
    return fig


def prepare_circos_data(corrs, feature_imps, node_classes, groupby_col='x', sort_col='abs_corr', interpret_col='corr', top_ns=[1, 3], filter_top_units=0):

    corrs = corrs.copy()
    if filter_top_units > 0:
        top_units = feature_imps.sort_values(by='importance (%)', ascending=False).head(filter_top_units).index.tolist()
        top_units = ['UR_' + str(x) for x in top_units]
        corrs = corrs.loc[
            corrs.y.isin(top_units)
        ]
    
    ds = defaultdict(list)
    for _, gdata in corrs.groupby(groupby_col):
        for top_n in top_ns:
            ds[top_n].append(
                gdata.sort_values(by=sort_col).tail(top_n)
            )
    
    ds = {
        k: pd.concat(v, axis=0) for k, v in ds.items()
    }

    uniques = list()
    for k, df in ds.items():
        df = interpret_corr(df, col=interpret_col)
        df['size'] = k * 5
        df['importance'] = df.y.apply(
            lambda u: feature_imps.loc[int(u.split('_')[-1]), 'importance (%)']
        )

        df['y_ann'] = df.apply( lambda x: '{} ({:.3f}%)'.format(x['y'], x.importance), axis=1)
        df['corr_size'] = df['abs_corr'] * 10

        uniques += df.x.unique().tolist() + df.y.unique().tolist()
    
    uniques = list(set(uniques))

    node_classes_e = {k: v for k,v in node_classes.items() if k in uniques}

    return ds, node_classes_e


def create_tax_labels(data):
    data['tax_label'] = 'Other'
    data.loc[(data.tax_kingdom == "Bacteria") & (data.tax_phylum=="Firmicutes"), 'tax_label'] = 'Bacteria-Firmicutes'
    data.loc[(data.tax_kingdom == "Bacteria") & (data.tax_phylum=="Proteobacteria"), 'tax_label'] = 'Bacteria-Proteobacteria'
    data.loc[(data.tax_kingdom == "Bacteria") & (data.tax_phylum=="Bacteroidetes"), 'tax_label'] = 'Bacteria-Bacteroidetes'
    data.loc[(data.tax_kingdom == "Bacteria") & (data.tax_phylum=="Actinobacteria"), 'tax_label'] = 'Bacteria-Actinobacteria'
    data.loc[(data.tax_kingdom == "Bacteria") & (data.tax_label=="Other"), 'tax_label'] = 'Bacteria-Other'
    data.loc[(data.tax_kingdom == "Metagenome"), 'tax_label'] = 'Metagenome'
    data.loc[(data.tax_kingdom == "Archaea"), 'tax_label'] = 'Archaea'
    tax_labels = data.tax_label.unique()

    return data, tax_labels

def eval_class(model, data, col_true, col_score, group_column, key):
    scores = {}
    aucs = []  
    
    fpr_col = 'FPR'
    tpr_col = 'TPR'
    auc_col = 'ROC AUC'
    
    for g, gdata in data.groupby(group_column):
        fpr, tpr, _ = roc_curve(gdata[col_true], gdata[col_score])
        roc_auc = auc(fpr, tpr)

        scores[g] = {
            fpr_col: fpr, 
            tpr_col: tpr, 
            auc_col: roc_auc
        }
        if not np.isnan(roc_auc):
            aucs.append(roc_auc)
    
    # calculate average (macro) AUC = (sum(AUCs))/(n_groups)
    scores['macro_average'] = {auc_col: np.mean(aucs)}
    
    # evaluate on full data
    fpr, tpr, _ = roc_curve(data[col_true], data[col_score])
    roc_auc = auc(fpr, tpr)
    scores['Ungrouped'] = {fpr_col: fpr, tpr_col: tpr, auc_col: roc_auc}
    
    scores = pd.DataFrame.from_dict(scores, orient='index')
    scores.index.name = group_column
    return scores


def eval_models(data, models, col_true, col_score, group_col):
    results, names = [], []
    for model_name, model_attr in models.items():
        data = data.copy()
        if model_name == 'UniRepRF':
            model = model_attr['model']
        else:
            model = model_attr['model'].baseline_rf
        
        input_cols = model_attr['input_cols']
        
        if not all([c in data.columns for c in input_cols]):
           print("Not correct columns in data for model {}".format(model_name))
           continue
        
        data[col_score] = model.predict_proba(data[input_cols])[:, 1]
        results.append(
            eval_class(model, data, col_true=col_true, col_score=col_score, group_column=group_col, key=model_name)
        )
        names.append(model_name)
    results = pd.concat(results, axis=1, keys=names)
    return results

def plot_seed_imps(counts, place_imps, figsize=(10,7), title='', plot_average_bars=False):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    avg_imps = pd.DataFrame(place_imps.groupby(['UR']).importance.mean()).sort_values(by='importance')
    avg_imps_index = {x: str(int(x)) if isinstance(x, int) else x for x in avg_imps.index}
    avg_imps.index = avg_imps.index.map(avg_imps_index)
    
    # make feature column/index to str format 
    place_imps['UR'] = place_imps['UR'].map(avg_imps_index)
    counts.index = counts.index.map(avg_imps_index)
    
    unit_order = avg_imps.index.tolist()
    place_imps['sort_order'] = place_imps['UR'].apply(lambda x: unit_order.index(x))

    place_imps.sort_values(by='sort_order', inplace=True)
    counts = counts.loc[unit_order, ]

    avg_imps.reset_index(inplace=True)
    
    counts.plot.barh(stacked=True, ax=axes[0])
    axes[0].set_ylabel('UR/Feature')
    axes[0].set_xlabel('Times picked')
    axes[0].legend(title='Place')

    if plot_average_bars:
        avg_imps['hue'] = 'Mean'
        sns.barplot(
            data=avg_imps,
            y='UR',
            x='importance',
            orient='h',
            alpha=.3,
            ax=axes[1],
            order=unit_order,
            hue='hue'
            
        )


    sns.catplot(
        data = place_imps,
        y = 'UR',
        x = 'importance',
        hue='place',
        ax=axes[1],
        kind='swarm',
        orient='h',
        alpha=.9,
        order=unit_order
    )
    
    axes[0].set_title(title)
    axes[1].set_ylabel('Importance (%)')
    fig.tight_layout()
    plt.close('all')

    return fig, axes

def inspect_seed_imps(results, top_n=10, title='', plot_average_bars=False):

    places = defaultdict(list)
    imps = defaultdict(list)
    for i, row in results.iterrows():
        top_vals = row.sort_values(ascending=False).head(top_n)
        for k, v in enumerate(top_vals.iteritems()):
            f = v[0]
            if isinstance(f, int): f += 1
            places[i].append(f)
            imps[i].append(v[1])
    
    # conver dicts to dataframes
    places = pd.DataFrame.from_dict(places, orient='index')
    places.columns += 1

    imps = pd.DataFrame.from_dict(imps, orient='index')
    imps.columns += 1

    # combine into one df
    picked = pd.concat([places, imps], keys=['places', 'imps'], axis=1)

    # now get average importance per importance place and average across all places
    UR_place_imp = pd.DataFrame(columns=['UR', 'place', 'importance'])
    i = 0
    for _, row in picked.iterrows():
        for k, ur in row['places'].iteritems():
            UR_place_imp.loc[i] = [ur, k, row['imps'][k]]
            i += 1

    UR_counts = places.apply(lambda x: x.value_counts())

    fig, _ = plot_seed_imps(UR_counts, UR_place_imp, title=title, plot_average_bars=plot_average_bars)

    return picked, UR_place_imp, UR_counts, fig