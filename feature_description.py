import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jaccard
import os

# Miscellaneous
Categories = ["AnGn", "AnGp",  "Antiviral", "Antifungal", "Anticancer", "Antimammal", "AntiMRSA"]
AALetter = ['A', 'C', 'D', 'E', 'F', 'G', 'H',
            'I', 'K', 'L', 'M', 'N', 'P', 'Q',
            'R', 'S', 'T', 'V', 'W', 'Y']
features = ["IEP", "Net Charge", "Hydrophobicity", "Hydrophobic Moment",
            "Transmembrane Propensity", "Boman Index", "Aliphatic Index",
            "Alpha Helical Propensity"]


"""
Function for plot violin plot of specific feature
Input:
    data_dict: dictionary of data with categories
    feature: feature to plot
    color: color for violin plot
Output:
    violin plot for specific feature
"""


def violin_feature(data_dict, feature="iep", color="#D43F3A", figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    vp = ax.violinplot(list(map(lambda df, feature: df[feature].values,
                                data_dict.values(), [feature] * len(data_dict))),
                       showmeans=True, showmedians=False)
    ax.set_xticks(np.arange(1, len(data_dict) + 1))
    ax.set_xticklabels(list(data_dict.keys()))

    for part_name in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        vpl = vp[part_name]
        vpl.set_edgecolor('#000000')
        vpl.set_linewidth(1.2)
    vp['cmeans'].set_hatch('o')
    for pc in vp['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor('#000000')
        pc.set_alpha(.78)
        pc.set_linewidth(1.2)
    if feature == 'iep':
        feature_name = 'IsoElectric Point'
    else:
        feature_name = feature
    ax.set_title(feature_name)
    return fig


"""
Function for plot mean AA Composition bars
Input:
    data_dict: dictionary of data with categories
Output:
    AA Composition plot with categories and documented mean AAC
"""


def mean_aac_plot(data_dict):

    # mean non-AMP from swissprot/uniprot
    # Classified by Polar Property
    aa_prop = {
        'Hydrophobic': ["A", "I", "L", "M", "V", "F", "W", "Y"],
        'GP': ["G", "P"],
        'Polar(Neutral)': ["N", "C", "Q", "S", "T"],
        'Polar(Acidic)': ["D", "E"],
        'Polar(Alkaline)': ["R", "H", "K"]
    }
    # Colors for categories include Uni-prot
    colors = ["#D43F3A", "#20B2AA", "#228B22", "#FF8C00",
              "#1E90FF", "#F4A460", "#C71585", "#808080"]

    grid = plt.GridSpec(2, 10, wspace=1.6, hspace=0.22)
    plot_areas = [grid[0, :8], grid[0, 8:],
                 grid[1, :5], grid[1, 5:7], grid[1, 7:]]

    # Get mean AA Composition
    def get_mean_acc(seq_df):
        aac_index = list(map(lambda aa: 'AAC_{}'.format(aa), AALetter))
        return seq_df[aac_index].mean()

    # DataFrame of mean_aac, categories as columns
    categorical_mean_aac = dict(zip(data_dict.keys(), map(get_mean_acc, data_dict.values())))
    # Concat mean_aac of swissprot/Uniprot non-AMP
    mean_aac_dbamp = pd.read_csv("./features/comparison-of-amino-acid.csv", index_col="Category")
    mean_aac_dbamp.index = map(lambda aa: "AAC_{}".format(aa), mean_aac_dbamp.index)
    categorical_mean_aac['Uni-prot'] = mean_aac_dbamp['Uniprot/Swiss-prot (557,275)']

    categorical_mean_aac = pd.DataFrame(categorical_mean_aac)
    cate_ind = categorical_mean_aac.columns.values.tolist()

    # Plot with categorical mean AA Composition
    fig = plt.figure(figsize=(16, 8.2))
    b_plots = []
    for prop, ind, p_area in zip(aa_prop.keys(), aa_prop.values(), plot_areas):
        plt.subplot(p_area)
        width, n = 0.1, len(cate_ind)
        total_width = width * n
        x = np.arange(len(ind))
        x = x - (total_width - width) / 2
        count = 0
        b_plots = []
        ind_aa = list(map(lambda aa: 'AAC_{}'.format(aa), ind))
        for cate, color in zip(categorical_mean_aac.columns, colors):
            b_plots.append(plt.bar(x + count * width,
                                   categorical_mean_aac.loc[ind_aa, cate].values,
                                   width=width, label=cate, color=color, lw=.8, edgecolor='#000000'))
            count += 1
        plt.title(prop)
        plt.xticks(x + total_width / 2, ind)
        plt.ylim([0, 20])
        if prop == "Hydrophobic" or prop == "Polar(Neutral)":
            plt.ylabel("Mean AAC (%)")
    fig.legend(b_plots, cate_ind,
               loc="lower center", ncol=len(cate_ind))
    fig.suptitle("Mean AAC of different functions of AMP", fontsize='x-large')
    return fig, categorical_mean_aac


"""
Function for generate diAAC heatmaps for all categories
Input:
    data_dict: dictionary of data with categories
Output:
    diAAC heatmap with categories and documented mean diAAC
"""


def mean_diaac_heatmap(data_dict):

    return


"""
Function for generate scatter plot w.r.t different features
Input:
    data_dict: dictionary of data with categories
    features: features that you want to plot
Output:
    Scatter plot with different features
"""


def feature_scatter(data_dict, features):
    nf = len(features)
    fig, axes = plt.subplots(nf, nf, figsize=(nf * 2.28, nf * 2))
    cmap = plt.get_cmap("Set1").colors
    splots, cates = [], []
    for ii, row in enumerate(axes):
        for jj, col in enumerate(row):
            if not col.is_first_col():
                col.set_yticks([])
            if not col.is_last_row():
                col.set_xticks([])
            cates = []
            for nn, (cc, data) in enumerate(data_dict.items()):
                splots.append(col.scatter(data[features[jj]], data[features[ii]],
                                          label="{} ({})".format(cc, data.shape[0]),
                                          alpha=.24, s=3.0, color=cmap[nn]))
                cates.append("{} ({})".format(cc, data.shape[0]))

    fig.legend(splots, cates, loc="lower center", fontsize='x-large', ncol=len(cates) // 2,
               scatterpoints=3, markerscale=1.6)
    return fig


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=8)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


if __name__ == '__main__':

    plt.rcParams.update({'font.size': 14})
    """ Concat train and test data """
    all_dict = {}
    for c in Categories:
        train_df = pd.read_csv("features/{}_train.csv".format(c))
        test_df = pd.read_csv("features/{}_test.csv".format(c))
        df = pd.concat([train_df, test_df], axis=0)
        df.index = list(range(len(df.index)))
        df.to_csv("features/{}_all.csv".format(c), index=False)
        all_dict[c] = df

    """Plot violin of lengths"""
    for k in all_dict.keys():
        all_dict[k]['SeqLength'] = all_dict[k]['Sequence'].str.len()
    violin_feature(all_dict, feature='SeqLength', color='#FFE4E1', figsize=(10.4, 5))
    plt.savefig("length_violin.png")

    """Plot Jaccard matrix for mtl data"""
    plt.clf()
    mtl_df = pd.read_csv("./features/mtl_all.csv")
    mat_jac = [[1 - jaccard(mtl_df[cc1], mtl_df[cc2]) for cc2 in Categories] for cc1 in Categories]
    ctg_label = ["{:s}\n({:d})".format(c, len(all_dict[c])) for c in Categories]
    mat_jac = pd.DataFrame(mat_jac, index=Categories, columns=Categories)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im, cbar = heatmap(mat_jac, ctg_label, ctg_label, ax=ax, cmap="Reds", cbarlabel="")
    plt.savefig("./jaccard.png")

    """ Create features scatter plot for specific category"""
    for c in Categories:
        spec_dict = {}
        non_spec_df = pd.concat([pd.read_csv("features/non_{}_5_train.csv".format(c)),
                                 pd.read_csv("features/non_{}_5_test.csv".format(c))],
                                axis=0)
        non_spec_df.index = list(range(len(non_spec_df.index)))
        spec_dict['non-{}'.format(c)] = non_spec_df
        spec_dict[c] = all_dict[c]
        plt.clf()
        feature_scatter(spec_dict, features)
        plt.savefig("plots/features_scatter_{}.png".format(c), dpi=480)

    """ Create features scatter plot for Antiviral, AntiBacteria, Antifungal"""
    anti_3_dicts = {}
    mtl_df['AntiBacteria'] = mtl_df['AnGn'] & mtl_df['AnGp']
    for c1 in ['AntiBacteria', 'Antiviral', 'Antifungal']:
        for c2 in ['AntiBacteria', 'Antiviral', 'Antifungal']:
            if c1 == c2:
                pass
                # anti_3_dicts[c1] = (df[df[c1] == 1])[features]
            else:
                anti_3_dicts['{} & {}'.format(c1, c2)] = (mtl_df[(mtl_df[c1] == 1) & (mtl_df[c2] == 1)])[features]
    anti_3_dicts['omni'] = (mtl_df[(mtl_df['AntiBacteria'] == 1) & (mtl_df['Antiviral'] == 1) & (mtl_df['Antiviral'] == 1)])[features]
    plt.clf()
    feature_scatter(anti_3_dicts, features)
    plt.savefig("plots/features_scatter_special.png", dpi=480)

    """ Plot AA Composition """
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.clf()
    mean_aac_fig, mean_aac_df = mean_aac_plot(all_dict)
    plt.savefig("plots/AAComposition.png")
    if not os.path.exists("features/statistics"):
        os.makedirs("features/statistics")
    mean_aac_df.to_csv("features/statistics/mean_aac.csv")

    """ Plot violin features """


    """ Create Violin Plots for features of all categories """
    if not os.path.exists("plots/violin"):
        os.makedirs("plots/violin")
    for feature in features:
        plt.clf()
        violin_feature(all_dict, feature=feature, color="#E6E6FA", figsize=(10.4, 5))
        plt.savefig("plots/violin/{}.png".format(feature))

    """ Create Violin Plots for features of AntiMRSA v.s. non-AntiMRSA """
    mrsa_dict = {}
    non_mrsa_df = pd.concat([pd.read_csv("features/non_AntiMRSA_5_train.csv"),
                             pd.read_csv("features/non_AntiMRSA_5_test.csv")],
                            axis=0)
    non_mrsa_df.index = list(range(len(non_mrsa_df.index)))
    mrsa_dict['non-AntiMRSA'] = non_mrsa_df
    mrsa_dict['AntiMRSA'] = all_dict['AntiMRSA']

    if not os.path.exists("plots/violin_MRSA"):
        os.makedirs("plots/violin_MRSA")
    for feature in features:
        plt.clf()
        violin_feature(mrsa_dict, feature=feature, color="#F0FFF0", figsize=(4.6, 6))
        plt.savefig("plots/violin_MRSA/{}.png".format(feature))

