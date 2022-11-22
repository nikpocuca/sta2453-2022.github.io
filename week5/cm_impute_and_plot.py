import numpy as np 
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt 
import sys 

def plot_imputed_values(df, ax1, ax2, ax3) -> None:
    """
            Takes in a dataframe and spits out a figure named <fig_name>
            
            
    """
    plt.rcParams['text.usetex'] = True

    marker_size = 1.7 
    alpha = 1.0

    #color_palatte = sns.color_palette("cubehelix",7)
    color_palatte = sns.color_palette("magma", 20)
    colors = "blue" #df.labels.apply(lambda label: color_palatte[0])
    edge_color = "lightblue"# colors

    sns.scatterplot(x='Y_LL', y='Y_LI', data=df, ax=ax1, alpha=alpha, s=marker_size, edgecolor=edge_color, color=colors)
    sns.scatterplot(x='Y_PI', y='Y_LI', data=df, ax=ax2, s=marker_size, alpha=alpha, color=colors, edgecolor=edge_color)
    sns.scatterplot(x='Y_e', y='Y_LI', data=df, ax=ax3, s=marker_size, alpha=alpha, color=colors, edgecolor=edge_color)
    


def plot_li_vs_others(df, fig_name="li_vs_others.png") -> None:
    """
        Takes in a dataframe and spits out a figure named <fig_name>
        
        
    """
    plt.rcParams['text.usetex'] = True


    fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(10,3))
    marker_size = 5.0
    alpha = 0.70

 #   color_palatte = sns.color_palette("bright",10)
#    marker_color_idx = 7
    marker_color = "grey"

    ax1.set_title('')
    sns.scatterplot(x='Y_LL', y='Y_LI', data=df, ax=ax1, alpha=alpha, s=marker_size, color=marker_color)
    ax1.set_xlabel(r"$Y_{LL}$")
    ax1.set_ylabel(r"$Y_{LI}$")

    ax2.set_title('')
    sns.scatterplot(x='Y_PI', y='Y_LI', data=df, ax=ax2, s=marker_size, alpha=alpha, color=marker_color)
    ax2.set_xlabel(r"$Y_{PI}$")
    ax2.set_ylabel("")
    ax2.axes.get_yaxis().set_visible(False)


    ax3.set_title('')
    sns.scatterplot(x='Y_e', y='Y_LI', data=df, ax=ax3, s=marker_size, alpha=alpha, color=marker_color)
    ax3.set_xlabel(r"$Y_{e}$")
    ax3.set_ylabel("")
    ax3.axes.get_yaxis().set_visible(False)
    fig.tight_layout()

    return ax1, ax2, ax3
   # plt.savefig(fig_name, dpi=300)
   # plt.clf()
   # plt.rcParams['text.usetex'] = False
   # plt.close()

def nnscale(df):
    """
        returns a scaled dataframe with data, mu, and stds accordingly. 
    """
    mu = df.mean()
    std = df.std() 
    data = (df - mu)/std
    return (data, mu, std)

def scale(df, mu, std):
    return (df - mu)/std

def unscale(df, mu, std):
    """
        returns an unscaled dataframe. 
    """
    return df*std + mu

"""
    # plot grey only 
    ax1, ax2, ax3 = plot_li_vs_others(df)
    # plot imputed values. 
    plot_imputed_values(dfna_imputed, ax1, ax2, ax3)

    # save plot
    plt.savefig("cm_imputation_plot.png", dpi=600)
    plt.clf()
    plt.rcParams['text.usetex'] = False
    plt.close()
"""
