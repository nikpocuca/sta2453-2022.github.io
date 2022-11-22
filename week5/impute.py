import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from sklearn.mixture import GaussianMixture 
from scipy.stats import multivariate_normal

from cm_impute_and_plot import *

def load_shanghai_dataset():
    """
        Load shanghai dataset with missing data. 
    """
    
    df = pd.read_csv("~/gits/sta2453-2022.github.io/SH_CLAY_11_4051.csv").iloc[1:,:]
    df = df.iloc[:,2:6]
    df.columns = ["Y_LL","Y_PI","Y_LI","Y_e"]
    
    return df 


def density(xi: np.array, pis: np.array, mus: np.array, Ss:np.array):
    """ 
        Calculate log density of a multivariate normal density
    """
    
    p = xi.shape[0]
    leading_constant = (np.pi * 2.0) ** (- p * 0.5)
    
    xm = xi - mus
    xm = np.expand_dims(xm,axis=0)
    Ss_inv = np.linalg.inv(Ss)
    res = (xm.transpose(1,0,2) @ Ss_inv).transpose(1,0,2)
    res = np.exp( ( -0.5 * (res * xm).sum(-1)) )
   
    # result 
    res = pis * np.sqrt(np.linalg.det(Ss_inv)) * res 
    res = float(leading_constant) * res	
    
    return res 



def conditional_mean_imputation(X: np.array, pis: np.array, mus: np.array, Ss: np.array):
    """
        Calculates cmi imputation based on z, mu, sigma, and x, for partials. 
        
        Takes in a vector x (1,p) 
        
    """

    # get missing vector index. 
    
    mis = pd.DataFrame(X).isna().to_numpy() 

    for i in range(X.shape[0]):
        xi = X[i,:] 
        mi = mis[i,:] 
        # get non missing entries
        xi_d = xi[~mi]

        # get parameters. 
        mu_d = mus[:, ~mi]
        mu_m = mus[:, mi] 
        Ss_dd = Ss[:, ~mi][:,:, ~mi]
        Ss_md = Ss[:, mi][:,:,~mi]
        Ss_mm = Ss[:, mi][:,:,mi]

        # compute conditional means. 
        dd_diff = np.expand_dims(xi_d - mu_d,-1)
        
        # leading matrix term. 
        lmatrix_term = Ss_md @ ( np.linalg.inv(Ss_dd))
        mu_tildes = mu_m + (lmatrix_term @ (dd_diff))[:,:,0] 

        zigs = density(xi_d, pis, mu_d, Ss_dd) 
        zigs = zigs / zigs.sum()
        zigs[0,-1] = 1.0 - zigs[0,:-1].sum()
        zigs = zigs.transpose() 
        xi_m_imputed = (mu_tildes * zigs).sum(0)
        xi[mi] = xi_m_imputed 
            

    return X 






if __name__ == "__main__": 
    print("Imputing dataset.")

    df = load_shanghai_dataset()
    

    # lets segment the dataset into missing/non-missing respectevly.
    
    # get na mask 
    na_mask = df.isna().any(axis=1) 
    df_na = df.loc[na_mask,:]
    
    # remove fully missing observations 
    df_na = df_na.loc[~df_na.isna().all(axis=1),:].copy() 
    
    # clean 
    df_clean = df.loc[~na_mask,:]

    # fit finite mixture model
    G = 5
    mm = GaussianMixture(n_components=G)
    
    # fit 
    mm.fit(df_clean.to_numpy())

    # acquire parameters.
    mus = mm.means_
    Ss = mm.covariances_
    pis = mm.weights_

    # impute dataset. 
    x_imputed_numpy = conditional_mean_imputation(np.copy(df_na.to_numpy()), pis, mus, Ss);
    df_imputed = pd.DataFrame(x_imputed_numpy)    
    
    df_imputed.columns = df_clean.columns    

    
    # plots 
    ax1, ax2, ax3 = plot_li_vs_others(df_clean)
    plot_imputed_values(df_imputed, ax1, ax2, ax3)   

    # save figure 
    plt.savefig("cm_imputation_plot.png", dpi=600) 
    plt.clf()
    plt.rcParams['text.usetex'] = False
    plt.close() 




    import pdb; pdb.set_trace() 







