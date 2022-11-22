import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import math as m 

import importlib
import cw_model 
importlib.reload(cw_model)

def load_french_motor():     
    """    
        Loads french motor data.     
    """    
    df = pd.read_csv("../week3/french_motor.csv")    
    df = df.iloc[:,1:]    
    return df   


if __name__ == "__main__": 
    print("Running")

    # define data. 
    G = 3 
    #data_s = torch.rand(100,3)
    data_df = load_french_motor()
    data_s = data_df[['y_log','dens']].to_numpy()
    data_s = torch.Tensor(data_s)
    data_s = (data_s - data_s.mean())/data_s.std()  
    
    # define model. 
    cwm = cw_model.ClusterWeightedModel(G=G, data=data_s)
    
    # run some tests. 
    test_lg = cwm.log_density(cwm.X, cwm.y)
    test_obj = cwm.objective_fn(cwm.X,cwm.y)

    # run a training 
    cwm.train(lr=1e-2, max_iterations=1000)
    
    cwm.plot(1) 
    labs = cwm.MAP(cwm.X, cwm.y)
    cwm.plot_colors(1, labs)



