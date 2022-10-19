import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import math as m 

import importlib
import flm_model 
importlib.reload(flm_model)

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
    flm = flm_model.FiniteLinearModel(G=G, data=data_s)
    
    # run some tests. 
    test_lg = flm.log_density(flm.X, flm.y)
    test_obj = flm.objective_fn(flm.X,flm.y)

    # run a training 
    flm.train(lr=1e-3, max_iterations=1000)
    
    flm.plot(1) 

