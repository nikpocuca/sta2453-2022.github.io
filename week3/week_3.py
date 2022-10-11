import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

import seaborn as sns 

from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression 


def load_french_motor(): 
    """
        Loads french motor data. 
    """
    df = pd.read_csv("french_motor.csv")
    df = df.iloc[:,1:]
    return df 

def plot_motor_grey(df): 

    color = "grey" 
    
    sns.scatterplot(x=df["dens"], y = df["y_log"], 
                     color = color, 
                     s = 5.0) 
    
   # plt.savefig("motor_grey.png"); print("plot motor_grey.png saved"); 

def plot_motor_lr(df, y_hat): 

    df["y_hat"] = y_hat 
    sns.scatterplot(x = df["dens"], 
                y = df["y_hat"], 
                color = "red", 
                s = 6,
                edgecolor = "black")
    
    #plt.savefig("motor_grey_lm.png"); print("plot motor_grey_lm.png saved"); 
    #plt.clf()

def plot_motor(df): 

    cplt = sns.color_palette("bright", 6) 
    
    df["color"] = df.labs.apply(lambda x: cplt[x])

    sns.scatterplot(x=df["dens"], y = df["y_log"], 
                     color = df.color, 
                     s = 5.0)  
    plt.savefig("motor.png"); print("plot motor.png saved"); 


if __name__ == "__main__": 
    print("Running ")
    
    # load dataset. 
    df = load_french_motor()
    
    # numpy matrices. 
    X = df[['dens']]
    y = df['y_log'] 
    labs = df["labs"]

    # define model.
    #lm = LinearRegression() 
    #lm.fit(X,y) 
    #y_hat = lm.predict(X) 
    
    # fit a logistic regression model.
    logr = LogisticRegression()
    
    Xr = df[['dens']]
    y = df['y_log'] 
    logr.fit(Xr,labs)
    
    Z = logr.predict_proba(Xr)
    labs_hat = logr.predict(Xr)
    
    weights_1 = Z[:,0]
    weights_2 = Z[:,1]
    weights_3 = Z[:,2] 

    lm1 = LinearRegression() 
    lm1.fit(X,y, sample_weight = weights_1)
    lm2 = LinearRegression() 
    lm2.fit(X,y, sample_weight = weights_2)
    lm3 = LinearRegression() 
    lm3.fit(X,y, sample_weight = weights_3)

    y_hat_1 = lm1.predict(X) 
    y_hat_2 = lm2.predict(X) 
    y_hat_3 = lm3.predict(X)

    plot_motor_grey(df) 
    plot_motor_lr(df, y_hat_1) 
    plot_motor_lr(df, y_hat_2)
    plot_motor_lr(df, y_hat_3)
    
    plt.savefig("three_plot.png") 
    plt.clf() 
    
    #plot_motor_grey(df) 

