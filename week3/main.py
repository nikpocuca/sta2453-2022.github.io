# standard preamble. 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch 
import seaborn as sns 

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression



def weighted_lm(X,W,y):
    """
        Weighted linear model. 
    """

    lm = {'X':X,
          'W':W, 
          'y':y}

    return lm 

# entry point for the program. 
if __name__ == "__main__":
    print("Running ")

    cpat = sns.color_palette('pastel',3)
    df = pd.read_csv("french_motor.csv").iloc[:,1:]
    df['colors'] = df.labs.apply(lambda x: cpat[x-1])
    
    sns.scatterplot(x=df.dens, 
                    y=df.y_log, 
                    color=df['colors'],
                    edgecolor = 'black',
                    s=5.0)
    plt.savefig('french_motor.png')
    plt.clf()

    
    # set up X and y 
    X = df[['dens']].to_numpy() 
    y = df['labs'].to_numpy() 

    # logistic regression model. 
    lr_model = LogisticRegression(random_state=0).fit(X, y)

    # get some weights and assignments 
    hard_class = lr_model.predict(X)
    soft_class = lr_model.predict_proba(X)
    
    # weighted regression. 
    lm = 
