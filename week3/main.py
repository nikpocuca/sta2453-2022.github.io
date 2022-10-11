# standard preamble. 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch 
import seaborn as sns 

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression 


def my_table(df,lab1,lab2):
    tbl = df.groupby([lab1, lab2]).size()
    tbl = tbl.unstack() 
    return tbl 

def plot_motor_data():
    """
        plots motor policy data with super grey. does not save fig. 
    """

    df = pd.read_csv("french_motor.csv").iloc[:,1:]
    
    sns.scatterplot(x=df.dens, 
                    y=df.y_log, 
                    color="grey",
                    alpha = 0.70, 
                    s=5.0)

# entry point for the program. 
if __name__ == "__main__":
    print("Running ")


    plot_motor_data()
    plt.savefig("french_motor_grey.png") 
    plt.clf() 

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
    # let us regress with the first values.
    y = df['y_log'].to_numpy()
    
    # define linear regression model with fits. 
    lm = LinearRegression() 
    lm.fit(X = X,y = y,sample_weight = soft_class[:,2])
    
    # predict results 
    y_hat = lm.predict(X = X) 

    # assign to dataframe. 
    df['y_hat'] = y_hat 
    df['hard_class'] = hard_class  

    sns.scatterplot(x=df.dens, 
                    y=df.y_hat, 
                    color = 'red') 
    plot_motor_data() 
    plt.savefig("lm_w1_fitted.png") 
    plt.clf()     
    print(my_table(df,'hard_class','labs'))


