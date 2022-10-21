import torch 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import scipy 



def objective_fxn(x_: torch.Tensor): 
    """
        Function optimize
    """

    return (fxn(x_) - 2.5)**2

def fxn(x_: torch.Tensor):
    """
        Function to optimize against 
    """

    return (x_ ** 2  - 3*x_ - 4)


def plot_fn(x_, f): 
    """
        Plots the function. 
    """

    plt.plot(x_, f(x_))
    plt.title("Objective Function to Optimzie") 
    plt.savefig("fxn.png") 
    plt.clf() 
    



if __name__ == "__main__":
    print("Running")



    x = torch.linspace(-2.0,6.0,400)

    plot_fn(x, fxn)



    # performing minimization.
    x_star = torch.randn(1) + 10
    
    # set gradient tracking. 
    x_star.requires_grad = True 

    # number of iterations. 
    iterations = 3000
    lr = 1e-2
    xs = torch.zeros(iterations)

    # declare optimizer. 
    optimizer = torch.optim.Adam([x_star], lr=lr)

    # iterate on every step. 
    for i in range(iterations): 

        # always zero the gradient. 
        optimizer.zero_grad() 
        loss = objective_fxn(x_star)
        loss.backward()
        optimizer.step() 

        if i % 100 == 0:
            print(f"Iteration {i} | loss: {float(loss)}")


        xs[i] = float(x_star)





