# Standard preamble. 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import math as m 

# pytorch related stuf.. 
from torch.nn import Module 
import torch.nn as nn 
from torch.nn.functional import softmax 
from tqdm import tqdm 

class FiniteLinearModel(nn.Module):
    """
        Class for Finite mixtures of Linear models. 
    """
   
    def __init__(self, G:int, data: torch.Tensor):
        """
            Constructor class for finite mixtures of linear models.  
        """
        
        if not isinstance(data, torch.Tensor):
            raise Exception("data is not of torch.Tensor type") 

        super(FiniteLinearModel, self).__init__() 

        # define constants. 
        self.n = data.shape[0] 
        self.G = G if G > 0 else exec("raise Exception('G has to be valid')") 

        # define data 
        self.X = data[:,1:]
        self.X = torch.concat( (self.X, torch.ones(self.n,1)), axis=1)
        self.y = data[:,0]

        # first column is y, the rest are covariates. 
        self.p = self.X.shape[1] 

        # define actual parameters. 
        self.betas = torch.rand(G,self.p) 
        self.sigmas = torch.rand(G,1).abs() 
        self.w = torch.rand(G,1) # define weights.  

        # set gradients. 
        self.w.requires_grad = True; 
        self.betas.requires_grad = True;
        self.sigmas.requires_grad = True; 
        self.leading_constants = - 0.5 * torch.Tensor([2.0 * torch.pi]).log()

    def compute_weights(self):
        """
            get mixing proportions. 
        """
        return softmax(self.w[:,0], dim=0)


    def log_density(self, X, y) -> torch.Tensor:
        """
            Takes in a covariate dataset X, and sample y, 
            computes an (n x G) matrix of log densities. 
        """
        
        # calculate log density. 
        ldens = torch.zeros(self.n, self.G) 

        # loop through the groups. 
        for g in range(0,self.G):

            # compute linear model.
            y_hats = (self.betas[g] * self.X).sum(-1)  
            
            # compute exponential terms. 
            exp_terms = - 0.5 * ((y_hats - y) / self.sigmas[g]).pow(2) 
            ldens[:,g] = exp_terms + self.leading_constants 
            ldens[:,g] += - self.sigmas[g].log() 
            
        return ldens  

        
    def objective_fn(self, X, y):
        """
            Objective function to minimize on, 
            takes in an (n x d + 1) matrix with the response variable 
            in the first column 
        """
        
        # compute log densities. 
        dens = self.log_density(X, y).exp() 

        # get weights. 
        W = self.compute_weights() 
        
        return -((dens * W).sum(-1).log()).sum() 
        
    def train(self, lr=1e-3, max_iterations=1000): 
        """
            train using some gradient descent algorithm 
        """

        # define optimizer 
        optimizer = torch.optim.Adam([self.w, 
                                      self.betas, 
                                      self.sigmas], lr=lr)
        # track loss 
        loss = np.zeros(max_iterations)
        tq_obj = tqdm(range(max_iterations), desc="Model Training")
       
        for it in tq_obj:
            optimizer.zero_grad() 
            cost = self.objective_fn(self.X, self.y)
            cost.backward() # compute gradient. 
            optimizer.step() 

            loss[it] = cost.data.cpu().numpy() 
            
            # log the loss. 
            if it % 100 == 0:    
                tqdm.write(f"Loss: {loss[it]}  λ: {optimizer.param_groups[0].get('lr')}")

    def fit(self, X, betas):                                                                                                                            
        """                                                                                                                                              
           Take input covariate X and fits using the coefficients betas.                                                                                
           Returns an (n x G) matrix, one for every component                                                                                           
        """                                                                                                                                              
        
        ys = torch.zeros(X.shape[0], betas.shape[0])                                                                                                     
        
        for g in range(self.G):                                                                                                                         
            ys[:,g] = (betas[g] * X).sum(-1)                                                                                                             
        
        return ys                                                                                                                                        

    def BIC(self):
        """
            Calculates the Bayesian Information Criterion for model performance comparisons.
        """

        # calculate number of parameters. 
        rho = self.betas.numel() + self.sigmas.numel() + self.w.numel()
        bic = self.objective_fn(self.X, self.y) * (-2.0) - rho * m.log(self.y.shape[0])
        return float(bic.detach())


    def plot(self, col):
        """
            col: reference column
        """
         
        plot_df = torch.concat((self.y.unsqueeze(-1), self.X), dim=1)
        plot_df = pd.DataFrame(plot_df.detach().numpy())
        plot_df = plot_df[[0,col]]
        
        sns.scatterplot(x=plot_df[col], y=plot_df[0], color="grey",s=2.0) 
        
        y_fits = self.fit(self.X, self.betas)

        for g in range(self.G):
            plot_df['y_fit'] = y_fits[:,g].detach().numpy()  
            sns.scatterplot(x=plot_df[col], y=plot_df['y_fit'], color="red", s=2.0)

        plt.savefig("flm-fit.png") 
        plt.clf() 

    def plot_colors(self, col, labs):
        """
            col: reference column. 
            labs: integer np.array 
        """
        
        color_plt = sns.color_palette("bright", int(np.max(labs)) + 1)
        
        plot_df = torch.concat((self.y.unsqueeze(-1), self.X), dim=1)
        plot_df = pd.DataFrame(plot_df.detach().numpy())
        plot_df = plot_df[[0,col]]
        plot_df['color'] = pd.Series(labs).apply(lambda x: color_plt[x])

        sns.scatterplot(x=plot_df[col], y=plot_df[0], color=plot_df['color'],s=2.0)         

        plt.savefig("flm-fit-color.png") 
        plt.clf() 

    def Estep(self, X, y):
        """
            Computes the expectation step using parameters for X ,y
        """
        
        with torch.no_grad():
            dens = self.log_density(X, y).exp()

            W = self.compute_weights() 
            dens = dens * W
            
            d_sum = dens.sum(-1).unsqueeze(-1)
            dens = dens / d_sum 

            #dens[:,-1] = 1.0 - dens[:,:-1].sum(-1)
        
        return dens 

    def MAP(self, X, y):
        """
            Computes labels using the maximum a posterori 
        """
        
        dens = self.Estep(X,y)
        labs = dens.argmax(-1) 
        labs = labs.detach().numpy()
        labs = labs.astype(int)
        return labs 



