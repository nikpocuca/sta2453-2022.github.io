import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt 
import scipy.stats as stats


def sample_mean(x: np.array) -> float: 
    """ 
        Take a given x and calculate sample mean. 
    """
    return np.sum(x)/x.shape[0]


def sample_std(x: np.array) -> float: 
    """
        Take a given x, calculate sample mean, then calculate sample std. 
    """
    mu = np.mean(x) 
    n = x.shape[0] 
    return np.sqrt(  np.sum((x - mu) ** 2 / (n-1)) ) 


def sample_skewness(x: np.array) -> float: 
    """
        Takes in a given x, 
        calculate sample mean, 
        sample std, and then calculate sample skewness.
    """
    
    # calculate sample mean. 
    n = x.shape[0]
    mu = sample_mean(x) 
    var = sample_std(x) ** 2 
    var = var * (n-1) / n
    
    M3 = np.sum( ((x - mu) ** 3 / n) )
    kappa = M3 / (var ** (1.5))
    kappa = (np.sqrt(n * (n - 1)) / (n - 2)) * kappa
    return kappa
    

def gill_statistic(x: np.array) -> float: 
    """
        See following for specifics.         
        https://www.jstor.org/stable/2988433?seq=6#metadata_info_tab_contents
    """
    n = x.shape[0]
    
    kappa = sample_skewness(x) 
    std_kappa = standard_error_of_skewness(n)
    
    return kappa/std_kappa
    
    
    
def generate_scenarios(scenarios = 1000, n_size = 1000): 
    """
        A function for understanding what a p value is. 
    """

    p_values = np.zeros(scenarios, dtype="float64")
    skewnesses = np.zeros(scenarios, dtype="float64")
    kurtosis = np.zeros(scenarios, dtype="float64")
    gills = np.zeros_like(p_values)
    
    # for each scenario...
    for i in range(scenarios):
            
        # generate data. 
        x_i = np.random.normal(loc = 0.0, scale = 1.0, size = n_size) 
        # x_i = np.exp(x_i) - 1
        
        mu_i = sample_mean(x_i)
        sigma_i = sample_std(x_i)            
        
        # we are going to use a t-statistic here. 
        t_statistic = (mu_i/ (sigma_i / np.sqrt(n_size)))
        
        # calculate the p value of said t-statistic. 
        p_value = stats.norm(0,1).cdf(t_statistic)
        p_values[i] = p_value

        skewnesses[i] = sample_skewness(x_i) 
        gills[i] = gill_statistic(x_i)
        # kurtosis[i] = sample_kurtosis(x_i)

    return p_values, skewnesses, gills 


def standard_error_of_skewness(n: int) -> float: 
    """
        See jones and Gill 1998 for specifics. 
    """    
    return np.sqrt(( 6 * n * ( n - 1 ) / ((n-2)*(n+1)*(n+3))))

def standard_error_of_kurtosis(n: int )-> float: 
    """
        See Cramer 1979 for specifics. 
    """
    pass 
    
if __name__ == "__main__":

    print("Lecture 1")
    raw_clay_data_path = "/home/nik/gits/sta-datascience/private/src/lectures/data/SH_CLAY_11_4051.csv"    
    
    # load datasets. 
    df = pd.read_csv(raw_clay_data_path)
    
    # lets grab the first 5
    df.head() 
    # what do you notice about this data?
    # MISSING VALUES . 
    # how to grab them? df.isna() 
    missing_ij = df.isna() 
    
    # do all columns have missing values? 
    # df.isna().any() 

    # how to drop nas 
    df_no_nas = df.dropna() 
        
    # number of observations. df.shape[0]
    # number of observations with missing values? 
    
    # select only the relevant columns
    
    df_relevant = df[df.columns[0:5]]
   
    # lets drop the nans from this one. 
    df_clean = df_relevant.dropna()
    
    # lets replace the column names so they are easier to parse. 
    df_clean.columns = ['site_id','depth', 'll', 'pi', 'li']
    
    # recomended reading : stats without tears, 
    # anything by peter westfall. 
    
    # id like to begin with the normal distribution. 
    
    n = 1000
    x = np.random.normal(loc=0, scale=1.0, size = n)
    # plt.hist(x, bins=20)
    # plt.show() 
    # plt.clf() 
    
    mu = sample_mean(x) 
    sigma = sample_std(x) 
    print(f"Sample mu: {mu}, Sample sigma {sigma} ")    
    
    # Explaining what p-values are. 
    # suppose we want to test whether our population mean is approx == 0.0 
    # we can use a hypothesis test. 

    #null : mu = 0,
    #alt : mu != 0


    num_scenarios = 3000
    num_samples = 1000
    p_values, kappas, gills = generate_scenarios(num_scenarios, num_samples)
    rej_number = (p_values > 0.95).sum()
    rej_number_gills = (np.abs(gills) > 2).sum() 

    percentage = np.round(100 / num_scenarios, 3) 

    print(f"Number of p_values we reject {rej_number} | Proportion: {rej_number * percentage}%")
    print(f"Number of Gills show skewness {rej_number_gills} | Proportion: {rej_number_gills * percentage}%")

    # plt.hist(p_values, bins=25)
    # plt.show() 
    # plt.clf() 
    
    
    
    
    
    
    # import pdb; pdb.set_trace() 
    
    