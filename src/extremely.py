#importing all the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skextremes as sk
import scipy.stats as stats
import math as mt
from evt.methods.block_maxima import BlockMaxima
from evt.dataset import Dataset
from evt.methods.peaks_over_threshold import PeaksOverThreshold
from evt.estimators.gpdmle import GPDMLE

#Block Maxima method for finding large values. 
def getBM(sample,period):
    """
    In Block Maxima method we divide the whole dataset into blocks and select the largest value in each
    block as an extreme value.

    Parameters
    ----------
    sample : pandas dataframe 
        The whole dataset
    period : string
        The time period on basis of which the blocks are created. Eg - yearly, monthly, weekly and daily.

    Returns
    -------
    maxima_reset : pandas dataframe
        Maxima values obtained

    """
    
    #Obtain the maximas
    colname=list(sample)
    sample.iloc[:,0]= pd.to_datetime(sample.iloc[:,0])
    maxima = sample.resample(period, on=colname[0]).max()
    maxima_reset=maxima.reset_index(drop=True)
    series=pd.Series(sample.iloc[:,1])
    series.index.name="index"
    dataset = Dataset(series)
    N_SAMPLES_PER_BLOCK = round(len(sample)/len(maxima_reset))
    block_maxima = BlockMaxima(dataset, N_SAMPLES_PER_BLOCK)
    
    #Plot the maximas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    block_maxima.plot_block_maxima(ax1)
    block_maxima.plot_block_maxima_boxplot(ax2)
    fig.tight_layout()
    plt.show()
    
    #Return the maximas
    return maxima_reset

#Fit large claims obtained from BM method to Generalized Extreme Value (GEV) Distribution.
def gevfit(sample,fit_method="mle",ci=0,ci_method="delta"):
    """
    GEV is a limit distribution of properly normalized maxima of sequence of independent and identically 
    distributed random variables. It is parameterized by scale, shape and location parameters.

    Parameters
    ----------
    sample : pandas dataframe
        maximas obtained from Block Maxima method
    fit_method : string
        Estimation method like Maximum Likelihood Estimation or Method of Moments. Default is MLE.
    ci : Float
        Confidence interval. Default is 0.
    ci_method : string
        Method used for Confidence Interval like Delta or Bootstrap. Default is Delta.
        

    Returns
    -------
    model : Object
        Object containing the information about GEV fit.

    """
    #Using classic model of skextreme for GEV fitting
    model = sk.models.classic.GEV(sample.iloc[:,1], fit_method = fit_method, ci = ci,ci_method=ci_method)
    return model

#Get estimated distribution parameters of GEV fit.
def gevparams(model):
    """
    Accesing estimated distribution parameters from the GEV fit.

    Parameters
    ----------
    model : object
        Object containing the information about GEV fit.

    Returns
    -------
    OrderedDict
        Returns estimated distribution parameters.

    """
    return model.params

#Plots summarizing the GEV fit.    
def gevsummary(model):
    """
    Plotting plots like QQplot, PPplot, Return Level plot and density plot for the GEV model.

    Parameters
    ----------
    model : Object
        Object containing the information about GEV fit.

    Returns
    -------
    None

    """
    model.plot_summary()
 
#Mean Residual Life Plot for finding appropriate threshold value.
def MRL(sample, alpha=0.05):
    """
    Mean Residual Life plot takes mean of excess values above a threshold minus threshold and plots it against
    that threshold value. If the plot is linear that is ok but if the plot starts loosing stability then choose
    that threshold value

    Parameters
    ----------
    sample : pandas dataframe
        the whole dataset
    alpha : float
        a number giving 1-alpha confidence levels to use. Default value is 0.05.

    Returns
    -------
    Mean Residual Life Plot.

    """
    
    #Defining the threshold array and its step
    step = np.quantile(sample.iloc[:,1], .995)/60
    threshold = np.arange(0, max(sample.iloc[:,1]), step=step) 
    z_inverse = stats.norm.ppf(1-(alpha/2))

    #Initialization of arrays
    mrl_array = [] 
    CImrl = [] 

    #First Loop for getting the mean residual life for each threshold value and the 
    #second one getting the confidence intervals for the plot
    for u in threshold:
        excess = []
        for data in sample.iloc[:,1]:
            if data > u:
                excess.append(data - u) 
        mrl_array.append(np.mean(excess)) #
        std_loop = np.std(excess) 
        CImrl.append(z_inverse*std_loop/(len(excess)**0.5)) 

    CI_Low = [] 
    CI_High = [] 

    #Loop to add in the confidence interval to the plot arrays
    for i in range(0, len(mrl_array)):
        CI_Low.append(mrl_array[i] - CImrl[i])
        CI_High.append(mrl_array[i] + CImrl[i])

    #Plot MRL
    plt.figure(figsize=(7,7))
    sns.lineplot(x = threshold, y = mrl_array)
    plt.fill_between(threshold, CI_Low, CI_High, alpha = 0.4)
    plt.xlabel('Threshold')
    plt.ylabel('Mean Excesses')
    plt.title('Mean Residual Life Plot')
    plt.show()

#Peaks-Over-Threshold method for finding large values.    
def getPOT(sample,threshold):
    """
    In Peak-Over-Threshold method the values greater than a given threshold are taken as extreme values.

    Parameters
    ----------
    sample : pandas dataframe
        The whole Dataset
    threshold : integer
        An integer value above which the values are taken as extreme values.

    Returns
    -------
    exce : pandas dataframe
        Excess values obtained.

    """
    colnames=list(sample)
    exce=sample[sample.iloc[:,1].gt(threshold)]
    
    #Plotting the excess values obtained using POT method.
    ax = sample.iloc[:,1].reset_index().plot(kind='scatter', x='index', y=colnames[1],
                                           color='Red', label='Below Threshold')
    exce.reset_index().plot(kind='scatter', x='index', y=colnames[1],
                                          color='Blue', label='Above threshold', ax=ax)
    return exce

#Fit the large claims obtained from POT method to Generalized Pareto distribution (GPD).
def gpdfit(sample, threshold):
    """
    GPD is a family of continous probability distributions and is often used to model the tails of
    another distribution. It is specified by two parameters - Shape and scale parameters.

    Parameters
    ----------
    sample : pandas dataframe
        The whole Dataset
    threshold : integer
        An integer value above which the values are taken as extreme values.

    Returns
    -------
    Estimated distribution parametrs of GPD fit, sample excess values and value above the threshold.

    """
    sample = np.sort(sample.iloc[:,1])  
    sample_excess = []
    sample_over_thresh = []
    for data in sample:
        if data > threshold+0.00001:
            sample_excess.append(data - threshold) 
            sample_over_thresh.append(data) 
    series=pd.Series(sample)
    series.index.name="index"
    dataset = Dataset(series)
    
    #Using PeaksOverThreshold function from evt library.
    pot = PeaksOverThreshold(dataset, threshold)
    mle = GPDMLE(pot)
    shape_estimate, scale_estimate= mle.estimate()
    shape=getattr(shape_estimate, 'estimate')
    scale=getattr(scale_estimate, 'estimate') 
    return(shape, scale,sample, sample_excess, sample_over_thresh,mle)

#Get estimated distribution parameters for GPD fit.
def gpdparams(fit):
    """
    Getting estimated distribution parameters for GPD fit.

    Parameters
    ----------
    fit : Object
        Object containing information about GPD fit.

    Returns
    -------
    None.

    """
    shape=fit[0]
    scale=fit[1]
    print("Shape:",shape)
    print("Scale:",scale)
    
    
#get PDF plot with histogram to diagnostic the model
def gpdpdf(sample, threshold, bin_method, alpha):
    """
    Probability density plots are used to understand data distribution for a continuous variable and we want 
    to know the likelihood (or probability) of obtaining a range of values that the continuous variable can 
    assume.

    Parameters
    ----------
    sample : pandas dataframe
        The whole Dataset
    threshold : integer
        An integer value above which the values are taken as extreme values.
    bin_method : string
        Binning algorithm, specified as one of the following - auto, scott, fd, sturges, integers and sqrt.
    alpha : float
        a number giving 1-alpha confidence levels to use. Default value is 0.05.

    Returns
    -------
    Probability Density Function plot.

    """
    [shape, scale, sample, sample_excess, sample_over_thresh,mle] = gpdfit(sample, threshold) 
    x_points = np.arange(0, max(sample), 0.001) 
    pdf = stats.genpareto.pdf(x_points, shape, loc=0, scale=scale) 

    #Plotting PDF
    plt.figure(figsize=(7,7))
    plt.xlabel('Data')
    plt.ylabel('PDF')
    plt.title('Data Probability Density Function')
    plt.plot(x_points, pdf, color = 'black', label = 'Theoretical PDF')
    plt.hist(sample_excess, bins = bin_method, density = True)    
    plt.legend()
    plt.show()
    
    
#plot gpd cdf with empirical points
def gpdcdf(sample, threshold, alpha): 
    """
    The cumulative distribution function of a real-valued random variable X, or just distribution function 
    of X, evaluated at x, is the probability that X will take a value less than or equal to x.

    Parameters
    ----------
    sample : pandas dataframe
        The whole Dataset
    threshold : integer
        An integer value above which the values are taken as extreme values.
     alpha : float
         a number giving 1-alpha confidence levels to use. Default value is 0.05.

    Returns
    -------
    Cumulative Distribution Function plot.

    """
    [shape, scale, sample, sample_excess, sample_over_thresh,mle] = gpdfit(sample, threshold)
    n = len(sample_over_thresh)
    y = np.arange(1,n+1)/n 

    i_initial = 0
    n = len(sample)
    for i in range(0,n):
        if sample[i] > threshold: 
            i_initial = i 
            break
    
    #Computing confidence interval with the Dvoretzky–Kiefer–Wolfowitz method based on the empirical points
    F1 = []
    F2 = []
    for i in range(i_initial,len(sample)):
        e = (((mt.log(2/alpha))/(2*len(sample_over_thresh)))**0.5)  
        F1.append(y[i-i_initial] - e)
        F2.append(y[i-i_initial] + e)  

    x_points = np.arange(0, max(sample), 0.001) 
    cdf = stats.genpareto.cdf(x_points, shape, loc=threshold, scale=scale) #getting theoretical cdf
    
    #Plotting cdf 
    plt.figure(figsize=(7,7))
    plt.plot(x_points, cdf, color = 'black', label='Theoretical CDF')
    plt.xlabel('Data')
    plt.ylabel('CDF')
    plt.title('Data Comulative Distribution Function')
    plt.scatter(sorted(sample_over_thresh), y, label='Empirical CDF')
    plt.plot(sorted(sample_over_thresh), F1, linestyle='--', color='red', alpha = 0.8, lw = 0.9, label = 'Dvoretzky–Kiefer–Wolfowitz Confidence Bands')
    plt.plot(sorted(sample_over_thresh), F2, linestyle='--', color='red', alpha = 0.8, lw = 0.9)
    plt.legend()
    plt.show()

#Plot QQplot with empirical points.
def gpdqqplot(mle): 
    """
    QQplot is a graphical technique for determining if two datasets come from populations with a common
    distribution. A 45 degree reference line is plotted, if the two sets come from populations with the same 
    distribution, the points should fall approximately along this reference line.

    Parameters
    ----------
    mle : Object
        MLE estimator object from evt library.

    Returns
    -------
    Quantile-Quantile plot.

    """
    fig, ax = plt.subplots()
    mle.plot_qq_gpd(ax)
    fig.tight_layout()
    plt.show()
    
#probability-probability plot to diagnostic the model
def gpdppplot(sample, threshold, alpha): 
    """
    PP-plot is used for assessing how closely two datasets agree. It plots the two cumulative distribution
    functions against each other.

    Parameters
    ----------
    sample : pandas dataframe
        The whole Dataset
    threshold : integer
        An integer value above which the values are taken as extreme values.
     alpha : float
         a number giving 1-alpha confidence levels to use. Default value is 0.05.

    Returns
    -------
    Probability-Probability plot.

    """
    [shape, scale, sample, sample_excess, sample_over_thresh,mle] = gpdfit(sample, threshold) 
    n = len(sample_over_thresh)
    y = np.arange(1,n+1)/n  
    cdf_pp = stats.genpareto.cdf(sample_over_thresh, shape, loc=threshold, scale=scale)
    
    #Getting Confidence Intervals using the Dvoretzky–Kiefer–Wolfowitz method
    i_initial = 0
    n = len(sample)
    for i in range(0, n):
        if sample[i] > threshold + 0.0001:
            i_initial = i
            break
    F1 = []
    F2 = []
    for i in range(i_initial,n):
        e = (((mt.log(2/alpha))/(2*len(sample_over_thresh)))**0.5)  
        F1.append(y[i-i_initial] - e)
        F2.append(y[i-i_initial] + e)

    #Plotting PP
    plt.figure(figsize=(7,7))
    sns.regplot(y, cdf_pp, ci = None, line_kws={'color':'black', 'label':'Regression Line'})
    plt.plot(y, F1, linestyle='--', color='red', alpha = 0.5, lw = 0.8, label = 'Dvoretzky–Kiefer–Wolfowitz Confidence Bands')
    plt.plot(y, F2, linestyle='--', color='red', alpha = 0.5, lw = 0.8)
    plt.legend()
    plt.title('P-P Plot')
    plt.xlabel('Empirical Probability')
    plt.ylabel('Theoritical Probability')
    plt.show()
    

#Plot the survival function, (1 - cdf)
def survival_function(sample, threshold, alpha): 
    """
    The survival function is a function that gives the probability that the object of interest will 
    survive past a certain time.

    Parameters
    ----------
    sample : pandas dataframe
        The whole Dataset
    threshold : integer
        An integer value above which the values are taken as extreme values.
     alpha : float
         a number giving 1-alpha confidence levels to use. Default value is 0.05.

    Returns
    -------
    Survival function

    """
    [shape, scale, sample, sample_excess, sample_over_thresh,mle] = gpdfit(sample, threshold)

    n = len(sample_over_thresh)
    y_surv = 1 - np.arange(1,n+1)/n

    i_initial = 0

    n = len(sample)
    for i in range(0, n):
        if sample[i] > threshold + 0.0001:
            i_initial = i 
            break
    #Computing confidence interval with the Dvoretzky–Kiefer–Wolfowitz
    F1 = []
    F2 = []
    for i in range(i_initial,len(sample)):
        e =  (((mt.log(2/alpha))/(2*len(sample_over_thresh)))**0.5)  
        F1.append(y_surv[i-i_initial] - e)
        F2.append(y_surv[i-i_initial] + e)  

    x_points = np.arange(0, max(sample), 0.001)
    surv_func = 1 - stats.genpareto.cdf(x_points, shape, loc=threshold, scale=scale)

    #Plotting survival function
    plt.figure(9)
    plt.plot(x_points, surv_func, color = 'black', label='Theoretical Survival Function')
    plt.xlabel('Data')
    plt.ylabel('Survival Function')
    plt.title('Data Survival Function Plot')
    plt.scatter(sorted(sample_over_thresh), y_surv, label='Empirical Survival Function')
    plt.plot(sorted(sample_over_thresh), F1, linestyle='--', color='red', alpha = 0.8, lw = 0.9, label = 'Dvoretzky–Kiefer–Wolfowitz Confidence Bands')
    plt.plot(sorted(sample_over_thresh), F2, linestyle='--', color='red', alpha = 0.8, lw = 0.9)
    plt.legend()
    plt.show()
    
