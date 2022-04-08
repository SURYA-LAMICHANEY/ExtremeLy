<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, minimum-scale=1" />
<meta name="generator" content="pdoc 0.10.0" />
<title>extremely API documentation</title>
<meta name="description" content="ExtremeLy is a python package for Extreme Value Analysis. It was found that there are not many packages
for EVA in python. Among existing packages …" />
<link rel="preload stylesheet" as="style" href="https://cdnjs.cloudflare.com/ajax/libs/10up-sanitize.css/11.0.1/sanitize.min.css" integrity="sha256-PK9q560IAAa6WVRRh76LtCaI8pjTJ2z11v0miyNNjrs=" crossorigin>
<link rel="preload stylesheet" as="style" href="https://cdnjs.cloudflare.com/ajax/libs/10up-sanitize.css/11.0.1/typography.min.css" integrity="sha256-7l/o7C8jubJiy74VsKTidCy1yBkRtiUGbVkYBylBqUg=" crossorigin>
<link rel="stylesheet preload" as="style" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.1.1/styles/github.min.css" crossorigin>
<style>:root{--highlight-color:#fe9}.flex{display:flex !important}body{line-height:1.5em}#content{padding:20px}#sidebar{padding:30px;overflow:hidden}#sidebar > *:last-child{margin-bottom:2cm}.http-server-breadcrumbs{font-size:130%;margin:0 0 15px 0}#footer{font-size:.75em;padding:5px 30px;border-top:1px solid #ddd;text-align:right}#footer p{margin:0 0 0 1em;display:inline-block}#footer p:last-child{margin-right:30px}h1,h2,h3,h4,h5{font-weight:300}h1{font-size:2.5em;line-height:1.1em}h2{font-size:1.75em;margin:1em 0 .50em 0}h3{font-size:1.4em;margin:25px 0 10px 0}h4{margin:0;font-size:105%}h1:target,h2:target,h3:target,h4:target,h5:target,h6:target{background:var(--highlight-color);padding:.2em 0}a{color:#058;text-decoration:none;transition:color .3s ease-in-out}a:hover{color:#e82}.title code{font-weight:bold}h2[id^="header-"]{margin-top:2em}.ident{color:#900}pre code{background:#f8f8f8;font-size:.8em;line-height:1.4em}code{background:#f2f2f1;padding:1px 4px;overflow-wrap:break-word}h1 code{background:transparent}pre{background:#f8f8f8;border:0;border-top:1px solid #ccc;border-bottom:1px solid #ccc;margin:1em 0;padding:1ex}#http-server-module-list{display:flex;flex-flow:column}#http-server-module-list div{display:flex}#http-server-module-list dt{min-width:10%}#http-server-module-list p{margin-top:0}.toc ul,#index{list-style-type:none;margin:0;padding:0}#index code{background:transparent}#index h3{border-bottom:1px solid #ddd}#index ul{padding:0}#index h4{margin-top:.6em;font-weight:bold}@media (min-width:200ex){#index .two-column{column-count:2}}@media (min-width:300ex){#index .two-column{column-count:3}}dl{margin-bottom:2em}dl dl:last-child{margin-bottom:4em}dd{margin:0 0 1em 3em}#header-classes + dl > dd{margin-bottom:3em}dd dd{margin-left:2em}dd p{margin:10px 0}.name{background:#eee;font-weight:bold;font-size:.85em;padding:5px 10px;display:inline-block;min-width:40%}.name:hover{background:#e0e0e0}dt:target .name{background:var(--highlight-color)}.name > span:first-child{white-space:nowrap}.name.class > span:nth-child(2){margin-left:.4em}.inherited{color:#999;border-left:5px solid #eee;padding-left:1em}.inheritance em{font-style:normal;font-weight:bold}.desc h2{font-weight:400;font-size:1.25em}.desc h3{font-size:1em}.desc dt code{background:inherit}.source summary,.git-link-div{color:#666;text-align:right;font-weight:400;font-size:.8em;text-transform:uppercase}.source summary > *{white-space:nowrap;cursor:pointer}.git-link{color:inherit;margin-left:1em}.source pre{max-height:500px;overflow:auto;margin:0}.source pre code{font-size:12px;overflow:visible}.hlist{list-style:none}.hlist li{display:inline}.hlist li:after{content:',\2002'}.hlist li:last-child:after{content:none}.hlist .hlist{display:inline;padding-left:1em}img{max-width:100%}td{padding:0 .5em}.admonition{padding:.1em .5em;margin-bottom:1em}.admonition-title{font-weight:bold}.admonition.note,.admonition.info,.admonition.important{background:#aef}.admonition.todo,.admonition.versionadded,.admonition.tip,.admonition.hint{background:#dfd}.admonition.warning,.admonition.versionchanged,.admonition.deprecated{background:#fd4}.admonition.error,.admonition.danger,.admonition.caution{background:lightpink}</style>
<style media="screen and (min-width: 700px)">@media screen and (min-width:700px){#sidebar{width:30%;height:100vh;overflow:auto;position:sticky;top:0}#content{width:70%;max-width:100ch;padding:3em 4em;border-left:1px solid #ddd}pre code{font-size:1em}.item .name{font-size:1em}main{display:flex;flex-direction:row-reverse;justify-content:flex-end}.toc ul ul,#index ul{padding-left:1.5em}.toc > ul > li{margin-top:.5em}}</style>
<style media="print">@media print{#sidebar h1{page-break-before:always}.source{display:none}}@media print{*{background:transparent !important;color:#000 !important;box-shadow:none !important;text-shadow:none !important}a[href]:after{content:" (" attr(href) ")";font-size:90%}a[href][title]:after{content:none}abbr[title]:after{content:" (" attr(title) ")"}.ir a:after,a[href^="javascript:"]:after,a[href^="#"]:after{content:""}pre,blockquote{border:1px solid #999;page-break-inside:avoid}thead{display:table-header-group}tr,img{page-break-inside:avoid}img{max-width:100% !important}@page{margin:0.5cm}p,h2,h3{orphans:3;widows:3}h1,h2,h3,h4,h5,h6{page-break-after:avoid}}</style>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.1.1/highlight.min.js" integrity="sha256-Uv3H6lx7dJmRfRvH8TH6kJD1TSK1aFcwgx+mdg3epi8=" crossorigin></script>
<script>window.addEventListener('DOMContentLoaded', () => hljs.initHighlighting())</script>
</head>
<body>
<main>
<article id="content">
<header>
<h1 class="title">Module <code>extremely</code></h1>
</header>
<section id="section-intro">
<p>ExtremeLy is a python package for Extreme Value Analysis. It was found that there are not many packages
for EVA in python. Among existing packages some of them were incomplete, some of them were internally using R
packages and some had only basic implementations without any plots for model assessment. So ExtremeLy brings all
those packages together, removes R dependencies and provides most of the fucntionalities for EVA in python
without being dependent on R packages. Some fucntionalities from the already existing packages have been used
as they are, some have been modified to accomodate additional requirements and for some just the R dependencies
are replaced with python implementation. The three already existing packages that are used here are:
1. scikit-extremes (skextremes) - <a href="https://scikit-extremes.readthedocs.io/en/latest/">https://scikit-extremes.readthedocs.io/en/latest/</a>
2. thresholdmodeling - <a href="https://github.com/iagolemos1/thresholdmodeling">https://github.com/iagolemos1/thresholdmodeling</a>
3. evt - <a href="https://pypi.org/project/evt/#description">https://pypi.org/project/evt/#description</a></p>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">###########################################################################################################
&#34;&#34;&#34;
ExtremeLy is a python package for Extreme Value Analysis. It was found that there are not many packages
for EVA in python. Among existing packages some of them were incomplete, some of them were internally using R 
packages and some had only basic implementations without any plots for model assessment. So ExtremeLy brings all 
those packages together, removes R dependencies and provides most of the fucntionalities for EVA in python
without being dependent on R packages. Some fucntionalities from the already existing packages have been used
as they are, some have been modified to accomodate additional requirements and for some just the R dependencies
are replaced with python implementation. The three already existing packages that are used here are:
    1. scikit-extremes (skextremes) - https://scikit-extremes.readthedocs.io/en/latest/
    2. thresholdmodeling - https://github.com/iagolemos1/thresholdmodeling
    3. evt - https://pypi.org/project/evt/#description
&#34;&#34;&#34;
############################################################################################################
&#34;&#34;&#34;
There are basically two approaches to Extreme Value Analysis:
    1. Block maxima method + Generalized Extreme Value (GEV) Distribution.
    2. Peak-over-threshold method + Generalized Pareto Distribution (GPD).
&#34;&#34;&#34;
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
    &#34;&#34;&#34;
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

    &#34;&#34;&#34;
    
    #Obtain the maximas
    colname=list(sample)
    sample.iloc[:,0]= pd.to_datetime(sample.iloc[:,0])
    maxima = sample.resample(period, on=colname[0]).max()
    maxima_reset=maxima.reset_index(drop=True)
    series=pd.Series(sample.iloc[:,1])
    series.index.name=&#34;index&#34;
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
def gevfit(sample,fit_method=&#34;mle&#34;,ci=0,ci_method=&#34;delta&#34;):
    &#34;&#34;&#34;
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

    &#34;&#34;&#34;
    #Using classic model of skextreme for GEV fitting
    model = sk.models.classic.GEV(sample.iloc[:,1], fit_method = fit_method, ci = ci,ci_method=ci_method)
    return model

#Get estimated distribution parameters of GEV fit.
def gevparams(model):
    &#34;&#34;&#34;
    Accesing estimated distribution parameters from the GEV fit.

    Parameters
    ----------
    model : object
        Object containing the information about GEV fit.

    Returns
    -------
    OrderedDict
        Returns estimated distribution parameters.

    &#34;&#34;&#34;
    return model.params

#Plots summarizing the GEV fit.    
def gevsummary(model):
    &#34;&#34;&#34;
    Plotting plots like QQplot, PPplot, Return Level plot and density plot for the GEV model.

    Parameters
    ----------
    model : Object
        Object containing the information about GEV fit.

    Returns
    -------
    None

    &#34;&#34;&#34;
    model.plot_summary()
 
#Value-at-Risk for large claims.
def gevVaR(sample):
    &#34;&#34;&#34;
    Calculates Value-at-Risk at 99% confidence interval for the maximas.

    Parameters
    ----------
    sample : pandas dataframe
        maximas obtained from Block Maxima method.

    Returns
    -------
    Value-at-Risk(VaR)

    &#34;&#34;&#34;
    #Calculate VaR at 99% cnfidence interval
    params = stats.genextreme.fit(sample.iloc[:,1])
    VaR_99 = stats.genextreme.ppf(0.99, *params)
    return VaR_99

#Mean Residual Life Plot for finding appropriate threshold value.
def MRL(sample, alpha=0.05):
    &#34;&#34;&#34;
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

    &#34;&#34;&#34;
    
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
            if data &gt; u:
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
    plt.xlabel(&#39;Threshold&#39;)
    plt.ylabel(&#39;Mean Excesses&#39;)
    plt.title(&#39;Mean Residual Life Plot&#39;)
    plt.show()

#Peaks-Over-Threshold method for finding large values.    
def getPOT(sample,threshold):
    &#34;&#34;&#34;
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

    &#34;&#34;&#34;
    colnames=list(sample)
    exce=sample[sample.iloc[:,1].gt(threshold)]
    
    #Plotting the excess values obtained using POT method.
    ax = sample.iloc[:,1].reset_index().plot(kind=&#39;scatter&#39;, x=&#39;index&#39;, y=colnames[1],
                                           color=&#39;Red&#39;, label=&#39;Below Threshold&#39;)
    exce.reset_index().plot(kind=&#39;scatter&#39;, x=&#39;index&#39;, y=colnames[1],
                                          color=&#39;Blue&#39;, label=&#39;Above threshold&#39;, ax=ax)
    return exce

#Fit the large claims obtained from POT method to Generalized Pareto distribution (GPD).
def gpdfit(sample, threshold):
    &#34;&#34;&#34;
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

    &#34;&#34;&#34;
    sample = np.sort(sample.iloc[:,1])  
    sample_excess = []
    sample_over_thresh = []
    for data in sample:
        if data &gt; threshold+0.00001:
            sample_excess.append(data - threshold) 
            sample_over_thresh.append(data) 
    series=pd.Series(sample)
    series.index.name=&#34;index&#34;
    dataset = Dataset(series)
    
    #Using PeaksOverThreshold function from evt library.
    pot = PeaksOverThreshold(dataset, threshold)
    mle = GPDMLE(pot)
    shape_estimate, scale_estimate= mle.estimate()
    shape=getattr(shape_estimate, &#39;estimate&#39;)
    scale=getattr(scale_estimate, &#39;estimate&#39;) 
    return(shape, scale,sample, sample_excess, sample_over_thresh,mle)

#Get estimated distribution parameters for GPD fit.
def gpdparams(fit):
    &#34;&#34;&#34;
    Getting estimated distribution parameters for GPD fit.

    Parameters
    ----------
    fit : Object
        Object containing information about GPD fit.

    Returns
    -------
    None.

    &#34;&#34;&#34;
    shape=fit[0]
    scale=fit[1]
    print(&#34;Shape:&#34;,shape)
    print(&#34;Scale:&#34;,scale)
    
    
#get PDF plot with histogram to diagnostic the model
def gpdpdf(sample, threshold, bin_method, alpha):
    &#34;&#34;&#34;
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

    &#34;&#34;&#34;
    [shape, scale, sample, sample_excess, sample_over_thresh,mle] = gpdfit(sample, threshold) 
    x_points = np.arange(0, max(sample), 0.001) 
    pdf = stats.genpareto.pdf(x_points, shape, loc=0, scale=scale) 

    #Plotting PDF
    plt.figure(figsize=(7,7))
    plt.xlabel(&#39;Data&#39;)
    plt.ylabel(&#39;PDF&#39;)
    plt.title(&#39;Data Probability Density Function&#39;)
    plt.plot(x_points, pdf, color = &#39;black&#39;, label = &#39;Theoretical PDF&#39;)
    plt.hist(sample_excess, bins = bin_method, density = True)    
    plt.legend()
    plt.show()
    
    
#plot gpd cdf with empirical points
def gpdcdf(sample, threshold, alpha): 
    &#34;&#34;&#34;
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

    &#34;&#34;&#34;
    [shape, scale, sample, sample_excess, sample_over_thresh,mle] = gpdfit(sample, threshold)
    n = len(sample_over_thresh)
    y = np.arange(1,n+1)/n 

    i_initial = 0
    n = len(sample)
    for i in range(0,n):
        if sample[i] &gt; threshold: 
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
    plt.plot(x_points, cdf, color = &#39;black&#39;, label=&#39;Theoretical CDF&#39;)
    plt.xlabel(&#39;Data&#39;)
    plt.ylabel(&#39;CDF&#39;)
    plt.title(&#39;Data Comulative Distribution Function&#39;)
    plt.scatter(sorted(sample_over_thresh), y, label=&#39;Empirical CDF&#39;)
    plt.plot(sorted(sample_over_thresh), F1, linestyle=&#39;--&#39;, color=&#39;red&#39;, alpha = 0.8, lw = 0.9, label = &#39;Dvoretzky–Kiefer–Wolfowitz Confidence Bands&#39;)
    plt.plot(sorted(sample_over_thresh), F2, linestyle=&#39;--&#39;, color=&#39;red&#39;, alpha = 0.8, lw = 0.9)
    plt.legend()
    plt.show()

#Plot QQplot with empirical points.
def gpdqqplot(mle): 
    &#34;&#34;&#34;
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

    &#34;&#34;&#34;
    fig, ax = plt.subplots()
    mle.plot_qq_gpd(ax)
    fig.tight_layout()
    plt.show()
    
#probability-probability plot to diagnostic the model
def gpdppplot(sample, threshold, alpha): 
    &#34;&#34;&#34;
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

    &#34;&#34;&#34;
    [shape, scale, sample, sample_excess, sample_over_thresh,mle] = gpdfit(sample, threshold) 
    n = len(sample_over_thresh)
    y = np.arange(1,n+1)/n  
    cdf_pp = stats.genpareto.cdf(sample_over_thresh, shape, loc=threshold, scale=scale)
    
    #Getting Confidence Intervals using the Dvoretzky–Kiefer–Wolfowitz method
    i_initial = 0
    n = len(sample)
    for i in range(0, n):
        if sample[i] &gt; threshold + 0.0001:
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
    sns.regplot(y, cdf_pp, ci = None, line_kws={&#39;color&#39;:&#39;black&#39;, &#39;label&#39;:&#39;Regression Line&#39;})
    plt.plot(y, F1, linestyle=&#39;--&#39;, color=&#39;red&#39;, alpha = 0.5, lw = 0.8, label = &#39;Dvoretzky–Kiefer–Wolfowitz Confidence Bands&#39;)
    plt.plot(y, F2, linestyle=&#39;--&#39;, color=&#39;red&#39;, alpha = 0.5, lw = 0.8)
    plt.legend()
    plt.title(&#39;P-P Plot&#39;)
    plt.xlabel(&#39;Empirical Probability&#39;)
    plt.ylabel(&#39;Theoritical Probability&#39;)
    plt.show()
    

#Plot the survival function, (1 - cdf)
def survival_function(sample, threshold, alpha): 
    &#34;&#34;&#34;
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

    &#34;&#34;&#34;
    [shape, scale, sample, sample_excess, sample_over_thresh,mle] = gpdfit(sample, threshold)

    n = len(sample_over_thresh)
    y_surv = 1 - np.arange(1,n+1)/n

    i_initial = 0

    n = len(sample)
    for i in range(0, n):
        if sample[i] &gt; threshold + 0.0001:
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
    plt.plot(x_points, surv_func, color = &#39;black&#39;, label=&#39;Theoretical Survival Function&#39;)
    plt.xlabel(&#39;Data&#39;)
    plt.ylabel(&#39;Survival Function&#39;)
    plt.title(&#39;Data Survival Function Plot&#39;)
    plt.scatter(sorted(sample_over_thresh), y_surv, label=&#39;Empirical Survival Function&#39;)
    plt.plot(sorted(sample_over_thresh), F1, linestyle=&#39;--&#39;, color=&#39;red&#39;, alpha = 0.8, lw = 0.9, label = &#39;Dvoretzky–Kiefer–Wolfowitz Confidence Bands&#39;)
    plt.plot(sorted(sample_over_thresh), F2, linestyle=&#39;--&#39;, color=&#39;red&#39;, alpha = 0.8, lw = 0.9)
    plt.legend()
    plt.show()
    </code></pre>
</details>
</section>
<section>
</section>
<section>
</section>
<section>
<h2 class="section-title" id="header-functions">Functions</h2>
<dl>
<dt id="extremely.MRL"><code class="name flex">
<span>def <span class="ident">MRL</span></span>(<span>sample, alpha=0.05)</span>
</code></dt>
<dd>
<div class="desc"><p>Mean Residual Life plot takes mean of excess values above a threshold minus threshold and plots it against
that threshold value. If the plot is linear that is ok but if the plot starts loosing stability then choose
that threshold value</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>sample</code></strong> :&ensp;<code>pandas dataframe</code></dt>
<dd>the whole dataset</dd>
<dt><strong><code>alpha</code></strong> :&ensp;<code>float</code></dt>
<dd>a number giving 1-alpha confidence levels to use. Default value is 0.05.</dd>
</dl>
<h2 id="returns">Returns</h2>
<p>Mean Residual Life Plot.</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def MRL(sample, alpha=0.05):
    &#34;&#34;&#34;
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

    &#34;&#34;&#34;
    
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
            if data &gt; u:
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
    plt.xlabel(&#39;Threshold&#39;)
    plt.ylabel(&#39;Mean Excesses&#39;)
    plt.title(&#39;Mean Residual Life Plot&#39;)
    plt.show()</code></pre>
</details>
</dd>
<dt id="extremely.getBM"><code class="name flex">
<span>def <span class="ident">getBM</span></span>(<span>sample, period)</span>
</code></dt>
<dd>
<div class="desc"><p>In Block Maxima method we divide the whole dataset into blocks and select the largest value in each
block as an extreme value.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>sample</code></strong> :&ensp;<code>pandas dataframe </code></dt>
<dd>The whole dataset</dd>
<dt><strong><code>period</code></strong> :&ensp;<code>string</code></dt>
<dd>The time period on basis of which the blocks are created. Eg - yearly, monthly, weekly and daily.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>maxima_reset</code></strong> :&ensp;<code>pandas dataframe</code></dt>
<dd>Maxima values obtained</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def getBM(sample,period):
    &#34;&#34;&#34;
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

    &#34;&#34;&#34;
    
    #Obtain the maximas
    colname=list(sample)
    sample.iloc[:,0]= pd.to_datetime(sample.iloc[:,0])
    maxima = sample.resample(period, on=colname[0]).max()
    maxima_reset=maxima.reset_index(drop=True)
    series=pd.Series(sample.iloc[:,1])
    series.index.name=&#34;index&#34;
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
    return maxima_reset</code></pre>
</details>
</dd>
<dt id="extremely.getPOT"><code class="name flex">
<span>def <span class="ident">getPOT</span></span>(<span>sample, threshold)</span>
</code></dt>
<dd>
<div class="desc"><p>In Peak-Over-Threshold method the values greater than a given threshold are taken as extreme values.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>sample</code></strong> :&ensp;<code>pandas dataframe</code></dt>
<dd>The whole Dataset</dd>
<dt><strong><code>threshold</code></strong> :&ensp;<code>integer</code></dt>
<dd>An integer value above which the values are taken as extreme values.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>exce</code></strong> :&ensp;<code>pandas dataframe</code></dt>
<dd>Excess values obtained.</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def getPOT(sample,threshold):
    &#34;&#34;&#34;
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

    &#34;&#34;&#34;
    colnames=list(sample)
    exce=sample[sample.iloc[:,1].gt(threshold)]
    
    #Plotting the excess values obtained using POT method.
    ax = sample.iloc[:,1].reset_index().plot(kind=&#39;scatter&#39;, x=&#39;index&#39;, y=colnames[1],
                                           color=&#39;Red&#39;, label=&#39;Below Threshold&#39;)
    exce.reset_index().plot(kind=&#39;scatter&#39;, x=&#39;index&#39;, y=colnames[1],
                                          color=&#39;Blue&#39;, label=&#39;Above threshold&#39;, ax=ax)
    return exce</code></pre>
</details>
</dd>
<dt id="extremely.gevVaR"><code class="name flex">
<span>def <span class="ident">gevVaR</span></span>(<span>sample)</span>
</code></dt>
<dd>
<div class="desc"><p>Calculates Value-at-Risk at 99% confidence interval for the maximas.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>sample</code></strong> :&ensp;<code>pandas dataframe</code></dt>
<dd>maximas obtained from Block Maxima method.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>Value-at-Risk(VaR)</code></dt>
<dd>&nbsp;</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def gevVaR(sample):
    &#34;&#34;&#34;
    Calculates Value-at-Risk at 99% confidence interval for the maximas.

    Parameters
    ----------
    sample : pandas dataframe
        maximas obtained from Block Maxima method.

    Returns
    -------
    Value-at-Risk(VaR)

    &#34;&#34;&#34;
    #Calculate VaR at 99% cnfidence interval
    params = stats.genextreme.fit(sample.iloc[:,1])
    VaR_99 = stats.genextreme.ppf(0.99, *params)
    return VaR_99</code></pre>
</details>
</dd>
<dt id="extremely.gevfit"><code class="name flex">
<span>def <span class="ident">gevfit</span></span>(<span>sample, fit_method='mle', ci=0, ci_method='delta')</span>
</code></dt>
<dd>
<div class="desc"><p>GEV is a limit distribution of properly normalized maxima of sequence of independent and identically
distributed random variables. It is parameterized by scale, shape and location parameters.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>sample</code></strong> :&ensp;<code>pandas dataframe</code></dt>
<dd>maximas obtained from Block Maxima method</dd>
<dt><strong><code>fit_method</code></strong> :&ensp;<code>string</code></dt>
<dd>Estimation method like Maximum Likelihood Estimation or Method of Moments. Default is MLE.</dd>
<dt><strong><code>ci</code></strong> :&ensp;<code>Float</code></dt>
<dd>Confidence interval. Default is 0.</dd>
<dt><strong><code>ci_method</code></strong> :&ensp;<code>string</code></dt>
<dd>Method used for Confidence Interval like Delta or Bootstrap. Default is Delta.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>model</code></strong> :&ensp;<code>Object</code></dt>
<dd>Object containing the information about GEV fit.</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def gevfit(sample,fit_method=&#34;mle&#34;,ci=0,ci_method=&#34;delta&#34;):
    &#34;&#34;&#34;
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

    &#34;&#34;&#34;
    #Using classic model of skextreme for GEV fitting
    model = sk.models.classic.GEV(sample.iloc[:,1], fit_method = fit_method, ci = ci,ci_method=ci_method)
    return model</code></pre>
</details>
</dd>
<dt id="extremely.gevparams"><code class="name flex">
<span>def <span class="ident">gevparams</span></span>(<span>model)</span>
</code></dt>
<dd>
<div class="desc"><p>Accesing estimated distribution parameters from the GEV fit.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>model</code></strong> :&ensp;<code>object</code></dt>
<dd>Object containing the information about GEV fit.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>OrderedDict</code></dt>
<dd>Returns estimated distribution parameters.</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def gevparams(model):
    &#34;&#34;&#34;
    Accesing estimated distribution parameters from the GEV fit.

    Parameters
    ----------
    model : object
        Object containing the information about GEV fit.

    Returns
    -------
    OrderedDict
        Returns estimated distribution parameters.

    &#34;&#34;&#34;
    return model.params</code></pre>
</details>
</dd>
<dt id="extremely.gevsummary"><code class="name flex">
<span>def <span class="ident">gevsummary</span></span>(<span>model)</span>
</code></dt>
<dd>
<div class="desc"><p>Plotting plots like QQplot, PPplot, Return Level plot and density plot for the GEV model.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>model</code></strong> :&ensp;<code>Object</code></dt>
<dd>Object containing the information about GEV fit.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>None</code></dt>
<dd>&nbsp;</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def gevsummary(model):
    &#34;&#34;&#34;
    Plotting plots like QQplot, PPplot, Return Level plot and density plot for the GEV model.

    Parameters
    ----------
    model : Object
        Object containing the information about GEV fit.

    Returns
    -------
    None

    &#34;&#34;&#34;
    model.plot_summary()</code></pre>
</details>
</dd>
<dt id="extremely.gpdcdf"><code class="name flex">
<span>def <span class="ident">gpdcdf</span></span>(<span>sample, threshold, alpha)</span>
</code></dt>
<dd>
<div class="desc"><p>The cumulative distribution function of a real-valued random variable X, or just distribution function
of X, evaluated at x, is the probability that X will take a value less than or equal to x.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>sample</code></strong> :&ensp;<code>pandas dataframe</code></dt>
<dd>The whole Dataset</dd>
<dt><strong><code>threshold</code></strong> :&ensp;<code>integer</code></dt>
<dd>An integer value above which the values are taken as extreme values.</dd>
</dl>
<p>alpha : float
a number giving 1-alpha confidence levels to use. Default value is 0.05.</p>
<h2 id="returns">Returns</h2>
<p>Cumulative Distribution Function plot.</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def gpdcdf(sample, threshold, alpha): 
    &#34;&#34;&#34;
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

    &#34;&#34;&#34;
    [shape, scale, sample, sample_excess, sample_over_thresh,mle] = gpdfit(sample, threshold)
    n = len(sample_over_thresh)
    y = np.arange(1,n+1)/n 

    i_initial = 0
    n = len(sample)
    for i in range(0,n):
        if sample[i] &gt; threshold: 
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
    plt.plot(x_points, cdf, color = &#39;black&#39;, label=&#39;Theoretical CDF&#39;)
    plt.xlabel(&#39;Data&#39;)
    plt.ylabel(&#39;CDF&#39;)
    plt.title(&#39;Data Comulative Distribution Function&#39;)
    plt.scatter(sorted(sample_over_thresh), y, label=&#39;Empirical CDF&#39;)
    plt.plot(sorted(sample_over_thresh), F1, linestyle=&#39;--&#39;, color=&#39;red&#39;, alpha = 0.8, lw = 0.9, label = &#39;Dvoretzky–Kiefer–Wolfowitz Confidence Bands&#39;)
    plt.plot(sorted(sample_over_thresh), F2, linestyle=&#39;--&#39;, color=&#39;red&#39;, alpha = 0.8, lw = 0.9)
    plt.legend()
    plt.show()</code></pre>
</details>
</dd>
<dt id="extremely.gpdfit"><code class="name flex">
<span>def <span class="ident">gpdfit</span></span>(<span>sample, threshold)</span>
</code></dt>
<dd>
<div class="desc"><p>GPD is a family of continous probability distributions and is often used to model the tails of
another distribution. It is specified by two parameters - Shape and scale parameters.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>sample</code></strong> :&ensp;<code>pandas dataframe</code></dt>
<dd>The whole Dataset</dd>
<dt><strong><code>threshold</code></strong> :&ensp;<code>integer</code></dt>
<dd>An integer value above which the values are taken as extreme values.</dd>
</dl>
<h2 id="returns">Returns</h2>
<p>Estimated distribution parametrs of GPD fit, sample excess values and value above the threshold.</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def gpdfit(sample, threshold):
    &#34;&#34;&#34;
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

    &#34;&#34;&#34;
    sample = np.sort(sample.iloc[:,1])  
    sample_excess = []
    sample_over_thresh = []
    for data in sample:
        if data &gt; threshold+0.00001:
            sample_excess.append(data - threshold) 
            sample_over_thresh.append(data) 
    series=pd.Series(sample)
    series.index.name=&#34;index&#34;
    dataset = Dataset(series)
    
    #Using PeaksOverThreshold function from evt library.
    pot = PeaksOverThreshold(dataset, threshold)
    mle = GPDMLE(pot)
    shape_estimate, scale_estimate= mle.estimate()
    shape=getattr(shape_estimate, &#39;estimate&#39;)
    scale=getattr(scale_estimate, &#39;estimate&#39;) 
    return(shape, scale,sample, sample_excess, sample_over_thresh,mle)</code></pre>
</details>
</dd>
<dt id="extremely.gpdparams"><code class="name flex">
<span>def <span class="ident">gpdparams</span></span>(<span>fit)</span>
</code></dt>
<dd>
<div class="desc"><p>Getting estimated distribution parameters for GPD fit.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>fit</code></strong> :&ensp;<code>Object</code></dt>
<dd>Object containing information about GPD fit.</dd>
</dl>
<h2 id="returns">Returns</h2>
<p>None.</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def gpdparams(fit):
    &#34;&#34;&#34;
    Getting estimated distribution parameters for GPD fit.

    Parameters
    ----------
    fit : Object
        Object containing information about GPD fit.

    Returns
    -------
    None.

    &#34;&#34;&#34;
    shape=fit[0]
    scale=fit[1]
    print(&#34;Shape:&#34;,shape)
    print(&#34;Scale:&#34;,scale)</code></pre>
</details>
</dd>
<dt id="extremely.gpdpdf"><code class="name flex">
<span>def <span class="ident">gpdpdf</span></span>(<span>sample, threshold, bin_method, alpha)</span>
</code></dt>
<dd>
<div class="desc"><p>Probability density plots are used to understand data distribution for a continuous variable and we want
to know the likelihood (or probability) of obtaining a range of values that the continuous variable can
assume.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>sample</code></strong> :&ensp;<code>pandas dataframe</code></dt>
<dd>The whole Dataset</dd>
<dt><strong><code>threshold</code></strong> :&ensp;<code>integer</code></dt>
<dd>An integer value above which the values are taken as extreme values.</dd>
<dt><strong><code>bin_method</code></strong> :&ensp;<code>string</code></dt>
<dd>Binning algorithm, specified as one of the following - auto, scott, fd, sturges, integers and sqrt.</dd>
<dt><strong><code>alpha</code></strong> :&ensp;<code>float</code></dt>
<dd>a number giving 1-alpha confidence levels to use. Default value is 0.05.</dd>
</dl>
<h2 id="returns">Returns</h2>
<p>Probability Density Function plot.</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def gpdpdf(sample, threshold, bin_method, alpha):
    &#34;&#34;&#34;
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

    &#34;&#34;&#34;
    [shape, scale, sample, sample_excess, sample_over_thresh,mle] = gpdfit(sample, threshold) 
    x_points = np.arange(0, max(sample), 0.001) 
    pdf = stats.genpareto.pdf(x_points, shape, loc=0, scale=scale) 

    #Plotting PDF
    plt.figure(figsize=(7,7))
    plt.xlabel(&#39;Data&#39;)
    plt.ylabel(&#39;PDF&#39;)
    plt.title(&#39;Data Probability Density Function&#39;)
    plt.plot(x_points, pdf, color = &#39;black&#39;, label = &#39;Theoretical PDF&#39;)
    plt.hist(sample_excess, bins = bin_method, density = True)    
    plt.legend()
    plt.show()</code></pre>
</details>
</dd>
<dt id="extremely.gpdppplot"><code class="name flex">
<span>def <span class="ident">gpdppplot</span></span>(<span>sample, threshold, alpha)</span>
</code></dt>
<dd>
<div class="desc"><p>PP-plot is used for assessing how closely two datasets agree. It plots the two cumulative distribution
functions against each other.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>sample</code></strong> :&ensp;<code>pandas dataframe</code></dt>
<dd>The whole Dataset</dd>
<dt><strong><code>threshold</code></strong> :&ensp;<code>integer</code></dt>
<dd>An integer value above which the values are taken as extreme values.</dd>
</dl>
<p>alpha : float
a number giving 1-alpha confidence levels to use. Default value is 0.05.</p>
<h2 id="returns">Returns</h2>
<p>Probability-Probability plot.</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def gpdppplot(sample, threshold, alpha): 
    &#34;&#34;&#34;
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

    &#34;&#34;&#34;
    [shape, scale, sample, sample_excess, sample_over_thresh,mle] = gpdfit(sample, threshold) 
    n = len(sample_over_thresh)
    y = np.arange(1,n+1)/n  
    cdf_pp = stats.genpareto.cdf(sample_over_thresh, shape, loc=threshold, scale=scale)
    
    #Getting Confidence Intervals using the Dvoretzky–Kiefer–Wolfowitz method
    i_initial = 0
    n = len(sample)
    for i in range(0, n):
        if sample[i] &gt; threshold + 0.0001:
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
    sns.regplot(y, cdf_pp, ci = None, line_kws={&#39;color&#39;:&#39;black&#39;, &#39;label&#39;:&#39;Regression Line&#39;})
    plt.plot(y, F1, linestyle=&#39;--&#39;, color=&#39;red&#39;, alpha = 0.5, lw = 0.8, label = &#39;Dvoretzky–Kiefer–Wolfowitz Confidence Bands&#39;)
    plt.plot(y, F2, linestyle=&#39;--&#39;, color=&#39;red&#39;, alpha = 0.5, lw = 0.8)
    plt.legend()
    plt.title(&#39;P-P Plot&#39;)
    plt.xlabel(&#39;Empirical Probability&#39;)
    plt.ylabel(&#39;Theoritical Probability&#39;)
    plt.show()</code></pre>
</details>
</dd>
<dt id="extremely.gpdqqplot"><code class="name flex">
<span>def <span class="ident">gpdqqplot</span></span>(<span>mle)</span>
</code></dt>
<dd>
<div class="desc"><p>QQplot is a graphical technique for determining if two datasets come from populations with a common
distribution. A 45 degree reference line is plotted, if the two sets come from populations with the same
distribution, the points should fall approximately along this reference line.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>mle</code></strong> :&ensp;<code>Object</code></dt>
<dd>MLE estimator object from evt library.</dd>
</dl>
<h2 id="returns">Returns</h2>
<p>Quantile-Quantile plot.</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def gpdqqplot(mle): 
    &#34;&#34;&#34;
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

    &#34;&#34;&#34;
    fig, ax = plt.subplots()
    mle.plot_qq_gpd(ax)
    fig.tight_layout()
    plt.show()</code></pre>
</details>
</dd>
<dt id="extremely.survival_function"><code class="name flex">
<span>def <span class="ident">survival_function</span></span>(<span>sample, threshold, alpha)</span>
</code></dt>
<dd>
<div class="desc"><p>The survival function is a function that gives the probability that the object of interest will
survive past a certain time.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>sample</code></strong> :&ensp;<code>pandas dataframe</code></dt>
<dd>The whole Dataset</dd>
<dt><strong><code>threshold</code></strong> :&ensp;<code>integer</code></dt>
<dd>An integer value above which the values are taken as extreme values.</dd>
</dl>
<p>alpha : float
a number giving 1-alpha confidence levels to use. Default value is 0.05.</p>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>Survival function</code></dt>
<dd>&nbsp;</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def survival_function(sample, threshold, alpha): 
    &#34;&#34;&#34;
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

    &#34;&#34;&#34;
    [shape, scale, sample, sample_excess, sample_over_thresh,mle] = gpdfit(sample, threshold)

    n = len(sample_over_thresh)
    y_surv = 1 - np.arange(1,n+1)/n

    i_initial = 0

    n = len(sample)
    for i in range(0, n):
        if sample[i] &gt; threshold + 0.0001:
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
    plt.plot(x_points, surv_func, color = &#39;black&#39;, label=&#39;Theoretical Survival Function&#39;)
    plt.xlabel(&#39;Data&#39;)
    plt.ylabel(&#39;Survival Function&#39;)
    plt.title(&#39;Data Survival Function Plot&#39;)
    plt.scatter(sorted(sample_over_thresh), y_surv, label=&#39;Empirical Survival Function&#39;)
    plt.plot(sorted(sample_over_thresh), F1, linestyle=&#39;--&#39;, color=&#39;red&#39;, alpha = 0.8, lw = 0.9, label = &#39;Dvoretzky–Kiefer–Wolfowitz Confidence Bands&#39;)
    plt.plot(sorted(sample_over_thresh), F2, linestyle=&#39;--&#39;, color=&#39;red&#39;, alpha = 0.8, lw = 0.9)
    plt.legend()
    plt.show()</code></pre>
</details>
</dd>
</dl>
</section>
<section>
</section>
</article>
<nav id="sidebar">
<h1>Index</h1>
<div class="toc">
<ul></ul>
</div>
<ul id="index">
<li><h3><a href="#header-functions">Functions</a></h3>
<ul class="two-column">
<li><code><a title="extremely.MRL" href="#extremely.MRL">MRL</a></code></li>
<li><code><a title="extremely.getBM" href="#extremely.getBM">getBM</a></code></li>
<li><code><a title="extremely.getPOT" href="#extremely.getPOT">getPOT</a></code></li>
<li><code><a title="extremely.gevVaR" href="#extremely.gevVaR">gevVaR</a></code></li>
<li><code><a title="extremely.gevfit" href="#extremely.gevfit">gevfit</a></code></li>
<li><code><a title="extremely.gevparams" href="#extremely.gevparams">gevparams</a></code></li>
<li><code><a title="extremely.gevsummary" href="#extremely.gevsummary">gevsummary</a></code></li>
<li><code><a title="extremely.gpdcdf" href="#extremely.gpdcdf">gpdcdf</a></code></li>
<li><code><a title="extremely.gpdfit" href="#extremely.gpdfit">gpdfit</a></code></li>
<li><code><a title="extremely.gpdparams" href="#extremely.gpdparams">gpdparams</a></code></li>
<li><code><a title="extremely.gpdpdf" href="#extremely.gpdpdf">gpdpdf</a></code></li>
<li><code><a title="extremely.gpdppplot" href="#extremely.gpdppplot">gpdppplot</a></code></li>
<li><code><a title="extremely.gpdqqplot" href="#extremely.gpdqqplot">gpdqqplot</a></code></li>
<li><code><a title="extremely.survival_function" href="#extremely.survival_function">survival_function</a></code></li>
</ul>
</li>
</ul>
</nav>
</main>
<footer id="footer">
<p>Generated by <a href="https://pdoc3.github.io/pdoc" title="pdoc: Python API documentation generator"><cite>pdoc</cite> 0.10.0</a>.</p>
</footer>
</body>
</html>
