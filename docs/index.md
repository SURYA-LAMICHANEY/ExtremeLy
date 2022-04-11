# Table of contents
 1. [Introduction](#introduction)

 2. [Features](#features)

 3. [Generalized Extreme Value (GEV) Distribution](#gev)
     1. [getBM](#getBM)
     2. [gevFit](#gevFit)
     3. [gevParams](#gevParams)
     4. [gevSummary](#gevSummary)
     
 4. [Generalized Pareto Distribution (GPD)](#gpd)
     1. [MRL](#MRL)
     2. [getPOT](#getPOT)
     3. [gpdfit](#gpdfit)
     4. [gpdparams](#gpdparams)
     5. [gpdpdf](#gpdpdf)
     6. [gpdcdf](#gpdcdf)
     7. [gpdqqplot](#gpdqqplot)
     8. [gpdppplot](#gpdppplot)
     9. [survival_function](#survival_function)
   
 5. [Python notebook with examples](#notebook)
 6. [Discussion](#discussion)

         

## Introduction <a name="introduction"></a>

ExtremeLy is a python package for Extreme Value Analysis. It was found that there are not many packages
for EVA in python. Among existing packages some of them were incomplete, some of them were internally using R
packages and some had only basic implementations without any plots for model assessment. So ExtremeLy brings all
those packages together, removes R dependencies and provides most of the functionalities for EVA in python
without being dependent on R packages. Some fucntionalities from the already existing packages have been used
as they are, some have been modified to accommodate additional requirements and for some just the R dependencies
are replaced with python implementation. The three already existing packages that are used here are:

   1. [scikit-extremes](https://scikit-extremes.readthedocs.io/en/latest/)
   2. [thresholdmodeling](https://github.com/iagolemos1/thresholdmodeling)
   3. [evt](https://pypi.org/project/evt/#description)
   
   
## Features <a name="features"></a>

There are basically two approaches to Extreme Value Analysis:

   1. Block maxima method + Generalized Extreme Value (GEV) Distribution.
   2. Peak-over-threshold method + Generalized Pareto Distribution (GPD).
   
ExtremeLy provides all the necessary functionalities for performing Extreme Value Analysis. You can carry out the following tasks:

   1. Finding extreme values from the data using methods like Block Maxima or Peaks-Over-Threshold.
   2. Fitting the extracted extreme values to the continuous distributions like Generalized Extreme Value (GEV) Distributions or Generalized Pareto Distribution (GPD).
   3. Using visual plots for extreme values, results and goodness-of-fit statistics of the fitted model.
   4. Estimating the extreme values of given probability and corresponding confidence interval.


## Generalized Extreme Value (GEV) Distribution <a name="gev"></a>

Generalized Extreme Value (GEV) Distribution is a limit distribution of properly normalized maxima of sequence of independent and identically
distributed random variables. It is specified by three parameters : location, shape and scale. Visit [Wikipedia](https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution) page for more information.<br/>
### 1.  _getBM(sample, period)_ <a name="getBM"></a>

   In Block Maxima method we divide the whole dataset into blocks and select the largest value in each block as an extreme value.
    
##### Parameters
    
   _sample_ : pandas dataframe <br/>
              The whole dataset
            
   _period_ : string <br/>
             The time period on basis of which the blocks are created. Eg - yearly, monthly, weekly and daily.

##### Returns
    
   _maxima_reset_ : pandas dataframe <br/>

 <details><summary> <strong>Expand for source code</strong> </summary>
{% highlight python %}
 
    #Block Maxima method for finding large values. 
    def getBM(sample,period): 
 
       #Obtain the maximas   
        colname=list(sample)   
        sample.iloc[:,0]= pd.to_datetime(sample.iloc[:,0])   
        maxima = sample.resample(period, on=colname[0]).max()   
        maxima_reset=maxima.reset_index(drop=True)   
        series=pd.Series(sample.iloc[:,1])   
        series.index.name=index    
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
{% endhighlight %}
</details>

#### Example
   
```python
   from ExtremeLy import extremely as ely
   #Here Y means Yearly, we can pass M for monthly, W for weekly and D for daily.
   maxima=ely.getBM(sample=data,period="Y") 
   maxima
```

#### Output      
       	Date 	Loss
    0 	1980-12-31 	263.250366
    1 	1981-12-31 	56.225426
    2 	1982-12-31 	65.707491
    3 	1983-12-30 	13.348165
    4 	1984-12-31 	19.162304
    5 	1985-12-29 	57.410636
    6 	1986-12-30 	29.026037
    7 	1987-12-31 	32.467532
    8 	1988-12-26 	47.019521
    9 	1989-12-31 	152.413209
    10 	1990-12-31 	144.657591

   ![BlockMaxima](https://raw.githubusercontent.com/surya-lamichaney/ExtremeLy/master/assets/blockMaxima.png)


### 2.  _gevfit(sample, fit_method='mle', ci=0, ci_method='delta')_ <a name="gevFit"></a>

   GEV is a limit distribution of properly normalized maxima of sequence of independent and identically distributed random variables. It is     parameterized by scale, shape and location parameters.
    
##### Parameters

   _sample_ : pandas dataframe <br/>
              maximas obtained from Block Maxima method
        
   _fit_method_ : string <br/>
                  Estimation method like Maximum Likelihood Estimation or Method of Moments. Default is MLE.
        
   _ci_ : float <br/>
          Confidence interval. Default is 0.
        
   _ci_method_ : string <br/>
                 Method used for Confidence Interval like Delta or Bootstrap. Default is Delta.

##### Returns

   model : object <br/>
           Object containing the information about GEV fit. 

<details><summary> <strong>Expand for source code</strong> </summary>
{% highlight python %}
 
     #Using classic model of skextreme for GEV fitting
     model = sk.models.classic.GEV(sample.iloc[:,1], fit_method = fit_method, ci = ci,ci_method=ci_method)
     return model
{% endhighlight %}
</details>

#### Example

```python
#Fitting the GEV distribution with maxima values. 
#Here, default fit_method is MLE and default Confidence interval method is delta.
fit=ely.gevfit(sample=maxima,fit_method="mle",ci=0,ci_method="delta")
```

### 3. _gevparams(model)_ <a name="gevParams"></a>

   Accesing estimated distribution parameters from the GEV fit.
   
##### Parameters

   _model_ : object <br/>
           Object containing the information about GEV fit.

##### Returns

   _OrderedDict_ <br/>
   Returns estimated distribution parameters. 
  
<details><summary> <strong>Expand for source code</strong> </summary>
{% highlight python %}
 
     #Return the estimated parameters
     return model.params
{% endhighlight %}
</details>

#### Example 

```python
#Getting estimated distribution parameters for GEV fit.
params=ely.gevparams(model=fit)
params
```
#### Output

    OrderedDict([('shape', -0.6384049125307144),
             ('location', 37.79353853187126),
             ('scale', 28.93607752286071)])
             

### 4. _gevsummary(model)_ <a name="gevSummary"></a>

   Plotting plots like QQplot, PPplot, Return Level plot and density plot for the GEV model.
##### Parameters

   _model_ : object <br/>
             Object containing the information about GEV fit.

##### Returns

   _None_

<details><summary> <strong>Expand for source code</strong> </summary>
{% highlight python %}
 
     #Display all the plots together
     model.plot_summary()
{% endhighlight %}
</details>

#### Example

```python
#Summarizing the GEV model with various plots like QQplot, PPplot, 
#Return Level Plot and Data Probability density plot
ely.gevsummary(model=fit)
```
#### Output

   ![GEVSummary](https://raw.githubusercontent.com/surya-lamichaney/ExtremeLy/master/assets/summary.png)
   

## Generalized Pareto Distribution (GPD) <a name="gpd"></a>

Generalized Pareto Distribution (GPD) is a family of of continuos probability distributions and is often used to model the tails of another distribution. It is sometimes specified by three parameters : location, shape and scale, but it can also be specified by only shape and scale parameters. Visit 
[Wikipedia](https://en.wikipedia.org/wiki/Generalized_Pareto_distribution) page for more information.
### 1. _MRL(sample, alpha=0.05)_  <a name="MRL"></a>

   Mean Residual Life plot takes mean of excess values above a threshold minus threshold and plots it against that threshold value. If the plot is linear that is ok but if the plot starts loosing stability then choose that threshold value. <br/>
##### Parameters

   _sample_ : pandas dataframe <br/>
              the whole dataset
              
   _alpha_ : float
            a number giving 1-alpha confidence levels to use. Default value is 0.05.

##### Returns

   None

<details><summary> <strong>Expand for source code</strong> </summary>
{% highlight python %}
 
     #Mean Residual Life Plot for finding appropriate threshold value.
     def MRL(sample, alpha=0.05):
    
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
{% endhighlight %}
</details>

#### Example

```python
#Mean Residual Life plot for finding appropriate threshold 
#value for Peak-Over-Threshold method.
ely.MRL(sample=data,alpha=0.05)
```

#### Output

![MeanResidualLifePlot](https://raw.githubusercontent.com/surya-lamichaney/ExtremeLy/master/assets/mrl.png)

  
### 2. _getPOT(sample, threshold)_ <a name="getPOT"></a>

   In Peak-Over-Threshold method the values greater than a given threshold are taken as extreme values.
##### Parameters

   _sample_ : pandas dataframe <br/>
              The whole Dataset
              
   _threshold_ : integer <br/>
                 An integer value above which the values are taken as extreme values.

##### Returns

 _exce_ : pandas dataframe <br/>
          Excess values obtained. 

<details><summary> <strong>Expand for source code</strong> </summary>
{% highlight python %}
 
      #Peaks-Over-Threshold method for finding large values.
      def getPOT(sample,threshold):
          colnames=list(sample)
          exce=sample[sample.iloc[:,1].gt(threshold)]
    
          #Plotting the excess values obtained using POT method.
          ax = sample.iloc[:,1].reset_index().plot(kind='scatter', x='index', y=colnames[1],
                                           color='Red', label='Below Threshold')
          exce.reset_index().plot(kind='scatter', x='index', y=colnames[1],
                                          color='Blue', label='Above threshold', ax=ax)
          return exce
{% endhighlight %}
</details>

#### Example 

 ```python
#Getting large claims using POT method using threshold value as 30.
pot=ely.getPOT(sample=data,threshold=30)
pot
``` 

#### Output
          	
    index 	  Date         	Loss
      82  	1980-07-15 	263.250366
      178   1981-02-10 	34.141547
      232 	1981-05-29 	56.225426
      330 	1981-12-21 	50.065531
      478 	1982-10-24 	65.707491
      887 	1985-03-04 	46.500000
      972 	1985-08-23 	57.410636
      1388 	1987-06-05 	32.467532
      1549 	1988-03-25 	38.154392
      1641 	1988-08-12 	47.019521
      1710 	1988-12-17 	31.055901
      1740 	1989-02-14 	42.091448
      1856 	1989-08-04 	152.413209
      1909 	1989-10-22 	32.387807
      2121 	1990-10-08 	144.657591
      
![Peak-over-threshold](https://raw.githubusercontent.com/surya-lamichaney/ExtremeLy/master/assets/pot.png)


### 3. _gpdfit(sample, threshold)_ <a name="gpdfit"></a>

   GPD is a family of continous probability distributions and is often used to model the tails of another distribution. It is specified by two parameters - Shape and scale parameters. <br/>
##### Parameters

   _sample_ : pandas dataframe <br/>
              The whole Dataset
        
   _threshold_ : integer
                 An integer value above which the values are taken as extreme values.

##### Returns

   Estimated distribution parametrs of GPD fit, sample excess values and value above the threshold.
   
<details><summary> <strong>Expand for source code</strong> </summary>
{% highlight python %}
 
    #Fit the large claims obtained from POT method to Generalized Pareto distribution (GPD).
    def gpdfit(sample, threshold):
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
{% endhighlight %}
</details>

#### Example

```python
#Fitting GPD with large claims obtained using POT method.
gpdfit=ely.gpdfit(sample=data,threshold=30)
```
        
### 4. _gpdparams(fit)_ <a name="gpdparams"></a>

   Getting estimated distribution parameters for GPD fit. <br/>
##### Parameters

   _fit_ : object <br/>
           Object containing information about GPD fit.

##### Returns

   None

<details><summary> <strong>Expand for source code</strong> </summary>
{% highlight python %}
 
    #Get estimated distribution parameters for GPD fit.
    def gpdparams(fit):
        shape=fit[0]
        scale=fit[1]
        print("Shape:",shape)
        print("Scale:",scale)
{% endhighlight %}
</details>

#### Example

```python
#Getting estimated distribution parameters for GPD fit.
ely.gpdparams(fit=gpdfit)
```

#### Output

    Shape: 0.6586260117024005
    Scale: 19.267021192664032
    
### 5. _gpdpdf(sample, threshold, bin_method, alpha)_ <a name="gpdpdf"></a>

   Probability density plots are used to understand data distribution for a continuous variable and we want to know the likelihood (or probability) of obtaining a range of values that the continuous variable can assume. <br/>
##### Parameters

   _sample_ : pandas dataframe <br/>
        The whole Dataset
        
   _threshold_ : integer <br/>
        An integer value above which the values are taken as extreme values.
        
   _bin_method_ : string <br/>
        Binning algorithm, specified as one of the following - auto, scott, fd, sturges, integers and sqrt.
        
   _alpha_ : float <br/>
        a number giving 1-alpha confidence levels to use. Default value is 0.05.

##### Returns

   None
<details><summary> <strong>Expand for source code</strong> </summary>
{% highlight python %}
 
     #get PDF plot with histogram to diagnostic the model
     def gpdpdf(sample, threshold, bin_method, alpha):
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
   
{% endhighlight %}
</details>

#### Example

```python
#Data Probability Density Function plot.
ely.gpdpdf(sample=data,threshold=30,bin_method="sturges",alpha=0.05)
```

#### Output

![GPD-PDF](https://raw.githubusercontent.com/surya-lamichaney/ExtremeLy/master/assets/gpdpdf.png)


### 6. _gpdcdf(sample, threshold, alpha)_ <a name="gpdcdf"></a>

   The cumulative distribution function of a real-valued random variable X, or just distribution function of X, evaluated at x, is the probability that X will take a value less than or equal to x. <br/>
##### Parameters

   _sample_ : pandas dataframe <br/>
              The whole Dataset
        
   _threshold_ : integer <br/>
                 An integer value above which the values are taken as extreme values.

   _alpha_ : float <br/>
             a number giving 1-alpha confidence levels to use. Default value is 0.05.
   
##### Returns

   None
   
### 7. _gpdqqplot(mle)_ <a name="gpdqqplot"></a>

   QQplot is a graphical technique for determining if two datasets come from populations with a common distribution. A 45 degree reference line is plotted, if the two sets come from populations with the same distribution, the points should fall approximately along this reference line. <br/>
##### Parameters

   _mle_ : object <br/>
           MLE estimator object from evt library.

##### Returns

   None
   
### 8. _gpdppplot(sample, threshold, alpha)_ <a name="gpdppplot"></a>

   PP-plot is used for assessing how closely two datasets agree. It plots the two cumulative distribution functions against each other.<br/>
##### Parameters

   _sample_ : pandas dataframe<br/>
              The whole Dataset
              
   _threshold_ : integer <br/>
                 An integer value above which the values are taken as extreme values.

   alpha : float <br/>
           a number giving 1-alpha confidence levels to use. Default value is 0.05.
           
##### Returns

   None
### 9. _survival_function(sample, threshold, alpha)_ <a name="survival_function"></a>

   The survival function is a function that gives the probability that the object of interest will survive past a certain time. <br/>
##### Parameters

   _sample_ : pandas dataframe <br/>
              The whole Dataset
        
   _threshold_ : integer <br/>
                 An integer value above which the values are taken as extreme values.

   _alpha_ : float <br/>
             a number giving 1-alpha confidence levels to use. Default value is 0.05.
   
##### Returns

None
