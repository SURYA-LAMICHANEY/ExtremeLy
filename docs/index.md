# Table of contents
 1. [Introduction](#introduction)

 2. [Features](#features)

 3. [Functions](#functions)
     1. [getBM](#getBM)
     2. [gevFit](#gevFit)
     3. [gevParams](#gevParams)
     4. [gevSummary](#gevSummary)
     5. [gevVar](#gevVaR)
     6. [MRL](#MRL)
     7. [getPOT](#getPOT)
     8. [gpdfit](#gpdfit)
     9. [gpdparams](#gpdparams)
     10. [gpdpdf](#gpdpdf)
     11. [gpdcdf](#gpdfcdf)
     12. [gpdqqplot](#gpdqqplot)
     13. [gpdppplot](#gpdppplot)
     14. [survival_function](#survival_function)
   
 4. [Python notebook with examples](#notebook)
 5. [Discussion](#discussion)

         

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


## Functions <a name="functions"></a>
### 1.  _getBM(sample, period)_ <a name="getBM"></a>

   In Block Maxima method we divide the whole dataset into blocks and select the largest value in each block as an extreme value.
    
##### Parameters
    
   _sample_ : pandas dataframe <br/>
              The whole dataset
            
   _period_ : string <br/>
             The time period on basis of which the blocks are created. Eg - yearly, monthly, weekly and daily.

##### Returns
    
   _maxima_reset_ : pandas dataframe <br/>
                    Maxima values obtained 

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
        

 ### 3. _gevparams(model)_ <a name="gevParams"></a>

   Accesing estimated distribution parameters from the GEV fit.
   
##### Parameters

   _model_ : object <br/>
           Object containing the information about GEV fit.

##### Returns

   _OrderedDict_ <br/>
   Returns estimated distribution parameters. 
  
  
### 4. _gevsummary(model)_ <a name="gevSummary"></a>

   Plotting plots like QQplot, PPplot, Return Level plot and density plot for the GEV model.
##### Parameters

   _model_ : object <br/>
             Object containing the information about GEV fit.

##### Returns

   _None_


### 5. _getPOT(sample, threshold)_ <a name="getPOT"></a>

   In Peak-Over-Threshold method the values greater than a given threshold are taken as extreme values.
##### Parameters

   _sample_ : pandas dataframe <br/>
              The whole Dataset
              
   _threshold_ : integer <br/>
                 An integer value above which the values are taken as extreme values.

##### Returns

 _exce_ : pandas dataframe <br/>
            Excess values obtained. 
