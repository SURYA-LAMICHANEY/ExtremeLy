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

         

## Introduction <a name="introduction"></a>
ExtremeLy is a python package for Extreme Value Analysis. It was found that there are not many packages
for EVA in python. Among existing packages some of them were incomplete, some of them were internally using R
packages and some had only basic implementations without any plots for model assessment. So ExtremeLy brings all
those packages together, removes R dependencies and provides most of the fucntionalities for EVA in python
without being dependent on R packages. Some fucntionalities from the already existing packages have been used
as they are, some have been modified to accomodate additional requirements and for some just the R dependencies
are replaced with python implementation. The three already existing packages that are used here are:

   1. [scikit-extremes](https://scikit-extremes.readthedocs.io/en/latest/)
   2. [thresholdmodeling](https://github.com/iagolemos1/thresholdmodeling)
   3. [evt](https://pypi.org/project/evt/#description)
   
   
There are basically two approaches to Extreme Value Analysis:

   1. Block maxima method + Generalized Extreme Value (GEV) Distribution.
   2. Peak-over-threshold method + Generalized Pareto Distribution (GPD).  


## Functions
### 1.  _getBM(sample, period)_

   In Block Maxima method we divide the whole dataset into blocks and select the largest value in each block as an extreme value.
    
   #### Parameters
    
   _sample_ : pandas dataframe
              The whole dataset
   _period_: string
             The time period on basis of which the blocks are created. Eg - yearly, monthly, weekly and daily.

   #### Returns
    
   _maxima_reset_ : pandas dataframe
                    Maxima values obtained 
  
### 2. _gevfit(sample, fit_method='mle', ci=0, ci_method='delta')_

   GEV is a limit distribution of properly normalized maxima of sequence of independent and identically distributed random variables. It is     parameterized by scale, shape and location parameters.
    
   #### Parameters

   _sample_ : pandas dataframe
              maximas obtained from Block Maxima method
        
   _fit_method_ : string
                  Estimation method like Maximum Likelihood Estimation or Method of Moments. Default is MLE.
        
   _ci_ : Float
          Confidence interval. Default is 0.
        
   _ci_method_ : string
                 Method used for Confidence Interval like Delta or Bootstrap. Default is Delta.

   #### Returns

   model : Object
           Object containing the information about GEV fit. 
        
![Screenshot](https://raw.githubusercontent.com/surya-lamichaney/ExtremeLy/master/assets/blockMaxima.png)




