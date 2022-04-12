# Table of contents

 1. [Introduction](#introduction)

 2. [Features](#features)

 3. [Generalized Extreme Value (GEV) Distribution](gev.md){:target="_blank"}
     
 4. [Generalized Pareto Distribution (GPD)](gpd.md){:target="_blank"}
   
 5. [Python notebook with examples](#notebook)
  

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

## Python [Notebook](https://github.com/SURYA-LAMICHANEY/ExtremeLy/blob/main/Examples.ipynb) with examples. <a name="notebook"></a>
