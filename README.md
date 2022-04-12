# ExtremeLy

ExtremeLy is a python package for Extreme Value Analysis. It was found that there are not many packages for EVA in python. 
Among existing packages some of them were incomplete, some of them were internally using R packages and some had only 
basic implementations without any plots for model assessment. So ExtremeLy brings all those packages together, removes 
R dependencies and provides most of the fucntionalities for EVA in pythonwithout being dependent on R packages. Some 
fucntionalities from the already existing packages have been usedas they are, some have been modified to accomodate 
additional requirements and for some just the R dependenciesare replaced with python implementation. 
The three already existing packages that are used here are:

   1. scikit-extremes skextremes - https://scikit-extremes.readthedocs.io/en/latest/
   2. thresholdmodeling - https://github.com/iagolemos1/thresholdmodeling
   3. evt - https://pypi.org/project/evt/#description

## Dependencies
   evt package will be downloaded with ExtremeLy package itself, threshmodeling is not required as it requires R environment 
   to run its functionalities. Those R dependencies have been removed in ExtremeLy. We still need to install skextremes before we can use ExtremeLy. 
   Scikit-extremes (skextremes) also has a dependency called lmoments3 which needs to be installed. These two libraries can be installed this way:
      
      pip install git+https://github.com/OpenHydrology/lmoments3.git

      git clone https://github.com/kikocorreoso/scikit-extremes.git

      cd scikit-extremes

      pip install -e .
 
 Now we are good to go and install ExtremeLy :)
 
 ## Installation
     pip install ExtremeLy
 
 ## Click [here](https://surya-lamichaney.github.io/ExtremeLy/) for the Documentation and [here](https://github.com/SURYA-LAMICHANEY/ExtremeLy/blob/main/Examples.ipynb) for example notebook.
