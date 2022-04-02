from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

VERSION = '2.0.0'
DESCRIPTION = 'For performing Extreme value analysis in Python'

# Setting up
setup(
    name="ExtremeLy",
    version=VERSION,
    author="Surya Lamichaney",
    author_email="suryalamichaney38741@gmail.com",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=["evt","lmoments3"],
    keywords=["Extreme Value Analysis","Block Maxima","Peak-Over-Threshold","Generalized Extreme Value Distribution (GEV)","Generalized Pareto Distribution (GPD)"],
    
)
