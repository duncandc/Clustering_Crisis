# Repository for "The Galaxy Clustering Crisis in Abundance Matching"


This repository contains all of the code necessary to re-create the analysis for 
Campbell et al. (2017), "The Galaxy Clustering Crisis in Abundance Matching".


### Files
---------
SHAM_model_components.py contains classes for rank order SHAM models

SMHM_model_components.py contains classes for evolving SMHM models

Halo_model_components.py contains classes for dark component models

make_fiducial_SHAM_mocks.py creates a fiducial mock for each model

utils.py contains some utility functions used in this work

cosmo_utils.py contains some cosmology functions

default.py contains some default filepaths and fiducial models

### Directories
---------------
data : contains scripts to download data products

notebooks : contains ipython notebooks used for the analysis

paper : contains tex files necessary toi compile the paper

figures : contains figures created during the analysis


The analysis requires halotools among other standard Python packages.


Please contact me at: duncan.campbell@yale.edu if you run into any problems.



&copy; Copyright 2017, Duncan Campbell 
