import os

PROJECT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))+'/'

#DATA_DIRECTORY = PROJECT_DIRECTORY + 'data/'
DATA_DIRECTORY = '/Volumes/burt/bam_data/'

#set default models
from halotools.empirical_models import SubhaloModelFactory
from SMHM_model_components import *
from SHAM_model_components import *


