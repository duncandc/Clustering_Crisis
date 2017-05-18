import os
import numpy as np

PROJECT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))+'/'

#DATA_DIRECTORY = PROJECT_DIRECTORY + 'data/'
DATA_DIRECTORY = '/Volumes/burt/bam_data/'

#set default models
from halotools.empirical_models import SubhaloModelFactory
from SMHM_model_components import *
from SHAM_model_components import *
from Halo_model_components import *

#define galaxy selection function
def galaxy_selection_func(table):
    """
    define which galaxies should appear in the mock galaxy table
    
    Parameters
    ----------
    table : astropy.table object
        table containing mock galaxies
    """
    
    mask = (table['stellar_mass'] >= 10**9.5) & (table['stellar_mass'] < np.inf)
    
    return mask

#carry over some halo properties to the mock table
additional_halo_properties = HaloProps()

Lbox = 250.0 #h^-1 Mpc size of simulation box
redshift = 0.0

##############################
##### Define fiducial RM model
##############################
prim_haloprop_key = 'halo_mpeak'
mstar_model = RankSmHm(prim_haloprop_key=prim_haloprop_key, Lbox=Lbox, redshift=redshift)
composite_model_1 = SubhaloModelFactory(stellar_mass = mstar_model, 
                                        haloprops = additional_halo_properties,
                                        galaxy_selection_func = galaxy_selection_func)
composite_model_1.param_dict['log_scatter'] = 0.0


##############################
##### Define fiducial RV model
##############################
prim_haloprop_key = 'halo_vpeak'
mstar_model = RankSmHm(prim_haloprop_key=prim_haloprop_key, Lbox=Lbox, redshift=redshift)
composite_model_2 = SubhaloModelFactory(stellar_mass = mstar_model,
                                        haloprops = additional_halo_properties,
                                        galaxy_selection_func = galaxy_selection_func)
composite_model_2.param_dict['log_scatter'] = 0.0


###############################
##### Define fiducial M13 model
###############################
prim_haloprop_key =  'halo_mpeak200c'
acc_scale_key='halo_acc_scale'
mstar_model = MosterSmHm13(redshift=redshift,
                           prim_haloprop_key=prim_haloprop_key,
                           acc_scale_key=acc_scale_key)
mstar_conv = Guo_to_Blanton()
composite_model_3 = SubhaloModelFactory(stellar_mass = mstar_model,
                                        haloprops = additional_halo_properties,
                                        galaxy_selection_func = galaxy_selection_func,
                                        stellar_mass_conversion = mstar_conv,
                                        model_feature_calling_sequence = ('haloprops','stellar_mass','stellar_mass_conversion'))
composite_model_3.param_dict['log_scatter'] =  0.18


###############################
##### Define fiducial Y12 model
###############################
prim_haloprop_key =  'halo_mpeak180b'
acc_scale_key='halo_acc_scale'
mstar_model = Yang12SmHm(redshift=redshift,
                         prim_haloprop_key=prim_haloprop_key,
                         acc_scale_key=acc_scale_key)
mstar_conv = Bell_to_Blanton()
correction = Yang_correction()
composite_model_4 = SubhaloModelFactory(stellar_mass = mstar_model,
                                        haloprops = additional_halo_properties,
                                        galaxy_selection_func = galaxy_selection_func,
                                        stellar_mass_conversion = mstar_conv,
                                        stellar_mass_correction = correction,
                                        model_feature_calling_sequence = ('haloprops','stellar_mass','stellar_mass_conversion','stellar_mass_correction'))
composite_model_4.param_dict['log_scatter'] =  0.173
composite_model_4.param_dict['c'] =  1.0 #consistent with uncertainty


###############################
##### Define fiducial B13 model
###############################
prim_haloprop_key =  'halo_mpeak'
acc_scale_key='halo_acc_scale'
mstar_model = BehrooziSmHm13(redshift=redshift,
                             prim_haloprop_key=prim_haloprop_key,
                             acc_scale_key=acc_scale_key)
mstar_conv = Moustakas_to_Blanton()
composite_model_5 = SubhaloModelFactory(stellar_mass = mstar_model,
                                        haloprops = additional_halo_properties,
                                        galaxy_selection_func = galaxy_selection_func,
                                        stellar_mass_conversion = mstar_conv,
                                        model_feature_calling_sequence = ('haloprops','stellar_mass','stellar_mass_conversion'))
composite_model_5.param_dict['log_scatter'] =  0.21
