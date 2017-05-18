"""
Module containing classes used in SHAM models
"""

from __future__ import (division, print_function, absolute_import)

import numpy as np
from astropy.table import Table
from astropy.io import ascii
from scipy.interpolate import interp1d
from warnings import warn

__all__ = ['HaloProps', 'Orphans']

class HaloProps(object):
    """
    class to carry over halo properties to galaxy mock
    """
    def __init__(self,
                 haloprop_keys = ['halo_mpeak','halo_vpeak',
                                  'halo_acc_scale','halo_half_mass_scale'],
                 **kwargs):
        """
        Parameters
        ----------
        haloprop_keys : list
        """
        
        self._mock_generation_calling_sequence = []
        self._galprop_dtypes_to_allocate = np.dtype([])
        self.list_of_haloprops_needed = haloprop_keys


class Orphans(object):
    """
    class to model orphan galaxies in SHAM models
    """
    def __init__(self, halo_clone_key='halo_clone', prim_haloprop_key='halo_mpeak', **kwargs):
        """
        Parameters
        ----------
        halo_clone_key : string
            key flagging cloned haloes
        
        prim_haloprop_key : string
            key indicating halo property for which the completeness is modeled
        
        c_m0 : float, optional
            completeness cut-off scale.  The mass at which completeness goes to 0
        
        c_gamma : float, optional
            completeness cut-off steepness.  The steepness with which completeness drops.
            
        c_ceil : float, optional
            completeness ceiling.  The maximum completeness at all masses.
        
        c_floor : float, optional
            completeness floor.  The minimum completeness at all masses.
        """
        
        self._mock_generation_calling_sequence = ['assign_orphan']
        self._galprop_dtypes_to_allocate = np.dtype([('orphan', 'i4'),('remove', 'i4')])
        self._methods_to_inherit = (['assign_orphan','subhalo_completeness'])
        self.list_of_haloprops_needed = [halo_clone_key, prim_haloprop_key]
        
        self.param_dict = self.retrieve_default_param_dict()
        for key in kwargs.keys():
            self.param_dict[key] = kwargs[key]
        
        self._halo_clone_key = halo_clone_key
        self._prim_haloprop_key = prim_haloprop_key
        
        self._sample_rate = 10.0 #number of times a extant sub-haloes are cloned
    
    def subhalo_completeness(self, m):
        """
        model for sub-halo completeness
        
        Parameters
        ----------
        m : array_like
            array of mass-like halo properties
        
        Returns
        -------
        c: numpy.array
            array of subhalo completeness at mass `m`
        """
        
        m0 = self.param_dict['c_m0']
        m0 = 10.0**m0
        gamma = self.param_dict['c_gamma']
        ceil = self.param_dict['c_ceil']
        floor = self.param_dict['c_floor']
        
        m = np.atleast_1d(m)
        c = -1.0*(m/m0)**(-1.0*10**gamma) + 1.0 - (1.0-ceil)
        c = ceil*((m/m0)**(-gamma) + 1.0)**(-1.0)
        
        return np.maximum(c, floor)
    
    def assign_orphan(self,  **kwargs):
        """
        determine which clone sub-haloes should be populated and flag as orphans
        """
        table = kwargs['table']
        
        clones = np.where(table[self._halo_clone_key]==1)[0]
        
        N_clones = len(clones)
        N_haloes = len(table)
        
        weigths = np.random.random(N_clones) * self._sample_rate
        
        c = self.subhalo_completeness(table[self._prim_haloprop_key][clones])
        
        #determine which clones to keep
        keep = (weigths <= (1.0/c - 1.0))
        
        #flag orphans
        table['orphan'] = 0
        table['orphan'][clones[keep]] = 1
        
        #flag (sub-)haloes which should not be used
        mask = np.array([True]*N_haloes)
        mask[clones] = False
        mask[clones[keep]] = True
        table['remove'] = (mask==False)
        
        return mask
    
    def retrieve_default_param_dict(self):
        """ 
        Method returns a dictionary of all model parameters
        
        Returns
        -------
        d : dict
            Dictionary containing parameter values.
        """
        
        d = {
            'c_m0' : 9.97963619845,
            'c_gamma' : 1.26550335196,
            'c_ceil' : 1.0,
            'c_floor' : 0.1
        }
        
        return d