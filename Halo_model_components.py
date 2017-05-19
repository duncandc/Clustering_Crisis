"""
Module containing classes used in SHAM models
"""

from __future__ import (division, print_function, absolute_import)

import re
import glob
import os
import numpy as np
from astropy.table import Table
from astropy.io import ascii
from scipy.interpolate import interp1d
from warnings import warn

from utils import table_to_array

from astropy.cosmology import FlatLambdaCDM
default_cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

__all__ = ['HaloProps', 'Orphans', 'PWGH',  'MAH', 'Continued_Growth']

PROJECT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))+'/'

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


class PWGH(object):
    """
    median potential well growth history of host-haloes 
    from van den Bosch et al. (2014), arXiv:1409.2750
    """
    
    def __init__(self, cosmo):
        """
        Parameters
        ----------
        cosmo : astropy.cosmology object
        """
        
        self.publications = ['arXiv:1409.2750']
        self.load_pwgh()
    
    def load_pwgh(self):
        """
        Load precomputed PWGHs from van den Bosch et al. (2014)
        """
        filepath = PROJECT_DIRECTORY + 'data/PWGH/'
        fnames = glob.glob(filepath+'./*.dat')
        
        mass = np.zeros((len(fnames),))
        arr = np.zeros((len(fnames),400,8))
        for i, fname in enumerate(fnames):
            m = re.findall("\d+\.\d+",fname)
            if len(m)==1:
                mass[i] = float(m[0])
            else:
                mass[i] = float(re.findall("\d+",fname)[0])
            table = ascii.read(fname)
            arr[i,:,:] = table_to_array(table)
            arr[i,:,3] = np.log10((10**arr[i,:,3])*10.0**mass[i])
            v_vir = 159.43 * (10.0**mass[i]/10**12)**(1.0/3.0)
            arr[i,:,5] = np.log10((10**arr[i,:,5])*v_vir)
        
        #sort arrays by z=0.0 mass
        sort_inds  = np.argsort(mass)
        
        self._pwgh_mass = mass[sort_inds]
        self._pwgh_arr = arr[sort_inds]
    
    def pwgh(self, v_max, z):
        """
        return the average potential well growth history of a 
        halo given ``v_max`` at a sepcifiec redshift, ``z``.
        
        Parameters
        ----------
        v_max : array_like
            maximum circular velocity
            
        z : array_like
            redshift correspinding to the maximum circular velocity
        
        Returns
        -------
        v_max(z) : numpy.array
            average pwgh
        
        z : numpy.array
            redshift
        """
        
        v_max = np.atleast_1d(v_max)
        z = np.atleast_1d(z)
        
        v_max = np.log10(v_max)
        
        N_haloes = len(v_max)
        
        #find the nearest redshift in the tabulated growth histories for each halo
        z1 = self._pwgh_arr[0,:,1]
        z_inds = np.searchsorted(z1,z)
        
        #find the growth history that matches each halo
        vs = (self._pwgh_arr[:,z_inds,5]).T
        v_max = v_max.reshape((1,N_haloes)).T
        vv = np.fabs(vs-v_max)
        v_inds = np.argmin(vv,axis=1)
        
        return self._pwgh_arr[v_inds,:,5], self._pwgh_arr[0,:,1]


class MAH(object):
    """
    median mass accretion history of host-haloes
    from van den Bosch et al. (2014), arXiv:1409.2750
    """
    
    def __init__(self, cosmo):
        """
        Parameters
        ----------
        cosmo : astropy.cosmology object
        """
        
        self.publications = ['arXiv:1409.2750']
        self.load_mah()
    
    def load_mah(self):
        """
        Load precomputed MAHs from van den Bosch et al. (2014)
        """
        filepath = PROJECT_DIRECTORY + 'data/PWGH/'
        fnames = glob.glob(filepath+'./*.dat')
        
        mass = np.zeros((len(fnames),))
        arr = np.zeros((len(fnames),400,8))
        for i, fname in enumerate(fnames):
            m = re.findall("\d+\.\d+",fname)
            if len(m)==1:
                mass[i] = float(m[0])
            else:
                mass[i] = float(re.findall("\d+",fname)[0])
            table = ascii.read(fname)
            arr[i,:,:] = table_to_array(table)
            arr[i,:,3] = np.log10((10**arr[i,:,3])*10.0**mass[i])
            v_vir = 159.43 * (10.0**mass[i]/10**12)**(1.0/3.0)
            arr[i,:,5] = np.log10((10**arr[i,:,5])*v_vir)
        
        #sort arrays by z=0.0 mass
        sort_inds  = np.argsort(mass)
        
        self._pwgh_mass = mass[sort_inds]
        self._pwgh_arr = arr[sort_inds]
    
    def mah(self, m_max, z):
        """
        return the median mass accreiton history of a 
        halo given ``m_max`` at a sepcifiec redshift, ``z``.
        
        Parameters
        ----------
        m_max : array_like
            maximum halo mass
            
        z : array_like
            redshift correspinding to the maximum circular velocity
        
        Returns
        -------
        m_max(z) : numpy.array
            average mah
        
        z : numpy.array
            redshift
        """
        
        v_max = np.atleast_1d(m_max)
        z = np.atleast_1d(z)
        
        v_max = np.log10(v_max)
        
        N_haloes = len(v_max)
        
        #find the nearest redshift in the tabulated growth histories for each halo
        z1 = self._pwgh_arr[0,:,1]
        z_inds = np.searchsorted(z1,z)
        
        #find the growth history that matches each halo
        vs = (self._pwgh_arr[:,z_inds,3]).T
        v_max = v_max.reshape((1,N_haloes)).T
        vv = np.fabs(vs-v_max)
        v_inds = np.argmin(vv,axis=1)
        
        return self._pwgh_arr[v_inds,:,3], self._pwgh_arr[0,:,1]


class Continued_Growth(object):
    """
    class to extrapolate subhalo mass at redshift z1 to redshift z2, z1>z2, 
    using the median MAHs from van den Bosch et al. (2014), arXiv:1409.2750
    
    The primary use of this class is to model continued stellar mass growth in satellites.
    """
    
    def __init__(self, 
        prim_haloprop_key = 'halo_mpeak',
        acc_scale_haloprop_key = 'halo_acc_scale',
        subhalo_haloprop_key = 'halo_upid',
        dt = 1.0,
        cosmo=None,
         **kwargs):
        """
        Parameters
        ----------
        prim_haloprop_key : string
            key word indicating the halo property to extrapolate
        
        acc_scale_haloprop_key : string
            key word indicatting the halo accretion scale property to use
        
        subhalo_haloprop_key : string
            key word indicating subhaloes.  subhaloes != -1
        
        dt : float
            time in Gyr to extrapolate after the time of accretion
        """
        
        if cosmo is None:
           cosmo = default_cosmo
        
        self.publications = ['arXiv:1409.2750']
        
        self._mock_generation_calling_sequence = ['assign_extrap_mass']
        self._galprop_dtypes_to_allocate = np.dtype([('extrap_'+prim_haloprop_key,'f4'),('extrap_scale','f4')])
        
        self.list_of_haloprops_needed = [prim_haloprop_key,
            acc_scale_haloprop_key, subhalo_haloprop_key]
        self._prim_haloprop_key = prim_haloprop_key
        self._acc_scale_haloprop_key = acc_scale_haloprop_key
        self._subhalo_haloprop_key = subhalo_haloprop_key
        
        self.param_dict = ({'extrap_dt' : dt})
        
        self.load_pwgh()
    
    def assign_extrap_mass(self,  **kwargs):
        """
        calculate the extrapolated mass at t0 + dt for subhaloes
        """
        table = kwargs['table']
        
        a_acc = table[self._acc_scale_haloprop_key]
        
        #only extrapolate subhaloes
        subs = (table[self._subhalo_haloprop_key]!=-1)
        if 'remove' in table.keys():
            not_remove = (table['remove']==0)
            mask = subs & not_remove
        else:
            mask = subs
        
        z_hist, m_hist, ex_a, m = self.extrapolate_mass(table[self._prim_haloprop_key][mask],
                                   a_acc[mask], self.param_dict['extrap_dt'])
        m = 10.0**m
        
        table['extrap_'+self._prim_haloprop_key] = table[self._prim_haloprop_key]
        table['extrap_'+self._prim_haloprop_key][mask] = m
        table['extrap_scale'] = a_acc
        table['extrap_scale'][mask] = ex_a
    
    def load_pwgh(self):
        """
        load average halo growth histories precomputed using code from van den Bosch 2014
        """
        filepath = '/Users/duncan/Documents/projects/beyond_age_matching/make_mocks/PWGH/'
        fnames = glob.glob(filepath+'./*.dat')
        
        mass = np.zeros((len(fnames),))
        arr = np.zeros((len(fnames),400,8))
        for i, fname in enumerate(fnames):
            m = re.findall("\d+\.\d+",fname)
            if len(m)==1:
                mass[i] = float(m[0])
            else:
                mass[i] = float(re.findall("\d+",fname)[0])
            table = ascii.read(fname)
            arr[i,:,:] = table_to_array(table)
            arr[i,:,3] = np.log10((10**arr[i,:,3])*10.0**mass[i])
        
        #sort arrays by z=0.0 mass
        sort_inds  = np.argsort(mass)
        
        self._pwgh_mass = mass[sort_inds]
        self._pwgh_arr = arr[sort_inds]
    
    def extrapolate_mass(self, m_acc, a_acc, dt):
        """
        Extrapolate the mass of a subhalo some time interval since accretion assuming 
        continued growth using average halo growth histories. 
        
        Parameters
        ----------
        m_acc : array_like
            halo mass at the time of accretion
            
        a_acc : array_like
            scale factor at time of accretion
            
        dt : array_like
            time since accretion to extrapolate mass
            
        Returns
        -------
        m : average mass growth history of halo(es)
        
        a1 : the scale factor the mass hase been extrapolated to
        
        m1 : extrapolated mass
        """
        
        m_acc = np.atleast_1d(m_acc)
        a_acc = np.atleast_1d(a_acc)
        dt = np.atleast_1d(dt)
        
        m_acc = np.log10(m_acc)
        
        N_haloes = len(m_acc)
        
        #convert scale factor at time of accretion to redshift
        z_acc = 1.0/a_acc - 1.0
        
        #find the nearest redshift in the tabulated growth histories for each halo
        z1 = self._pwgh_arr[0,:,1]
        z_inds = np.searchsorted(z1,z_acc)
        
        #find the growth history that matches each halo
        ms = (self._pwgh_arr[:,z_inds,3]).T
        m_acc = m_acc.reshape((1,N_haloes)).T
        mm = np.fabs(ms-m_acc)
        m_inds = np.argmin(mm,axis=1)
        
        #find the extrapolated mass
        lookback_t = self._pwgh_arr[0,:,2]
        t0 = lookback_t[z_inds]
        t1 = t0 - dt
        
        t_inds = np.searchsorted(lookback_t,t1)
        a1 = 1.0/(1.0+z1[t_inds])
        
        return self._pwgh_arr[0,:,1], self._pwgh_arr[m_inds,:,3], a1, 10**self._pwgh_arr[m_inds,t_inds,3]