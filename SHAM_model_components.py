"""
Module containing classes used in SHAM models
"""

from __future__ import (division, print_function, absolute_import)

import numpy as np

from astropy.table import Table
from astropy.io import ascii

from halotools.empirical_models.smhm_models.smhm_helpers import safely_retrieve_redshift
from halotools.empirical_models import model_helpers as model_helpers
from halotools.empirical_models.component_model_templates import PrimGalpropModel

from scipy.stats import norm
from scipy.special import erfinv
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz

from lss_observations import stellar_mass_functions

from utils import scatter_ranks


__all__ = ['RankSmHm','ParamSmHm']

class RankSmHm(object):
    """
    class to assign stellar mass based on halo mass rank
    """
    def __init__(self,
                 prim_haloprop_key = 'halo_mpeak',
                 stellar_mass_function = 'LiWhite_2009',
                 scatter = 0.2,
                 Lbox = 250.0,
                 **kwargs):
        """
        Parameters
        ----------
        prim_haloprop_key : string
            key indicating halo property for which the completeness is modeled
        
        stellar_mass_function : string
            string indicating which stellar mass function to use in lss_observations
            module
        
        Lbox : float
            length of simulation box side
        """
        
        if 'redshift' in list(kwargs.keys()):
            self.redshift = kwargs['redshift']
        else:
            self.redshift=0.0
        
        self._mock_generation_calling_sequence = ['_assign_stellar_mass']
        self._galprop_dtypes_to_allocate = np.dtype([('stellar_mass', 'f4')])
        self.list_of_haloprops_needed = [prim_haloprop_key]
        
        self._prim_haloprop_key = prim_haloprop_key
        self.stellar_mass_function = stellar_mass_function
        self._Lbox = Lbox
        self.param_dict = {'scatter':scatter}
        
        #select stellar mass function
        if self.stellar_mass_function == 'LiWhite_2009':
            self.sdss_phi = stellar_mass_functions.LiWhite_2009_phi()
        elif self.stellar_mass_function == 'Yang_2012':
            self.sdss_phi = stellar_mass_functions.Yang_2012_phi()
    
    def _assign_stellar_mass(self,  **kwargs):
        """
        assign stellar mass based on halo mass rank
        """
        
        table = kwargs['table']
        
        #sort haloes by prim_halo_prop
        halo_sort_inds = np.argsort(table[self._prim_haloprop_key])
        
        #remove haloes that are not to be populated
        if 'remove' in table.keys():
            mask = (table['remove'][halo_sort_inds]==0)
        else:
            mask = np.array([True]*len(table))
        
        N_haloes_to_fill = np.sum(mask)
        
        #MC realization of stellar masses
        mstar = self.sample_stellar_mass_func(N_haloes_to_fill, self._Lbox**3.0)
        #mstar = np.sort(mstar)
        galaxy_sort_inds = np.argsort(mstar)
        
        if self.param_dict['scatter']>0.0:
            #add scatter in steps.  choose step size
            sub_scatter = self.param_dict['scatter']/5.0
            
            #find where the log mass changes by the scatter
            mass = np.log10(mstar)
            i = (mass[galaxy_sort_inds]/sub_scatter).astype('int')
            dummy, inds = np.unique(i, return_index=True)
            
            #calculate the change in index corresponding to the change in mass
            dn = np.log10(np.diff(inds))
            n = np.log10((inds[1:]+inds[:-1])/2.0)
            dn[0] = 0.0
            
            fn = interp1d(n, dn, fill_value='extrapolate')
            
            sigma = 10.0**fn(np.log10(np.arange(0,len(table[mask]))))
            sigma = 1.0*sigma/len(table)
            
            for i in range(0, 5**2):
                galaxy_sort_inds = scatter_ranks(galaxy_sort_inds, sigma)
        
        #assign stellar masses based on rank
        table['stellar_mass'] = -99
        table['stellar_mass'][halo_sort_inds[mask]] = mstar[galaxy_sort_inds]
    
    def sample_stellar_mass_func(self, N, V):
        """
        Draw stellar masses from a stellar mass function ``N`` times 
        in a specified volume, ``V``.
        
        Parameters
        ----------
        N : int
            integer number of times to sample the stellar mass function
        
        V : float
            volume in h^-3 Mpc^3
        
        Returns
        -------
        mstar : numpy.array
            array of stellar masses
        """
        
        f = self.cumulative_mass_function(1, 10.0**12.0,
                                          self.stellar_mass_function, V)
        
        #calculate the minimum stellar mass
        min_mstar = f(np.log10(N))
        
        f = self.cumulative_mass_function(10.0**min_mstar, 10.0**12.0,
                                          self.stellar_mass_function, V)
        
        #sample mass function to get stellar masses
        s = np.log10(np.random.uniform(0.0, N, N))
        mstar = 10**f(s)
        
        return mstar
    
    def cumulative_mass_function(self, min, max, stellar_mass_function, volume, dm_log = 0.01):
        """
        numerically integrate a differential stellar mass function
        and interpolate result
        
        Parameters
        ----------
        min : float
            minimum stellar mass
        
        max : float
            maximum stellar mass
        
        stellar_mass_function : function object
            number density per dex as a function of stellar mass
        
        volume : float
            volume in h^-3Mpc^3
        
        Return
        ------
        N(>m) : function object
           cumulative stellar mass function
        """
        
        #integrate to get cumulative distribution
        mstar_sample = np.arange(np.log10(min), np.log10(max), dm_log)
        mstar_sample = 10**mstar_sample[::-1]
        
        dN = self.sdss_phi(mstar_sample)*self.sdss_phi.f(mstar_sample) * (volume)
        cumulative_dist = cumtrapz(dN, dx = dm_log, initial=0.0)
        #dN = self.sdss_phi(mstar_sample) * (volume) * (dm_log)
        #cumulative_dist = np.cumsum(dN)
        
        #interpolate mas function
        f = interp1d(np.log10(cumulative_dist), np.log10(mstar_sample), fill_value='extrapolate', kind='linear')
        
        return f


class ParamSmHm(object):
    """
    class to assign stellar mass based on parameterized dependence on halo mass
    """
    def __init__(self, prim_haloprop_key = 'halo_mpeak', log_scatter = 0.0, **kwargs):
        """
        Parameters
        ----------
        prim_haloprop_key : string
            key indicating halo property for which the completeness is modeled
        
        scatter : float
            fixed log-normal scatter in stellar mass
        """
        
        if 'redshift' in list(kwargs.keys()):
            self.redshift = kwargs['redshift']
        else:
            self.redshift=0.0
        
        self._mock_generation_calling_sequence = ['assign_stellar_mass']
        self._galprop_dtypes_to_allocate = np.dtype([('stellar_mass', 'f4')])
        self.list_of_haloprops_needed = [prim_haloprop_key]
        
        self._prim_haloprop_key = prim_haloprop_key
        self.param_dict = self.retrieve_default_param_dict(log_scatter)
        
    
    def mean_stellar_mass(self, **kwargs):
        """ 
        Return the mean stellar mass as a function of the input table.
        
        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed.
        
        scatter : float, optional
        
        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument must be passed.
        
        Returns
        -------
        mstar : array_like
            Array containing stellar masses living in the input table.
        """
        
        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            x = kwargs['table'][self._prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            x = kwargs['prim_haloprop']
        else:
            raise KeyError("Must pass one of the following keyword arguments to mean_occupation:\n"
                "``table`` or ``prim_haloprop``.")
        
        if  'log_scatter' in list(kwargs.keys()):
            sigma = kwargs['log_scatter']
        else:
            sigma =  self.param_dict['sigma']
        
        log_y0 = self.exp_model(sigma, self.param_dict['x_1'], self.param_dict['alpha_1'], self.param_dict['b_1'], self.param_dict['a_1'])
        log_x0 = self.exp_model(sigma, self.param_dict['x_2'], self.param_dict['alpha_2'], self.param_dict['b_2'], self.param_dict['a_2'])
        alpha = self.exp_model(sigma, self.param_dict['x_3'], self.param_dict['alpha_3'], self.param_dict['b_3'], self.param_dict['a_3'])
        beta = self.exp_model(sigma, self.param_dict['x_4'], self.param_dict['alpha_4'], self.param_dict['b_4'], self.param_dict['a_4'])
        
        y0 = 10.0**log_y0
        x0 = 10.0**log_x0
        
        return (2.0*y0/x0)*x*((x/x0)**(alpha) + (x/x0)**(beta))**(-1)
    
    def exp_model(self, x, x0, alpha, b, a=1.0):
        return a*np.exp((x/x0)**alpha) + b
    
    def assign_stellar_mass(self,  **kwargs):
        """
        assign stellar mass based on halo mass
        """
        
        table = kwargs['table']
        Ngal = len(table)
        
        mean_log_mstar = np.log10(self.mean_stellar_mass(**kwargs))
        
        #add scatter
        log_mstar =  mean_log_mstar + norm.rvs(loc=0, scale=self.param_dict['sigma'], size=Ngal)
        
        table['stellar_mass'] = 10.0**log_mstar
        
    
    def retrieve_default_param_dict(self, log_scatter=0.0):
        """ 
        Method returns a dictionary of all model parameters
        
        Returns
        -------
        d : dict
            Dictionary containing parameter values.
        """
        
        if self._prim_haloprop_key is 'halo_mpeak':
            d = {'x_1': 0.867,
                 'alpha_1':2.26,
                 'b_1':8.92,
                 'a_1':1.0,
                 'x_2': 0.925,
                 'alpha_2':2.09,
                 'b_2':10.63,
                 'a_2':1.0,
                 'x_3': 1.51,
                 'alpha_3':1.61,
                 'b_3':-2.16,
                 'a_3':1.0,
                 'x_4': 0.538,
                 'alpha_4':2.38,
                 'b_4':-0.395,
                 'a_4':1.0,
                 'sigma':log_scatter
                }
        elif self._prim_haloprop_key is 'halo_vpeak':
            d = {'x_1': 1.56066,
                 'alpha_1':1.69450,
                 'b_1':8.943279,
                 'a_1':1,
                 'x_2': 3.34455,
                 'alpha_2':1.60497,
                 'b_2':1.17665,
                 'a_2':1,
                 'x_3': 1.75682,
                 'alpha_3':0.97744,
                 'b_3':-6.9654,
                 'a_3':1,
                 'x_4': 0.37405,
                 'alpha_4':2.0464,
                 'b_4':-1.2588441,
                 'a_4':1,
                 'sigma':log_scatter
                }
        return d
