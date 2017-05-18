"""
Module containing classes used in SHAM models that paramaterize the SMHM relation
"""

from __future__ import (division, print_function, absolute_import)

import numpy as np
from scipy.stats import norm
from astropy.table import Table
from astropy.io import ascii
from scipy.interpolate import interp1d
from halotools.empirical_models import model_helpers as model_helpers
from warnings import warn


__all__ = ['MosterSmHm13', 'BehrooziSmHm13', 'Yang12SmHm',
           'Bell_to_Blanton', 'Kauffmann_to_Blanton', 'Moustakas_to_Blanton','Guo_to_Blanton','Yang_correction']

class MosterSmHm13(object):
    """ 
    Stellar-to-halo-mass relation based on Moster et al. (2013), arXiv:1205.5807
    """
    
    def __init__(self, prim_haloprop_key = 'halo_mpeak', acc_scale_key = 'halo_acc_scale',  **kwargs):
        """
        Parameters
        ----------
        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing stellar mass.
        
        acc_scale_key : string, optional
            String givig the column name of the halo property indicating the accretion scale for sub-haloes
        """
        
        if 'redshift' in list(kwargs.keys()):
            self.redshift = kwargs['redshift']
        else:
            self.redshift=0.0
        
        kwargs['prim_haloprop_key'] = prim_haloprop_key
        self.prim_haloprop_key = prim_haloprop_key
        kwargs['acc_scale_key'] = acc_scale_key
        self.acc_scale_key = acc_scale_key
        
        self._mock_generation_calling_sequence = ['_assign_stellar_mass']
        self._galprop_dtypes_to_allocate = np.dtype([('stellar_mass', 'f4')])
        
        self._methods_to_inherit = ['_assign_stellar_mass','mean_stellar_mass']
        
        self.list_of_haloprops_needed = [prim_haloprop_key, acc_scale_key]
        self.param_dict = self.retrieve_default_param_dict()
        self.littleh = 0.701
        
        self.publications = ['arXiv:1205.5807']
    
    def mean_stellar_mass(self, **kwargs):
        """ 
        Return the stellar mass of a central galaxy as a function
        of the input table.
        
        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed.
        
        halo_acc_scale : array, optional
            Array of halo accretion scale
        
        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument must be passed.
        
        redshift : float or array, optional
            Redshift of the halo hosting the galaxy.
            If passing an array, must be of the same length as
            the ``prim_haloprop`` or ``table`` argument.  Defualt is z=0.0.
        
        Returns
        -------
        mstar : array_like
            Array containing stellar masses living in the input table.
        """
        
        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
            acc_scale = kwargs['table'][self.acc_scale_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = kwargs['prim_haloprop']
            acc_scale = kwargs['halo_acc_scale']
        else:
            msg = ("Must pass one of the following keyword arguments to mean_occupation:\n"
                   "``table`` or ``prim_haloprop`` and ``halo_acc_scale``")
            raise KeyError(msg)
        
        if 'redshift' in list(kwargs.keys()):
            redshift = kwargs['redshift']
        else:
            redshift=0.0
        
        mass = mass/self.littleh
        
        # compute the parameter values that apply to at a given scale factor
        a = np.atleast_1d(acc_scale)
        a_max = 1.0/(1.0+redshift)
        if np.any(a>a_max):
            msg = ("Accretion scale is later than the redshift.")
            warn(msg)
        
        m1 = self.param_dict['m10'] + self.param_dict['m11']*(1-a)
        n = self.param_dict['n10'] + self.param_dict['n11']*(1-a)
        beta = self.param_dict['beta10'] + self.param_dict['beta11']*(1-a)
        gamma = self.param_dict['gamma10'] + self.param_dict['gamma11']*(1-a)
        
        # Calculate each term contributing to Eqn 2
        norm = 2.*n*mass
        m_by_m1 = mass/(10.**m1)
        denom_term1 = m_by_m1**(-beta)
        denom_term2 = m_by_m1**gamma
        
        mstar = norm / (denom_term1 + denom_term2)
        
        return mstar*self.littleh**2
    
    def _assign_stellar_mass(self,  **kwargs):
        """
        assign stellar mass
        """
        
        table = kwargs['table']
        Ngal = len(table)
        
        mean_log_mstar = np.log10(self.mean_stellar_mass(**kwargs))
        
        #add scatter
        log_mstar =  mean_log_mstar + norm.rvs(loc=0, scale=self.param_dict['log_scatter'], size=Ngal)
        
        table['stellar_mass'] = 10.0**log_mstar
    
    def retrieve_default_param_dict(self):
        """ 
        Method returns a dictionary of all model parameters
        set to the values in Table 1 of Moster et al. (2013).
        
        Returns
        -------
        d : dict
            Dictionary containing parameter values.
        """
        
        # All calculations are done internally using the same h=0.701 units
        # as in Moster et al. (2013), so the parameter values here are
        # the same as in Table 1, even though the 
        # mean_stellar_mass method accepts and returns arguments in h=1 units.
        
        d = {
        'm10': 11.590,
        'm11': 1.195,
        'n10': 0.0351,
        'n11': -0.0247,
        'beta10': 1.376,
        'beta11': -0.826,
        'gamma10': 0.608,
        'gamma11': 0.329,
        'log_scatter': 0.2
        }
        
        return d


class BehrooziSmHm13(object):
    """ 
    Stellar-to-halo-mass relation based on Behroozi et al. (2013), arXiv:1207.6105
    """

    def __init__(self, prim_haloprop_key='halo_mpeak', acc_scale_key = 'halo_acc_scale', **kwargs):
        """
        Parameters
        ----------
        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing stellar mass.
            Default is set in the `~halotools.empirical_models.model_defaults` module.
        
        acc_scale_key : string, optional
            String givig the column name of the halo property indicating the accretion scale for sub-haloes
        """
        
        if 'redshift' in list(kwargs.keys()):
            self.redshift = kwargs['redshift']
        else:
            self.redshift=0.0
        
        kwargs['prim_haloprop_key'] = prim_haloprop_key
        self.prim_haloprop_key = prim_haloprop_key
        kwargs['acc_scale_key'] = acc_scale_key
        self.acc_scale_key = acc_scale_key
        
        self._mock_generation_calling_sequence = ['_assign_stellar_mass']
        self._galprop_dtypes_to_allocate = np.dtype([('stellar_mass', 'f4')])
        
        self._methods_to_inherit = ['_assign_stellar_mass','mean_stellar_mass']
        
        self.list_of_haloprops_needed = [prim_haloprop_key, acc_scale_key]
        self.param_dict = self.retrieve_default_param_dict()
        self.littleh = 0.7
        
        self.publications = ['arXiv:1207.6105']
    
    def mean_stellar_mass(self, **kwargs):
        """ 
        Return the stellar mass of a central galaxy as a function
        of the input table.
        
        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed.
        
        halo_acc_scale : array, optional
            Array of halo accretion scale
        
        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument must be passed.
        
        redshift : float or array, optional
            Redshift of the halo hosting the galaxy.
            Default is set in `~halotools.sim_manager.sim_defaults`.
            If passing an array, must be of the same length as
            the ``prim_haloprop`` or ``table`` argument.
        
        Returns
        -------
        mstar : array_like
            Array containing stellar masses living in the input table.
        """
        
        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
            acc_scale = kwargs['table'][self.acc_scale_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = kwargs['prim_haloprop']
            acc_scale = kwargs['halo_acc_scale']
        else:
            msg = ("Must pass one of the following keyword arguments to mean_occupation:\n"
                   "``table`` or ``prim_haloprop`` and ``halo_acc_scale``")
            raise KeyError(msg)
        
        if 'redshift' in list(kwargs.keys()):
            redshift = kwargs['redshift']
        else:
            redshift=0.0
        
        mass = np.atleast_1d(mass)
        mass = mass/self.littleh
        
        a = np.atleast_1d(acc_scale)
        a_max = 1.0/(1.0+redshift)
        if np.any(a>a_max):
            msg = ("Accretion scale is later than the redshift.")
            warn(msg)
        z = 1.0/a - 1.0
        
        m_10 = self.param_dict['m_10']
        m_1a = self.param_dict['m_1a']
        m_1z = self.param_dict['m_1z']
        epsilon_0 = self.param_dict['epsilon_0']
        epsilon_a = self.param_dict['epsilon_a']
        epsilon_z = self.param_dict['epsilon_z']
        epsilon_a2 = self.param_dict['epsilon_a2']
        alpha_0 = self.param_dict['alpha_0']
        alpha_a = self.param_dict['alpha_a']
        delta_0 = self.param_dict['delta_0']
        delta_a = self.param_dict['delta_a']
        delta_z = self.param_dict['delta_z']
        gamma_0 = self.param_dict['gamma_0']
        gamma_a = self.param_dict['gamma_a']
        gamma_z = self.param_dict['gamma_z']
        
        nu = np.exp(-4.0*a**2.0)
        logM1 = m_10 + (m_1a*(a-1.0) + m_1z*z)*nu
        logepsilon = epsilon_0 + (epsilon_a*(a-1.0) + epsilon_z*z)*nu + epsilon_a2*(a-1.0)
        alpha = alpha_0 + (alpha_a*(a-1.0))*nu
        delta = delta_0 + (delta_a*(a-1.0) + delta_z*z)*nu
        gamma = gamma_0 + (gamma_a*(a-1.0) + gamma_z*z)*nu
        
        M1=10.0**logM1
        epsilon = 10.0**logepsilon
        
        def f(x):
            return -1.0 * np.log10(10.0**(alpha*x)+1.0) +\
                   delta*(((np.log10(1.0+np.exp(x)))**gamma)/(1.0+np.exp(10.0**(-x))))
        
        logmstar = np.log10(epsilon * M1) + f(np.log10(mass/M1))- f(0.0)
        mstar = 10.0**logmstar
        
        return mstar*self.littleh**2
    
    def _assign_stellar_mass(self,  **kwargs):
        """
        assign stellar mass
        """
        
        table = kwargs['table']
        Ngal = len(table)
        
        mean_log_mstar = np.log10(self.mean_stellar_mass(**kwargs))
        
        #add scatter
        log_mstar =  mean_log_mstar + norm.rvs(loc=0, scale=self.param_dict['log_scatter'], size=Ngal)
        
        table['stellar_mass'] = 10.0**log_mstar
    
    def retrieve_default_param_dict(self):
        """ 
        Method returns a dictionary of all model parameters
        set to the values in section 5 of Behroozi et al. (2013).
        
        Returns
        -------
        d : dict
            Dictionary containing parameter values.
        """
        
        # All calculations are done internally using the same h=0.7 units
        # as in Behroozi et al. (2013), so the parameter values here are
        # the same as in section 5, even though the 
        # mean_stellar_mass method accepts and returns arguments in h=1 units.
        
        d = {
            'm_10': 11.514,
            'm_1a': -1.793,
            'm_1z': -0.251,
            'epsilon_0': -1.777,
            'epsilon_a': -0.006,
            'epsilon_z': 0.000,
            'epsilon_a2': 0.119,
            'alpha_0': -1.412,
            'alpha_a': 0.731,
            'delta_0': 3.508,
            'delta_a': 2.608,
            'delta_z': -0.043,
            'gamma_0': 0.316,
            'gamma_a': 1.319,
            'gamma_z': 0.279,
            'm_hicl0': 12.515,
            'm_hicla': -2.503,
            'rho_0.5':0.799,
            'log_scatter': 0.218
        }
        
        return d


class Yang12SmHm(object):
    """ 
    Stellar-to-halo-mass relation based on Yang et al. (2012), arXiv:1110.1420
    """
    
    def __init__(self, prim_haloprop_key='halo_mpeak', acc_scale_key='halo_acc_scale', **kwargs):
        """
        Parameters
        ----------
        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing stellar mass.
            Default is set in the `~halotools.empirical_models.model_defaults` module.
        
        acc_scale_key : string, optional
            String givig the column name of the halo property indicating the 
            accretion scale for sub-haloes
        """
        
        if 'redshift' in list(kwargs.keys()):
            self.redshift = kwargs['redshift']
        else:
            self.redshift=0.0
        
        kwargs['prim_haloprop_key'] = prim_haloprop_key
        
        #default fit parameters
        if 'fit_type' in list(kwargs.keys()):
            self._fit_type = kwargs['fit_type']
        else:
            self._fit_type = 'FIT-CSMF'
        if 'fit_cosmo' in list(kwargs.keys()):
            self._fit_cosmo = kwargs['fit_cosmo']
        else:
            self._fit_cosmo = 'WMAP7'
        if 'fit_smf' in list(kwargs.keys()):
            self._fit_type = kwargs['fit_smf']
        else:
            self._fit_smf = 'SMF2'
        
        kwargs['prim_haloprop_key'] = prim_haloprop_key
        self.prim_haloprop_key = prim_haloprop_key
        kwargs['acc_scale_key'] = acc_scale_key
        self.acc_scale_key = acc_scale_key
        
        self._mock_generation_calling_sequence = ['_assign_stellar_mass']
        self._galprop_dtypes_to_allocate = np.dtype([('stellar_mass', 'f4')])
        
        self._methods_to_inherit = ['_assign_stellar_mass','mean_stellar_mass']
        
        self.list_of_haloprops_needed = [prim_haloprop_key, acc_scale_key]
        self.param_dict = self.retrieve_default_param_dict()
        self.littleh = 1.0
        
        self.publications = ['arXiv:1110.1420']
    
    def mean_stellar_mass(self, **kwargs):
        """ 
        Return the stellar mass of a central galaxy as a function
        of the input table.
        
        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed.
        
        halo_acc_scale : array, optional
            Array of halo accretion scale
        
        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument must be passed.
        
        redshift : float or array, optional
            Redshift of the halo hosting the galaxy.
            Default is set in `~halotools.sim_manager.sim_defaults`.
            If passing an array, must be of the same length as
            the ``prim_haloprop`` or ``table`` argument.
        
        Returns
        -------
        mstar : array_like
            Array containing stellar masses living in the input table.
        """
        
        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
            acc_scale = kwargs['table'][self.acc_scale_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = kwargs['prim_haloprop']
            acc_scale = kwargs['halo_acc_scale']
        else:
            msg = ("Must pass one of the following keyword arguments to mean_occupation:\n"
                   "``table`` or ``prim_haloprop`` and ``halo_acc_scale``")
            raise KeyError(msg)
        
        if 'redshift' in list(kwargs.keys()):
            redshift = kwargs['redshift']
        else:
            redshift = 0.0
        
        mass = mass/self.littleh
        
        z_now = redshift
        z_acc = 1.0/acc_scale - 1.0
        
        a_max = 1.0/(1.0+z_now)
        if np.any(acc_scale>a_max):
            msg = ("Accretion scale is later than the redshift.")
            warn(msg)
        
        #calculate for centrals
        z = z_now
        
        #equations 40
        logm0 = self.param_dict['m0'] + self.param_dict['gamma_1']*z
        logm1 = self.param_dict['m1'] + self.param_dict['gamma_2']*z
        alpha = self.param_dict['alpha'] + self.param_dict['gamma_3']*z
        logbeta = np.log10(self.param_dict['beta']) +\
                  self.param_dict['gamma_4']*z +\
                  self.param_dict['gamma_5']*z**2.0
        c = self.param_dict['c']
        
        m0=10.0**logm0
        m1=10.0**logm1
        beta = 10.0**logbeta
        
        #equation 17
        numerator = (mass/m1)**(alpha+beta)
        denominator = (1.0+mass/m1)**beta
        mstar_1 = m0*(numerator/denominator)
        
        #calculate for satellites
        z = z_acc
        
        #equations 40
        logm0 = self.param_dict['m0'] + self.param_dict['gamma_1']*z
        logm1 = self.param_dict['m1'] + self.param_dict['gamma_2']*z
        alpha = self.param_dict['alpha'] + self.param_dict['gamma_3']*z
        logbeta = np.log10(self.param_dict['beta']) +\
                  self.param_dict['gamma_4']*z +\
                  self.param_dict['gamma_5']*z**2.0
        c = self.param_dict['c']
        
        m0=10.0**logm0
        m1=10.0**logm1
        beta = 10.0**logbeta
        
        #equation 17
        numerator = (mass/m1)**(alpha+beta)
        denominator = (1.0+mass/m1)**beta
        mstar_2 = m0*(numerator/denominator)
        
        #equation 22
        mstar = (1.0-c)*mstar_2 + c*mstar_1
        
        return mstar*(self.littleh**2)
    
    def _assign_stellar_mass(self,  **kwargs):
        """
        assign stellar mass
        """
        
        table = kwargs['table']
        Ngal = len(table)
        
        mean_log_mstar = np.log10(self.mean_stellar_mass(**kwargs))
        
        #add scatter
        log_mstar =  mean_log_mstar + norm.rvs(loc=0, scale=self.param_dict['log_scatter'], size=Ngal)
        
        table['stellar_mass'] = 10.0**log_mstar
    
    def retrieve_default_param_dict(self):
        """ 
        Method returns a dictionary of all model parameters
        
        Returns
        -------
        d : dict
            Dictionary containing parameter values.
        """
        
        fit_type = self._fit_type
        fit_cosmo = self._fit_cosmo
        fit_smf = self._fit_smf
        
        #taken table 4 in Yang et al. (2012)
        if (fit_type == 'FIT0') & (fit_smf=='SMF2') & (fit_cosmo=='WMAP7'):
            d = {
            'm0': 10.45,
            'm1': 11.17,
            'alpha':0.25,
            'beta':3.23,
            'gamma_1':-0.98,
            'gamma_2':-0.25,
            'gamma_3':0.42,
            'gamma_4':-0.07,
            'gamma_5':0.01,
            'pt':0.0,
            'c':0.0,
            'log_scatter': 0.173
            }
        elif (fit_type == 'FIT1') & (fit_smf=='SMF2') & (fit_cosmo=='WMAP7'):
            d = {
            'm0': 10.45,
            'm1': 11.43,
            'alpha':0.27,
            'beta':3.85,
            'gamma_1':-0.79,
            'gamma_2':-0.24,
            'gamma_3':0.38,
            'gamma_4':-0.16,
            'gamma_5':0.02,
            'pt':np.inf,
            'c':1.0,
            'log_scatter': 0.173
            }
        elif (fit_type == 'FIT-2PCF') & (fit_smf=='SMF2') & (fit_cosmo=='WMAP7'):
            d = {
            'm0': 10.46,
            'm1': 11.21,
            'alpha':0.25,
            'beta':4.0,
            'gamma_1':-0.93,
            'gamma_2':-0.26,
            'gamma_3':0.39,
            'gamma_4':-0.15,
            'gamma_5':0.04,
            'pt':1.18,
            'c':0.91,
            'log_scatter': 0.173
            }
        elif (fit_type == 'FIT-CSMF') & (fit_smf=='SMF2') & (fit_cosmo=='WMAP7'):
            d = {
            'm0': 10.36,
            'm1': 11.06,
            'alpha':0.27,
            'beta':4.34,
            'gamma_1':-0.96,
            'gamma_2':-0.23,
            'gamma_3':0.41,
            'gamma_4':-0.11,
            'gamma_5':0.01,
            'pt':0.88,
            'c':0.98,
            'log_scatter': 0.173
            }
        else:
            raise ValueError("default fit parameters not recognized.")
        
        return d


class Bell_to_Blanton(object):
    """
    class to convert Bell et al. (2003) to Blanton \& Roweis (2007) stellar masses 
    """
    def __init__(self,
                 haloprop_keys = [],
                 **kwargs):
        """
        Parameters
        ----------
        """
        
        self._mock_generation_calling_sequence = ['assign_new_stellar_masses']
        self._galprop_dtypes_to_allocate = np.dtype([])
        self.list_of_haloprops_needed = haloprop_keys
        
        self.publications = ['arXiv:0901.0706']
        self.param_dict = self.retrieve_default_param_dict()
    
    def assign_new_stellar_masses(self, **kwargs):
        """
        convert Bell et al. stellar masses to Blanton et al. stellar masses
        """
        table = kwargs['table']
        
        new_mstar = self.convert_stellar_mass(table=table)
        table['stellar_mass'] = new_mstar
        
    def retrieve_default_param_dict(self):
        """ 
        Method returns a dictionary of function parameters
        for eq. A2 in Li \& White (2011).
        
        Returns
        -------
        d : dict
            Dictionary containing parameter values.
        """
        
        d = {'a1':2.0,
             'a2':-0.043,
             'a3':-0.045,
             'a4':0.0032,
             'a5':-2.1*10**(-5.0),
             }
        
        return d
    
    def convert_stellar_mass(self, **kwargs):
        """
        conversion function from Bell et al. (2003) to Blanton \& Roweis (2007) 
        stellar masses using eq. A2 in Li \& White (2011).
        
        Parameters
        ----------
        stellar_mass : array_like
            Bell et al. stellar mass(es) in $h^{-2} M_{\odot}$
        
        Returns
        -------
        stellar_mass : array_like
            Blanton \& Roweis stellar mass(es) in $h^{-2} M_{\odot}$
        """
        
        if 'table' in list(kwargs.keys()):
            m = np.log10(kwargs['table']['stellar_mass'])
        elif 'stellar_mass' in list(kwargs.keys()):
            m = np.log10(kwargs['stellar_mass'])
        
        a1 = self.param_dict['a1']
        a2 = self.param_dict['a2']
        a3 = self.param_dict['a3']
        a4 = self.param_dict['a4']
        a5 = self.param_dict['a5']
        
        #invert conversion function
        m_blanton = np.linspace(5.0,13.0,1000) #blanton stellar masses
        delta_m = a1 + a2*m_blanton + a3*m_blanton**2 +\
                  a4*m_blanton**3 + a5*m_blanton**4
        m_bell = m_blanton + delta_m
        
        #interpolate inversion
        f_conv = interp1d(m_bell, delta_m, fill_value='extrapolate')
        
        return 10.0**(m - f_conv(m))

class Kauffmann_to_Blanton(object):
    """
    class to convert Kauffmann (2003) to Blanton \& Roweis (2007) stellar masses 
    """
    def __init__(self,
                 haloprop_keys = [],
                 **kwargs):
        """
        Parameters
        ----------
        """
        
        self._mock_generation_calling_sequence = ['assign_new_stellar_masses']
        self._galprop_dtypes_to_allocate = np.dtype([])
        self.list_of_haloprops_needed = haloprop_keys
        
        self.publications = ['arXiv:0901.0706']
        self.param_dict = self.retrieve_default_param_dict()
    
    def assign_new_stellar_masses(self, **kwargs):
        """
        convert Kauffmann et al. stellar masses to Blanton et al. stellar masses
        """
        table = kwargs['table']
        
        new_mstar = self.convert_stellar_mass(table=table)
        table['stellar_mass'] = new_mstar
        
    def retrieve_default_param_dict(self):
        """ 
        Method returns a dictionary of function parameters
        for eq. A1 in Li \& White (2011).
        
        Returns
        -------
        d : dict
            Dictionary containing parameter values.
        """
        
        d = {'a1':0.0256,
             'a2':0.0478,
             'a3':9.73,
             'a4':0.417
             }
        
        return d
    
    def convert_stellar_mass(self, **kwargs):
        """
        conversion function from Bell et al. (2003) to Blanton \& Roweis (2007) 
        stellar masses using eq. A1 in Li \& White (2011).
        
        Parameters
        ----------
        stellar_mass : array_like
            Bell et al. stellar mass(es) in $h^{-2} M_{\odot}$
        
        Returns
        -------
        stellar_mass : array_like
            Blanton \& Roweis stellar mass(es) in $h^{-2} M_{\odot}$
        """
        
        if 'table' in list(kwargs.keys()):
            m = np.log10(kwargs['table']['stellar_mass'])
        elif 'stellar_mass' in list(kwargs.keys()):
            m = np.log10(kwargs['stellar_mass'])
        
        a1 = self.param_dict['a1']
        a2 = self.param_dict['a2']
        a3 = self.param_dict['a3']
        a4 = self.param_dict['a4']
        
        #invert conversion function
        m_blanton = np.linspace(5.0,13.0,1000) #blanton stellar masses
        delta_m = a1 + a2*np.tanh((m_blanton-a3)/a4)
        m_kauff = m_blanton + delta_m
        
        #interpolate inversion
        f_conv = interp1d(m_kauff, delta_m, fill_value='extrapolate')
        
        return 10.0**(m - f_conv(m))

class Moustakas_to_Blanton(object):
    """
    class to convert Moustakas et al. (2013) to Blanton \& Roweis (2007) stellar masses 
    """
    def __init__(self,
                 haloprop_keys = [],
                 **kwargs):
        """
        Parameters
        ----------
        """
        
        self._mock_generation_calling_sequence = ['assign_new_stellar_masses']
        self._galprop_dtypes_to_allocate = np.dtype([])
        self.list_of_haloprops_needed = haloprop_keys
        
        self.publications = ['arXiv:0901.0706']
        self.param_dict = self.retrieve_default_param_dict()
        self.littleh = 0.7
        
        
    def assign_new_stellar_masses(self, **kwargs):
        """
        convert Moustakas et al. stellar masses to Blanton et al. stellar masses
        """
        table = kwargs['table']
        
        new_mstar = self.convert_stellar_mass(table=table)
        table['stellar_mass'] = new_mstar
        
    def retrieve_default_param_dict(self):
        """ 
        Method returns a dictionary of function parameters
        for eq. A1 in Li \& White in order to reproduce the 
        top panel in Moustakas et al. (2013) upper left panel of figure A1.
        
        Returns
        -------
        d : dict
            Dictionary containing parameter values.
        """
        
        d = {'a1':0.0056,
             'a2':-0.0978,
             'a3':9.73+0.8,
             'a4':0.817
             }
        
        return d
    
    def convert_stellar_mass(self, **kwargs):
        """
        conversion function from Moustakas et al. (2013) to Blanton \& Roweis (2007) 
        stellar masses using eq. A1 in Li \& White (2011).
        
        Parameters
        ----------
        stellar_mass : array_like
            Moustakas et al. stellar mass(es) in $h^{-2} M_{\odot}$
        
        Returns
        -------
        stellar_mass : array_like
            Blanton \& Roweis stellar mass(es) in $h^{-2} M_{\odot}$
        """
        
        if 'table' in list(kwargs.keys()):
            m = kwargs['table']['stellar_mass']
        elif 'stellar_mass' in list(kwargs.keys()):
            m = kwargs['stellar_mass']
        
        m = m/self.littleh**2.0
        m = np.log10(m)
        
        a1 = self.param_dict['a1']
        a2 = self.param_dict['a2']
        a3 = self.param_dict['a3']
        a4 = self.param_dict['a4']
        
        delta_m = a1 + a2*np.tanh((m-a3)/a4)
        
        return 10.0**(m + delta_m)*self.littleh**2.0


class Guo_to_Blanton(object):
    """
    Undo the tranformation suggested by Guo et al. (2010) which transforms the stellar 
    masses based on the SDSS r-band Petrosian magnitudes to ones based on 
    SDSS r-band model magnitudes. 
    """
    def __init__(self,
                 haloprop_keys = [],
                 **kwargs):
        """
        Parameters
        ----------
        """
        
        self._mock_generation_calling_sequence = ['assign_new_stellar_masses']
        self._galprop_dtypes_to_allocate = np.dtype([])
        self.list_of_haloprops_needed = haloprop_keys
        
        self.param_dict = self.retrieve_default_param_dict()
    
    def assign_new_stellar_masses(self, **kwargs):
        """
        assign new stellar masses
        """
        table = kwargs['table']
        
        new_mstar = self.convert_stellar_mass(table=table)
        table['stellar_mass'] = new_mstar
        
    def retrieve_default_param_dict(self):
        """ 
        Method returns a dictionary of parameters
        
        Returns
        -------
        d : dict
            Dictionary containing parameter values.
        """
        
        d = {'f':0.9
             }
        
        return d
    
    def convert_stellar_mass(self, **kwargs):
        """
        multiply stellar mass by a constant factor
        
        Parameters
        ----------
        stellar_mass : array_like
            Bell et al. stellar mass(es) in $h^{-2} M_{\odot}$
        
        Returns
        -------
        stellar_mass : array_like
            Blanton \& Roweis stellar mass(es) in $h^{-2} M_{\odot}$
        """
        
        if 'table' in list(kwargs.keys()):
            m = np.log10(kwargs['table']['stellar_mass'])
        elif 'stellar_mass' in list(kwargs.keys()):
            m = np.log10(kwargs['stellar_mass'])
        
        f = self.param_dict['f']
        
        return f*10.0**(m)


class Yang_correction(object):
    """
    correct Yang et al. (2012) stellar masses by multiplying by a factor of h=0.7 
    """
    def __init__(self,
                 haloprop_keys = [],
                 **kwargs):
        """
        Parameters
        ----------
        """
        
        self._mock_generation_calling_sequence = ['correct_stellar_masses']
        self._galprop_dtypes_to_allocate = np.dtype([])
        self.list_of_haloprops_needed = haloprop_keys
        
        self.param_dict = self.retrieve_default_param_dict()
    
    def correct_stellar_masses(self, **kwargs):
        """
        assign new stellar masses
        """
        table = kwargs['table']
        
        new_mstar = self.convert_stellar_mass(table=table)
        table['stellar_mass'] = new_mstar
        
    def retrieve_default_param_dict(self):
        """ 
        Method returns a dictionary of parameters
        
        Returns
        -------
        d : dict
            Dictionary containing parameter values.
        """
        
        d = {'f':0.7
             }
        
        return d
    
    def convert_stellar_mass(self, **kwargs):
        """
        multiply stellar mass by a constant factor
        
        Parameters
        ----------
        stellar_mass : array_like
            Bell et al. stellar mass(es) in $h^{-2} M_{\odot}$
        
        Returns
        -------
        stellar_mass : array_like
            Blanton \& Roweis stellar mass(es) in $h^{-2} M_{\odot}$
        """
        
        if 'table' in list(kwargs.keys()):
            m = np.log10(kwargs['table']['stellar_mass'])
        elif 'stellar_mass' in list(kwargs.keys()):
            m = np.log10(kwargs['stellar_mass'])
        
        f = self.param_dict['f']
        
        return f*10.0**(m)