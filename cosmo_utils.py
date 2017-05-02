"""
cosmological functons
"""

import numpy as np

from astropy.cosmology import FlatLambdaCDM
default_cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from astropy import constants as const
from astropy import units as u
from scipy.interpolate import interp1d

__all__ = ['critical_density','mean_density','delta_vir',
           'dynamical_time', 'lookback_time',
           'halo_mass_conversion']

def critical_density(z, cosmo='default'):
    """
    critical density of the universe
    
    Paramaters
    ----------
    z : array_like
        redshift
    
    cosmo : astropy.cosmology object
    
    Returns
    -------
    rho_c : numpy.array
        critical density of the universe at redshift z in g/cm^3
    """
    
    if cosmo=='default':
        cosmo = default_cosmo
    
    rho = (3.0*cosmo.H(z)**2)/(8.0*np.pi*const.G)
    rho = rho.to(u.g / u.cm**3)
    
    return rho


def mean_density(z, cosmo='default'):
    """
    mean density of the universe
    
    Paramaters
    ----------
    z : array_like
        redshift
    
    cosmo : astropy.cosmology object
    
    Returns
    -------
    rho_b : numpy.array
         mean density of the universe at redshift z in g/cm^3
    """
    
    if cosmo=='default':
        cosmo = default_cosmo
    
    scale_factor = 1.0/(1.0+z)
    
    rho = (3.0/(8.0*np.pi*const.G))*(cosmo.H(z)**2)*(cosmo.Om(z)*scale_factor**(-3))
    rho = rho.to(u.g / u.cm**3)
    
    return rho


def delta_vir(z, cosmo='default', wrt='background'):
    """
    The average over-density of a collapsed dark matter halo. 
    fitting function from Bryan & Norman (1998)
    
    Paramaters
    ----------
    z : array_like
        redshift
    
    cosmo : astropy.cosmology object
    
    Returns
    -------
    delta_vir : numpy.array
        average density with respect to the mean density of the Universe
    """
    
    if cosmo=='default':
        cosmo = default_cosmo
    
    z = np.atleast_1d(z)
    
    x = cosmo.Om(z)-1.0
    
    if wrt=='critical':
        return (18.0*np.pi**2 + 82.0*x - 39.0*x**2)
    elif wrt=='background':
        return (18.0*np.pi**2 + 82.0*x - 39.0*x**2)/cosmo.Om(z)


def r_vir(m, z=0.0, cosmo='default'):
    """
    The virial radius of a collapsed dark matter halo
    
    Paramaters
    ----------
    m : array_like
        halo mass
    
    z : array_like
        redshift
    
    cosmo : astropy.cosmology object
    
    Returns
    -------
    r_vir : numpy.array
        virial radius in h^{-1}Mpc
    """
    
    if cosmo=='default':
        cosmo = default_cosmo
    
    z = np.atleast_1d(z)
    m = (np.atleast_1d(m)*u.Msun)/cosmo.h
    
    dvir = delta_vir(z, cosmo, wrt='background')
    
    r =  (m / ((4.0/3.0)*np.pi*dvir*mean_density(z, cosmo)))**(1.0/3.0)
    r = r.to(u.Mpc)*cosmo.h
    return r.value


def virial_halo_mass(m_h, c, delta_h=200, z=0.0, cosmo='default',
                         wrt='background'):
    """
    Convert halo mass to virial halo mass
    fitting function from Hu \& Kravtsov (2003).
    
    Parameters
    ----------
    m_h : array_like
        halo mass
    
    c : array_like
        concentration
    
    delta_h : float
        density contrast
    
    delta_vir : float, optional
        virial over-density wrt the mean density.  If given, cosmology is ignored 
        when calculating delta_vir.
        
    cosmo : astropy.cosmology object, optional
        cosmology used to calculated the virial over-density
    
    wrt : string
        halo over-density wrt respect to the 
        'background' density or 'critical' density
    
    Returns
    -------
    m_vir : array_like
        virial halo mass in h^{-1}M_{\odot}
    """
    
    m_h = np.atleast_1d(m_h)
    c = np.atleast_1d(c)
    z = np.atleast_1d(z)
    
    if cosmo=='default':
        cosmo = default_cosmo
    
    if wrt=='background':
        pass
    elif wrt=='critical':
        delta_h = delta_h/cosmo.Om(z)
    else:
        msg = 'mean density wrt paramater not recognized.'
        raise ValueError(msg)
    
    def f_x(x):
        """
        eq. C3
        """
        return x**3*(np.log(1.0+1.0/x)-1.0/(1.0+x))
    
    def x_f(f):
        """
        fitting function to inverse of f_x, eq. C11
        """
        
        a1 = 0.5116
        a2 = -0.4283
        a3 = -3.13*10**(-3)
        a4 = -3.52*10**(-5)
        p = a2 +a3*np.log(f) + a4*np.log(f)**2
        
        return (a1*f**(2.0*p) + (3/4)**2)**(-0.5) + 2.0*f
    
    f_h = delta_h/delta_vir(z, cosmo)
    
    r_ratio = x_f(f_h*f_x(1.0/c))
    
    f = (f_h)*r_ratio**(-3)*(1.0/c)**3.0
    
    return m_h/f


def halo_mass_conversion(m_h, c, delta_h=200, z=0.0, cosmo='default',
                         wrt_h='background', delta_new=360.0, wrt_new='background'):
    """
    Converrt between halo mass definitions
    fitting function from Hu \& Kravtsov (2003).
    
    Parameters
    ----------
    m_h : array_like
        halo mass
    
    c : array_like
        concentration
    
    delta_h : float
        over-density
    
    cosmo : astropy.cosmology object, optional
        cosmology used to calculated the virial over-density
    
    wrt_h : string
        halo over-density wrt respect to the 
        'background' density or 'critical' density
    
    delta_new : float
        convert to over-density
        
    wrt_new : string
        covert to halo over-density wrt respect to the 
        'background' density or 'critical' density
    
    Returns
    -------
    m_vir : array_like
        virial halo mass in h^{-1}M_{\odot}
    """
    
    m_h = np.atleast_1d(m_h)
    c = np.atleast_1d(c)
    z = np.atleast_1d(z)
    
    if cosmo=='default':
        cosmo = default_cosmo
    
    if wrt_h=='background':
        pass
    elif wrt_h=='critical':
        delta_h = delta_h/cosmo.Om(z)
    else:
        msg = 'mean density wrt paramater not recognized.'
        raise ValueError(msg)
        
    if wrt_new=='background':
        pass
    elif wrt_new=='critical':
        delta_new = delta_new/cosmo.Om(z)
    else:
        msg = 'mean density wrt paramater not recognized.'
        raise ValueError(msg)
    
    def f_x(x):
        """
        eq. C3
        """
        return x**3*(np.log(1.0+1.0/x)-1.0/(1.0+x))
    
    def x_f(f):
        """
        fitting function to inverse of f_x, eq. C11
        """
        
        a1 = 0.5116
        a2 = -0.4283
        a3 = -3.13*10**(-3)
        a4 = -3.52*10**(-5)
        p = a2 +a3*np.log(f) + a4*np.log(f)**2
        
        return (a1*f**(2.0*p) + (3/4)**2)**(-0.5) + 2.0*f
    
    f_h = delta_h/delta_vir(z, cosmo)
    
    r_ratio = x_f(f_h*f_x(1.0/c))
    
    f_1 = (f_h)*r_ratio**(-3)*(1.0/c)**3.0
    
    f_h = delta_new/delta_vir(z, cosmo)
    
    r_ratio = x_f(f_h*f_x(1.0/c))
    
    f_2 = (f_h)*r_ratio**(-3)*(1.0/c)**3.0
    
    f = f_1/f_2
    
    return m_h/f


def dynamical_time(z, cosmo='default'):
    """
    dynamical time
    
    Paramaters
    ----------
    z : array_like
        redshift
    
    cosmo : astropy.cosmology object
    
    Returns
    -------
    t : array_like
        dynamical time at redshift z h^-1 Gyr
    """
    
    if cosmo=='default':
        cosmo = default_cosmo
    
    z = np.atleast_1d(z)
    
    t_dyn = 1.628*((delta_vir(z, cosmo)/178.0)**(-0.5)) * ((cosmo.H(z)/cosmo.H0)**(-1.0))
    
    return t_dyn.value


def lookback_time(z, cosmo='default'):
    """
    lookback time
    
    Paramaters
    ----------
    z : array_like
        redshift
    
    cosmo : astropy.cosmology object
    
    Returns
    -------
    t : array_like
        lookback time to redshift z in h^-1 Gyr
    
    Notes
    -----
    This function builds an interpolation function instead of doing an integral for 
    each z which makes this substantially faster than astropy.cosmology lookback_time() 
    routine for large arrays.
    """
    
    if cosmo=='default':
        cosmo = default_cosmo
    
    z = np.atleast_1d(z)
    
    #if z<0, set t=0.0
    mask = (z>0.0)
    t = np.zeros(len(z))
    
    #build interpolation function for t_look(z)
    max_z = np.max(z)
    z_sample = np.logspace(0,np.log10(max_z+1),1000) - 1.0
    t_sample = cosmo.lookback_time(z_sample).to('Gyr').value
    f = interp1d(np.log10(1+z_sample), np.log10(t_sample), fill_value='extrapolate')
    
    t[mask] = 10.0**f(np.log10(1+z[mask])) * cosmo.h
    
    return t