"""
utility functons
"""

import numpy as np
from scipy.special import erfinv
from default import PROJECT_DIRECTORY, DATA_DIRECTORY
from halotools.sim_manager import UserSuppliedHaloCatalog
import h5py
from astropy.table import Table


__all__ = ['scatter_ranks','gaussian_transform','table_to_array','load_project_halocat']


def load_project_halocat(simname='bolshoi_250', version_name='custom', halo_finder='Rockstar',
                         redshift=0.0, dz_tol=0.01, fname=None):
    """
    Load halotools halo catalogues without saving it in cache.
    
    Paramaters
    ----------
    simname : string, optional
        Nickname of the simulation used as a shorthand way to keep track of the halo catalogs
    
    version_name : string, optional
        Nickname of the version of the halo catalog.
    
    redshift : float, optional
        Redshift of the halo catalog.
    
    dz_tol :  float, optional
        Tolerance within to search for a catalog with a matching redshift.
    
    fname : string, optional
        file name of hdf5 file storing reduced halo catalogue.
        If provided, all other information will be ignored.
    
    Returns
    -------
    halocat : object
    """
    
    if fname is None:
        
        cache_table = load_project_halo_catalogue_cache_table(PROJECT_DIRECTORY)
        
        z, mask_0 = find_nearest(cache_table['redshift'],redshift)
        if np.fabs(z-redshift)>dz_tol:
            msg = ('No halo catalogue for the provided redshift within dz_tol.')
            raise ValueError(msg)
        scale_factor = 1.0/(1+z)
        str_scale_factor = str('{0:.5f}'.format(float(scale_factor)))
        str_redshift = str('{0:.4f}'.format(float(z)))
        
        mask_1 = np.in1d(cache_table['simname'], simname)
        mask_2 = np.in1d(cache_table['version_name'], version_name)
        mask_3 = np.in1d(cache_table['halo_finder'], halo_finder)
        
        mask = (mask_0 & mask_1) & (mask_2 & mask_3)
        if np.sum(mask)<1:
            msg = ('no matchng entry in the project halo catalogue cache table.')
            raise ValueError(msg)
        elif np.sum(mask)>1:
            msg = ('more than one matching entry in the project halo catalogue cache table.')
            raise ValueError(msg)
        elif np.sum(mask)==1:
            fname = cache_table['fname'][mask]
            fname = fname[0]
            print('opening halo catalogue: {0}'.format(fname))
        else:
           msg = ('should never get here.')
           raise ValueError(msg)
    
    #load halo table and attributes
    f = h5py.File(DATA_DIRECTORY + fname, 'r')
    halo_table = Table.read(DATA_DIRECTORY + fname, path='data')
    simname = f.attrs['simname']
    redshift = float(f.attrs['redshift'])
    
    #get additional attributes
    Lbox = f.attrs['Lbox']
    particle_mass = f.attrs['particle_mass']
    
    #initialize halo catalogue
    d = {key:halo_table[key] for key in halo_table.keys()}
    halocat = UserSuppliedHaloCatalog(simname=simname, redshift=redshift, Lbox=Lbox, particle_mass=particle_mass, **d)
    
    return halocat


def load_project_halo_catalogue_cache_table(PROJECT_DIRECTORY):
    """
    return project halo table cache
    """
    
    t = Table.read(PROJECT_DIRECTORY + 'project_halo_catalogue_cache.txt',
        names=['version_name','simname','fname','halo_finder','redshift'],format='ascii')
    
    return t


def scatter_ranks(arr, sigma):
    """
    Scatter the index of values in an array.
    
    Parameters
    ----------
    arr : array_like
        array of values to scatter
        
    sigma : array_like
        scatter relative to len(arr) 
    
    Returns
    -------
    scatter_array : numpy.array
        array with same values as ``arr``, but the locations of those values 
        have been scatter.
    """
    
    sigma = np.atleast_1d(sigma)
    if len(sigma)==1:
        sigma = np.repeat(sigma, len(arr))
    elif len(sigma)!=len(arr):
        raise ValueError("sigma must be same length as ``arr``.")
    
    #get array of indicies before scattering
    N = len(arr)
    inds = np.arange(0,N)
    
    mask = (sigma>1000.0)
    sigma[mask] = 1000.0
    
    #get array to scatter positions
    mask = (sigma>0.0)
    dn = np.zeros(N)
    dn[mask] = np.random.normal(0.0,sigma[mask]*N)
    
    #get array of new indicies
    new_inds = inds + dn
    new_inds = np.argsort(new_inds, kind='mergesort')
    
    return arr[new_inds]


def gaussian_transform(percentiles, mean, sigma):
    """ 
    map percentiles to values in normal distribution distribution
    
    Parameters
    ----------
    percentiles : array_like
        array of values between (0,1)
        
    mean : float
        mean of the normal distribution
    
    sigma : float
        standard deviation of the normal distribution
    
    Returns
    -------
    x : numpy.array
        values
    """
    return mean + 2**0.5 * sigma * erfinv(percentiles * 2 - 1)


def table_to_array(table):
    """
    convert astropy.table object numpy ndarray
    
    Parameters
    ----------
    table : astropy.table object
    
    Returns
    -------
    arr : numpy.array
    """
    colnames = table.colnames
    Ncols = len(colnames)
    Nrows = len(table)
    
    
    arr = np.ndarray((Nrows,Ncols))
    for i in range(0,Ncols):
        arr[:,i] = np.array(table[colnames[i]])
    
    return arr


def find_nearest(array,value):
    """
    find nearest value in an array
    
    Parameters
    ----------
    array : array_like
    
    value : float
    
    Returns
    -------
    closest_value : float
    
    mask : array
    """
    idx = (np.abs(array-value)).argmin()
    return array[idx], array==array[idx]
    

