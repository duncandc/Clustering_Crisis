"""
utility functons
"""

import numpy as np
from scipy.special import erfinv

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
