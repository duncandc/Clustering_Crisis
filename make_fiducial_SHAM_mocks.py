"""
create fiducial SHAM mocks using default paramterizations
and plot the stellar mass functions
"""

#add project directory to python path
from __future__ import print_function, division
import sys
sys.path.append(".") #Place the project directory in the python path
from default import PROJECT_DIRECTORY, DATA_DIRECTORY
#import standard packages
import numpy as np
import matplotlib.pyplot as plt

def main():
    
    #set random seed to ensure reproducibility
    np.random.seed(42)
    
    from halotools import sim_manager
    
    #load halo catalogue
    print("loading Halo catalogue...")
    simname = 'bolshoi_250'
    halocat = sim_manager.CachedHaloCatalog(simname = simname, redshift=0.0, dz_tol = 0.001,
                                            version_name='custom', halo_finder='Rockstar')
    Nhalo = len(halocat.halo_table)
    
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.27) #Bolshoi cosmology
    
    #load supplemtary halo catalogue properties
    from astropy.table import Table
    
    #filepath = DATA_DIRECTORY
    #print("loading additonal halo properties table...")
    #add_halo_props = Table.read(filepath+"bolshoi_additional_halo_properties.hdf5", path='data')
    
    #replace with custom properties
    #halocat.halo_table['halo_mpeak'] = add_halo_props['halo_mpeak_prime']
    #halocat.halo_table['halo_acc_scale'] = add_halo_props['halo_prime_acc_scale_2']
    
    #the last snapshot is slightly after z=0, so just set z<0 to z=0
    mask = (halocat.halo_table['halo_acc_scale']>1.0)
    halocat.halo_table['halo_acc_scale'][mask] = 1.0
    
    #clear up some memory
    add_halo_props = 0.0
    
    #convert 360b halo masses to the ones used in the various models
    from cosmo_utils import halo_mass_conversion
    m180b = halo_mass_conversion(halocat.halo_table['halo_mpeak'],
                             halocat.halo_table['halo_nfw_conc'],
                             delta_h=360, delta_new=180,
                             cosmo=cosmo, wrt_h='background', wrt_new='background')
    m200c = halo_mass_conversion(halocat.halo_table['halo_mpeak'],
                             halocat.halo_table['halo_nfw_conc'],
                             delta_h=360, delta_new=200,
                             cosmo=cosmo, wrt_h='background',
                             wrt_new='critical')
    halocat.halo_table['halo_mpeak200c'] = m200c
    halocat.halo_table['halo_mpeak180b'] = m180b
    
    #create RM mock
    from default import composite_model_1
    composite_model_1.populate_mock(halocat = halocat)
    print("building RM mock...")
    mock_1 = composite_model_1.mock.galaxy_table
    print("     number of galaxies: {0}".format(len(mock_1)))
    
    #create RV mock
    from default import composite_model_2
    composite_model_2.populate_mock(halocat = halocat)
    print("building RV mock...")
    mock_2 = composite_model_2.mock.galaxy_table
    print("     number of galaxies: {0}".format(len(mock_2)))
    
    #create M13 mock
    from default import composite_model_3
    composite_model_3.populate_mock(halocat = halocat)
    print("building M13 mock...")
    mock_3 = composite_model_3.mock.galaxy_table
    print("     number of galaxies: {0}".format(len(mock_3)))
    
    #create Y12 mock
    from default import composite_model_4
    composite_model_4.populate_mock(halocat = halocat)
    print("building Y12 mock...")
    mock_4 = composite_model_4.mock.galaxy_table
    print("     number of galaxies: {0}".format(len(mock_4)))
    
    #create M13 mock
    from default import composite_model_5
    composite_model_5.populate_mock(halocat = halocat)
    print("building B13 mock...")
    mock_5 = composite_model_5.mock.galaxy_table
    print("     number of galaxies: {0}".format(len(mock_5)))
    
    #save mocks
    print("saving mocks to {0} ...".format(DATA_DIRECTORY))
    mock_1.write( DATA_DIRECTORY + 'RM_fiducial.dat', format = 'ascii')
    mock_2.write( DATA_DIRECTORY + 'RV_fiducial.dat', format = 'ascii')
    mock_3.write( DATA_DIRECTORY + 'M13_fiducial.dat', format = 'ascii')
    mock_4.write( DATA_DIRECTORY + 'Y12_fiducial.dat', format = 'ascii')
    mock_5.write( DATA_DIRECTORY + 'B13_fiducial.dat', format = 'ascii')
    
    def stellar_mass_func(mock):
        """
        caclulate stellar mass function
        """
        
        #define mass bins
        log_dm = 0.1
        bins = np.arange(9.5,12,log_dm)
        bins = 10.0**bins
        bin_centers = (bins[:-1]+bins[1:])/2.0
        
        #calculate number density
        counts = np.histogram(mock['stellar_mass'],bins=bins)[0]
        dndm = counts/(halocat.Lbox[0]**3)/log_dm 
        
        return dndm, bin_centers, bins
    
    dndm_1, bin_centers, bins = stellar_mass_func(mock_1)
    dndm_2, bin_centers, bins = stellar_mass_func(mock_2)
    dndm_3, bin_centers, bins = stellar_mass_func(mock_3)
    dndm_4, bin_centers, bins = stellar_mass_func(mock_4)
    dndm_5, bin_centers, bins = stellar_mass_func(mock_5)
    
    #load sdss results
    from lss_observations.stellar_mass_functions import LiWhite_2009_phi
    sdss_phi = LiWhite_2009_phi()
    sdss_dndm = sdss_phi.data_table['phi']
    sdss_err = sdss_phi.data_table['err']
    sdss_m = sdss_phi.data_table['bin_center']
    
    #plot stellar mass function
    fig = plt.figure(figsize=(3.3,4.3))
    
    #upper panel
    rect = 0.2,0.4,0.7,0.55
    ax = fig.add_axes(rect)
    p0 = ax.errorbar(sdss_m, sdss_dndm, yerr=sdss_err, fmt='o', color='black', ms=2)
    p1, = ax.plot(bin_centers, dndm_1, '-', color='darkorange')
    p2, = ax.plot(bin_centers, dndm_2, '-', color='green', lw=0.75)
    p3, = ax.plot(bin_centers, dndm_3, '-', color='black', lw=0.75)
    p4, = ax.plot(bin_centers, dndm_4, '--', color='black', lw=0.75)
    p5, = ax.plot(bin_centers, dndm_5, ':', color='black', lw=0.75)
    ax.set_xlim([10**9,10**12])
    ax.set_ylim([10**-6,10**-1])
    ax.set_yticks([10**-5,10**-4,10**-3,10**-2,10**-1])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$\phi(M_{*})~[h^{3}{\rm Mpc}^{-3}{\rm dex}^{-1}]$', labelpad=-1)
    ax.xaxis.set_visible(False)
    plt.legend((p0,p1,p2,p3,p4,p5),
               ("Li \& White (2009)","RM","RV","Moster et al. (2013)",\
                "Yang et al. (2012)","Behroozi et al. (2013)"),\
               loc=3, fontsize=8, frameon=False, numpoints=1)
    
    #lower panel
    rect = 0.2,0.125,0.7,0.275
    ax = fig.add_axes(rect)
    ax.plot(bin_centers, (dndm_1-sdss_phi(bin_centers))/sdss_phi(bin_centers),'-', color='darkorange', lw=1)
    ax.plot(bin_centers, (dndm_2-sdss_phi(bin_centers))/sdss_phi(bin_centers),'-', color='green', lw=0.75)
    ax.plot(bin_centers, (dndm_3-sdss_phi(bin_centers))/sdss_phi(bin_centers),'-', color='black', lw=0.75)
    ax.plot(bin_centers, (dndm_4-sdss_phi(bin_centers))/sdss_phi(bin_centers),'--', color='black', lw=0.75)
    ax.plot(bin_centers, (dndm_5-sdss_phi(bin_centers))/sdss_phi(bin_centers),':', color='black', lw=0.75)
    ax.errorbar(sdss_m,  (sdss_dndm-sdss_phi(sdss_m))/sdss_phi(sdss_m), yerr=(sdss_err)/sdss_phi(sdss_m), fmt='o', color='black', ms=2, lw=1)
    ax.set_ylim([-0.25,0.25])
    ax.set_yticks([-0.2,-0.1,0.0,0.1,0.2])
    ax.set_xlim([10**9,10**12])
    ax.set_xscale('log')
    ax.set_ylabel(r'$\Delta\phi/\phi_{\rm SDSS}$', labelpad=-2)
    ax.set_xlabel(r'$M_{*} ~[h^{-2}M_{\odot}]$')
    
    plt.show()
    
    filepath = PROJECT_DIRECTORY + 'figures/'
    filename = 'fiducial_mocks_stellar_mass_function'
    fig.savefig(filepath+filename+'.pdf', dpi=300)
    
if __name__ == '__main__':
    main()