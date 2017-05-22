c=======================================================================
c
c                      INCLUDE FILE fitpar.h
c
c=======================================================================
c  Written by: Frank van den Bosch
c=======================================================================

      IMPLICIT NONE

c=======================================================================
c                        PARAMETERS
c=======================================================================

c---the constant pi

      REAL  pi
      PARAMETER (pi=3.141592654)

c---The gravitational constant in M_sun^-1 Mpc (km/s)^2

      REAL  gee
      PARAMETER (gee = 4.2994E-9)

c---The number of redhifts used to sample MAH

      INTEGER  Nz
      PARAMETER (Nz = 400)

c--Number of interpolation points for mass variance as well as range

      INTEGER Nsigma
      REAL    Mminvar, Mmaxvar
      PARAMETER (Nsigma = 1000)
      PARAMETER (Mminvar = 4.0)
      PARAMETER (Mmaxvar =15.5)

c---parameters needed for quadpack integration routines

      INTEGER Nlimit,Nlenw
      PARAMETER (Nlimit=10000)
      PARAMETER (Nlenw =41000)
      
c---the null character

      CHARACTER null*1
      PARAMETER (null='0')

c=======================================================================
c                        COMMON BLOCKS
c=======================================================================

c---cosmology

      REAL  omega_0,omega_lambda,sigma8,xhubble
      REAL  nspec,omega_b_h2,c8
      COMMON /cosmo_model/ omega_0,omega_lambda,sigma8,xhubble,
     &                     nspec,omega_b_h2,c8

      REAL  omega_b,f_bar,rho_aver_com
      COMMON /cosmo_baryon/ omega_b,f_bar,rho_aver_com

      REAL     xH_0,xH_0_recip,tff0,t0,cstlbt1,cstlbt2
      REAL     rho_crit_0,sigma8_norm,deltacrit0
      COMMON /cosmo_param/ xH_0,xH_0_recip,tff0,t0,
     &             cstlbt1,cstlbt2,rho_crit_0,sigma8_norm,
     &             deltacrit0

c---Eisenstein & Hu power spectrum

      REAL     sEH,bnode,ksilk,keq,alpha_c,alpha_b,beta_c,beta_b
      COMMON /ehpar/ sEH,bnode,ksilk,keq,alpha_c,alpha_b,beta_c,beta_b

c---mass variance

      REAL   vectorM(Nsigma),vectorS(Nsigma)
      REAL   vectorMM(Nsigma),vectorSS(Nsigma)
      REAL   vectorM2(Nsigma),vectorS2(Nsigma)
      REAL   vectorD(Nsigma),vectorZ(Nsigma),vectorZ2(Nsigma)
      COMMON /spl_sigma/ vectorM,vectorS,vectorSS,vectorMM,vectorS2,
     &                   vectorM2,vectorD,vectorZ,vectorZ2
        
c---MAH

      REAL    zz(Nz),tt(Nz),xlgMAH(Nz),acc_rate(Nz)
      COMMON /Mahpar/ zz,tt,xlgMAH,acc_rate
      
c---MODEL

      REAL  apar1,apar2,apar3,apar4,apar5
      COMMON /modpar/ apar1,apar2,apar3,apar4,apar5
      
c---work space required for quadpack integration routines

      INTEGER Neval,ierr,last,iwork(Nlimit)
      REAL*8  work(Nlenw)
      COMMON /quadpackint/ work,iwork,Neval,ierr,last

c=======================================================================
c                             END
c=======================================================================





