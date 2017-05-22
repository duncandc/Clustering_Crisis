c**********************************************************************
c                    COSMOLOGY SUBROUTINES
c**********************************************************************
c
c  xH(z)           - compute hubble constant at redshift z (in km/s/Mpc)
c  omega(z)        - compute universal matter density at reshift z
c  Delta_c(z)      - compute the critical density (i.e. 1.686 for SCDM)
c  Delta_crit(z)   - compute the virial mass fraction (i.e. 178 for SCDM)
c  time(z)         - time at redshift z, in terms of 1/H_0 with time(infty)=0
c  lookbacktime(z) - lookbacktime at redshift z in Gyrs with t(0) = 0
c  growth_rate(z)  - the linear growth factor at redshift z
c  tranfer_WDM(k)  - transfer function for WDM (wrt CDM)
c  power_spec(k)   - the CDM power spectrum (with arbitrary normalization)
c  variance(M)     - the variance of the CDM power spectrum on scale of mass M 
c
c  So far the following models are supported:
c         Omega_0 + Omega_Lambda = 1.0
c         Omega_0 < 1 and Omega_Lambda = 0.0
c
c  Frank van den Bosch                             Dec. 1999
c**********************************************************************

      SUBROUTINE init_cosmology  
c---------------------------------------------------------------------------
c
c  Subroutine to initialize cosmology related  stuff
c
c---------------------------------------------------------------------------
 
      INCLUDE 'paramfile.h'

c---parameter that sets photon temperature (needed for EH transfer function)

      REAL   theta
      PARAMETER (theta = 1.0093)

      REAL     sTH,sGS,sSK,m_WDM,rfilter
      REAL     f,bb1,bb2,zeq,zd,y,Fy,Gy,Req,Rd,a1,a2,b1,b2

      REAL     time,Delta_crit,XEXP
      EXTERNAL time,Delta_crit,XEXP
     
c---

c---Calculate H0 in km/s/Mpc and its reciprocal in Gyr. From the latter
c   we then calculate the age of the Universe, t0, in Gyr

      xH_0 = 100.0 * xhubble
      xH_0_recip = 1.0 / (xH_0 * 1.023E-3)            
  
c---calculate free-fall time at z=0 for an overdensity of 200
 
      tff0 = (pi/SQRT(800.0 * omega_0)) * xH_0_recip

c---calculate parameters used to speed up calculation of (lookback)time

      cstlbt1 = SQRT((1.0-omega_0)/omega_0)
      cstlbt2 = 2.0 / (3.0*SQRT(1.0-omega_0))

      t0 = time(0.0) * xH_0_recip

c---set critical density at z=0 and comoving mass density [h^2 Msun Mpc^{-3}]

      rho_crit_0 = 3.0E+4 / (8.0 * pi * gee)
      rho_aver_com = rho_crit_0 * omega_0

c---calculate critical density for collapse at z=0

      deltacrit0 = Delta_crit(0.0)

c---calculate baryonic mass fraction

      f_bar = omega_b/omega_0

c---define a number of parameters needed to compute the Eisenstein & Hu
c   power specrum

      f = omega_0 * xhubble**2

      bb1 = 0.313 * f**(-0.419) * (1.0 + 0.607*f**0.674)
      bb2 = 0.238 * f**(0.223)
 
      bnode = 8.41*f**0.435

      keq = 7.46E-2 * f / theta**2.0
      ksilk = 1.6 * (omega_b_h2)**0.52 * f**0.73 * 
     &       (1.0 + (10.4*f)**(-0.95))

      zeq = 2.5E+4 * f / theta**4.0
      zd = 1291.0 * ((f**0.251)/(1.0 + 0.659*f**0.828)) *
     &               (1.0 + bb1 * omega_b_h2**bb2)

      y = ((1.0+zeq)/(1.0+zd))
      Fy = ALOG((SQRT(1.0+y) + 1.0) / (SQRT(1.0+y) - 1.0))
      Gy = y * (-6.0 * SQRT(1.0+y) + (2.0+3.0*y) * Fy)

      Req = 31.5 * omega_b_h2 * (1000.0/zeq) / theta**4.0
      Rd  = 31.5 * omega_b_h2 * (1000.0/zd) / theta**4.0

      sEH = (2.0/(3.0*keq)) * SQRT(6.0/Req) *
     &      ALOG((SQRT(1.0+Rd) + SQRT(Rd+Req))/(1.0+SQRT(Req)))
        
      a1 = (46.9*f)**(0.670) * (1.0 + (32.1*f)**(-0.532))
      a2 = (12.0*f)**(0.424) * (1.0 + (45.0*f)**(-0.582))
      b1 = 0.944 / (1.0+(458.0*f)**(-0.708))
      b2 = (0.395*f)**(-0.0266)

      alpha_c = a1**(-f_bar) * a2**(-(f_bar**3))
      beta_c = 1.0 + b1*(((omega_0-omega_b)/omega_0)**b2 - 1.0)
      beta_c = 1.0/beta_c

      alpha_b = 2.07 * keq * sEH * (1.0+Rd)**(-0.75) * Gy
      beta_b = 0.5 + f_bar + 
     &   (3.0-2.0*f_bar) * SQRT((17.2*f)**2 + 1.0)

      RETURN
      END

c**********************************************************************

      REAL FUNCTION xH(z)
c----------------------------------------------------------------------
c
c Calculate hubble constant (in physical units; km/s/Mpc) at redshift z.
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL   z,z1,fac

c---

      z1 = 1.0 + z
    
      fac = omega_lambda + (1.0 - omega_lambda - omega_0) * z1**2 + 
     &      omega_0 * z1**3

      xH = xH_0 * SQRT(fac)

      END

c**********************************************************************

      REAL FUNCTION omega(z)
c----------------------------------------------------------------------
c
c Calculate the density parameter omega at redshift z. Note that omega 
c is only the mater contribution.
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL     z,xH
  
      EXTERNAL xH

c---

      omega = omega_0 * (1.0 + z)**3 / (xH(z)/xH_0)**2.0

      END

c**********************************************************************

      REAL FUNCTION Delta_collapse(z)
c----------------------------------------------------------------------
c
c The overdensity for collapse at redshift z, called W in the
c EPS formalism of Lacey & Cole (1993). 
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL     z

      REAL     Delta_c,growth_rate
      EXTERNAL Delta_c,growth_rate

c---

      Delta_collapse = Delta_c(z) / growth_rate(z)
c      Delta_collapse = 1.686 / growth_rate(z)

      END

c**********************************************************************

      REAL FUNCTION Delta_c(z)
c----------------------------------------------------------------------
c
c Calculate the critical overdensity.
c We use the approximation in NFW97
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL     z,dc0,omz

      REAL     omega
      EXTERNAL omega

c---

      omz = omega(z)
      dc0 = 0.0
 
      IF (ABS(1.0-omega_0-omega_lambda).LT.1.0E-4) THEN
        dc0 = 0.15 * (12.0 * pi)**(2.0/3.0) * omz**(0.0055)
      END IF

      IF (omega_0.LT.1.0.AND.omega_lambda.EQ.0.0) THEN
        dc0 = 0.15 * (12.0 * pi)**(2.0/3.0) * omz**(0.0185)
      END IF
    
      IF (dc0.EQ.0.0) THEN
        WRITE(*,*)' Delta_c not defined for this cosmology'
        STOP
      ELSE
        Delta_c = dc0
      END IF

      END

c**********************************************************************

      REAL FUNCTION Delta_crit(z)
c----------------------------------------------------------------------
c
c Calculate the virial density in terms of critical density of the 
c Universe. We use the fitting formulae of Bryan & Norman, 1998, ApJ,
c 495, 80. These are accurate to better than 1% for 0.1 < omega_0 < 1.0
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL   z,x,omega
 
      EXTERNAL omega
c---

      x = omega(z) - 1.0

      IF (ABS(1.0-omega_0-omega_lambda).LT.1.0E-4) THEN
        Delta_crit = 18.0 * pi**2 + 82.0*x - 39.0 * x**2
        GOTO 3
      END IF
      
      IF (omega_0.LT.1.0.AND.omega_lambda.EQ.0.0) THEN
        Delta_crit = 18.0 * pi**2 + 60.0*x - 32.0 * x**2
        GOTO 3
      END IF

      WRITE(*,*)' Delta_crit not defined for this cosmology'
      STOP

 3    CONTINUE

      END

c**********************************************************************

      REAL FUNCTION time(z)
c----------------------------------------------------------------------
c
c Calculate time at redshift z in units of (1/H0)
c
c  This is a slightly modified version from a subroutine that appears 
c  in `subroutines.f' from the code of Navarro to calculate the 
c  concentration parameter c for NFW-halos.
c
c I have checked this procedure for SCDM, OCDM, and LCDM: works fine.
c
c Frank C. van den Bosch                                Feb. 1999
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL    z,tz,z1,a,atmp,omega1,const0,param0
      REAL    xx,xxz,xxmax,sstep

c---

c---Omega_0 = 1

      IF (omega_0.EQ.1.0) THEN
        tz=(2.0/3.0)*(1.0+z)**(-1.5)     
        GOTO 2
      END IF

c---Omega_0 + Omega_lambda=1

      IF (ABS(1.0-omega_0-omega_lambda).LT.1.0E-4) THEN
        z1 = 1.0 + z
        param0 = cstlbt1 * z1**(-1.5)
        tz=cstlbt2 * ALOG(param0 + SQRT(1.0+param0*param0))
        GOTO 2
      END IF

      WRITE(*,*)' Time not defined for this cosmology'
      STOP

 2    CONTINUE
      time = tz

      END

c**********************************************************************

      REAL FUNCTION lookbacktime(z)
c--------------------------------------------------------------------
c
c Computes lookbacktime in Gyrs at redshift z.
c   Here lookbacktime is defined such that
c       lookbacktime = 0  at z=0
c       lookbacktime = t0 at z=infty
c
c--------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL     z

      REAL     time
      EXTERNAL time

c---

      IF (z.EQ.0.0) THEN
        lookbacktime = 0.0
      ELSE
        lookbacktime = t0 - time(z) * xH_0_recip
      END IF

      END

c**********************************************************************

      REAL FUNCTION growth_rate(z)
c--------------------------------------------------------------------
c
c The linear growth factor at redshift z (see NFW97)
c
c--------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL     z,ddtemp
      REAL*8   ww,w_0,y_0,y
  
      REAL*8   f2,f3
      EXTERNAL f2,f3

c---

      IF (omega_0.EQ.1.0.AND.omega_lambda.EQ.0.0) THEN
        ddtemp = 1.0/(1.0 + z)
        GOTO 3
      END IF

      IF (ABS(1.0-omega_0-omega_lambda).LT.1.0E-4) THEN
        w_0 = 1.0d0/DBLE(omega_0) - 1.0d0
        y_0 = (2.0d0 * w_0)**(1.0d0/3.0d0)
        y = y_0/(1.0d0 + DBLE(z))  
        ddtemp = SNGL((f2(y) * f3(y))/(f2(y_0) * f3(y_0)))
        GOTO 3
      END IF

 3    CONTINUE
      growth_rate = ddtemp

      END

c**********************************************************************

      REAL FUNCTION dlnDdlnt(z)
c----------------------------------------------------------------------
c
c  The logarithmic derivative dlnD/dlnt(t) where D(t) is the linear
c  growth rate
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL      eps
      PARAMETER (eps=0.01)

      REAL     z,z1,z2,t1,t2,dd1,dd2

      REAL     growth_rate,time
      EXTERNAL growth_rate,time

c---

      z1 = (1.0-eps) * z
      z2 = (1.0+eps) * z
       
      z1 = MAX(0.0,z1)

      t1 = time(z1) * xH_0_recip
      t2 = time(z2) * xH_0_recip
       
      dd1 = growth_rate(z1)
      dd2 = growth_rate(z2)

      dlnDdlnt = (ALOG(dd2)-ALOG(dd1)) / (ALOG(t2)-ALOG(t1))

      END

c*********************************************************************

      REAL FUNCTION power_spec(xk)
c--------------------------------------------------------------------
c
c The CDM power spectrum with arbitrary normalization.
c The transfer function is taken from Eisenstein & Hu.
c The initial power-spectrum directly after inflation is a power-law
c with index `nspec'. For nspec=1 this yields the standard 
c Harrison-Zel'dovich spectrum. The normalization is set by xk0
c which defines the wavelength at which the amplitude of the initial
c fluctuation spectrum is unity (this is arbitrary). The actual
c normalisation of the power-spectrum is set by sigma8.
c
c--------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL    e
      PARAMETER (e=2.7182818)

      REAL    xk,k,q,t1,t2,tk,xk0,T_WDM
      REAL    silk,stilde,fff,C1,C2,t11,t12,t3,tb1,tb2,T_c,T_b

      REAL     XEXP,transferWDM
      EXTERNAL XEXP,transferWDM

c---

c---set normalization of initial power spectrum

      xk0 = 1.0/3000.0

c---the Eisenstein & Hu fitting function

      k = xk * xhubble
      q = k/(13.41*keq)

      silk = (k/ksilk)**1.4
      silk = XEXP(-silk)
      stilde = sEH / ((1.0 + (bnode/(k*sEH))**3)**0.333)

      fff = 1.0 / (1.0 + (k*sEH/5.4)**4)
      C1 = 14.2 + 386.0/(1.0+69.9*q**(1.08))
      C2 = 14.2/alpha_c + 386.0/(1.0+69.9*q**(1.08))

      t11 = ALOG(e + 1.8 * beta_c * q)
      t12 = ALOG(e + 1.8 * q)

      t1 = t11 / (t11 + C1*q**2)
      t2 = t11 / (t11 + C2*q**2)
      t3 = t12 / (t12 + C1*q**2)

      tb1 = t3 / (1.0 + (k*sEH/5.2)**2)
      tb2 = (alpha_b/(1.0 + (beta_b/(k*sEH))**3)) * silk

c---for small xk*stilde I approximate sin(x)/x=1

      T_c = fff * t1 + (1.0-fff) * t2
      IF ((k*stilde).LT.1.0E-25) THEN
        T_b = (tb1 + tb2)
      ELSE
        T_b = (tb1 + tb2) * SIN(k*stilde)/(k*stilde)
      END IF

      tk = (omega_b/omega_0) * T_b + ((omega_0-omega_b)/omega_0) * T_c

      power_spec = tk**2 * (xk/xk0)**nspec

      END

c**********************************************************************

      SUBROUTINE init_variance
c---------------------------------------------------------------------------
c
c  Subroutine to initialize mass variance
c
c---------------------------------------------------------------------------
 
      INCLUDE 'paramfile.h'

      INTEGER  i,ivtemp
      REAL     xM,z,yp1,ypn,xp1,xpn

      REAL     var_numerical,Delta_collapse
      EXTERNAL var_numerical,Delta_collapse

c---

c---normalize mass variance (using sigma_8)

      sigma8_norm = sigma8
      sigma8_norm = var_numerical(5.9543E+14 * omega_0)
      c8 = 1.0

c---set up the mass variance on a grid. The grid is a one-D vector,
c   for which we compute the mass variance numerically. The grid
c   consistes of Nsigma points with 5 <= log(M) <= 18.0

      DO i=1,Nsigma
        vectorM(i) = Mminvar + 
     &       FLOAT(i-1)/FLOAT(Nsigma-1) * (Mmaxvar-Mminvar)
        vectorZ(i) = FLOAT(i-1)/FLOAT(Nsigma-1) * 100.0
        xM = 10.0**vectorM(i)
        z = vectorZ(i)
        vectorS(i) = var_numerical(xM)
        vectorD(i) = Delta_collapse(z)
      END DO      

c---compute the derivatives at the two ends of one-D grids

      yp1 = (vectorS(2) - vectorS(1)) / 
     &      (vectorM(2) - vectorM(1))
      ypn = (vectorS(Nsigma) - vectorS(Nsigma-1)) / 
     &      (vectorM(Nsigma) - vectorM(Nsigma-1))

      xp1 = (vectorZ(2) - vectorZ(1)) /
     &      (vectorD(2) - vectorD(1))
      xpn = (vectorZ(Nsigma) - vectorZ(Nsigma-1)) / 
     &      (vectorD(Nsigma) - vectorD(Nsigma-1))

c---and compute the spline coefficients, to be used for spline interpolation
c   note that we compute the spline coefficients both ways!

      DO i=1,Nsigma
        vectorSS(i) = vectorS(Nsigma+1-i)
        vectorMM(i) = vectorM(Nsigma+1-i)
      END DO

      CALL spline(vectorM,vectorS,Nsigma,yp1,ypn,vectorS2)
      CALL spline(vectorSS,vectorMM,Nsigma,2.0E+30,2.0E+30,vectorM2)
      CALL spline(vectorD,vectorZ,Nsigma,xp1,xpn,vectorZ2)

      RETURN
      END

c*********************************************************************

      REAL FUNCTION variance(M)
c--------------------------------------------------------------------
c
c This function yields the mass variance s(M) for a CDM power spectrum.
c We use the BBKS transfer function, and there is a choise of
c three different filters (Top-Hat, Gaussian, and Sharp-k).
c 
c [M] = h^{-1} Msun
c 
c NOTE: This rms mass variance is normalized by sigma8.
c       For that you need to initialize it by calling this
c       routine with sigma8_norm set to sigma8 and the mass
c       set to M8 = 5.9543E+14 * omega_0. Call the resulting
c       value sigma8_norm. Subsequent calls than yield
c       the `sigma8-normalized' values.
c 
c--------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL  M,Mbuf,logM

      REAL     var_numerical,var_spline
      EXTERNAL var_numerical,var_spline

c---

      Mbuf = M
      logM = ALOG10(M)

      IF (logM.LT.Mminvar.OR.logM.GT.Mmaxvar) THEN
        variance = var_numerical(Mbuf)
      ELSE
        variance = var_spline(Mbuf)
      END IF

      END

c*********************************************************************

      REAL FUNCTION var_numerical(M)
c--------------------------------------------------------------------
c
c Mass variance s(M), computed numerically by using the appropriate
c integral equation. Based on Top-Hat Filter
c 
c [M] = h^{-1} Msun
c 
c--------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      INTEGER ierr1,ierr2,ierr3
      REAL    M,SS1,SS2,SS3,SS1err,SS2err,SS3err

      REAL  Rf
      COMMON /filtering/ Rf

      REAL     toint1
      EXTERNAL toint1
 
c---

      Rf = ((3.0*M)/(4.0*pi*rho_aver_com))**(1.0/3.0)

      IF (Rf.LT.2.0) THEN
        CALL qags(toint1,0.0,0.5,1.0E-5,1.0E-5,SS1,SS1err,Neval,
     &            ierr1,Nlimit,Nlenw,last,iwork,work)
        CALL qags(toint1,0.5,1.0/Rf,1.0E-5,1.0E-5,SS2,SS2err,Neval,
     &            ierr2,Nlimit,Nlenw,last,iwork,work)
        CALL qagi(toint1,1.0/Rf,1,1.0E-5,1.0E-5,SS3,SS3err,Neval,
     &            ierr3,Nlimit,Nlenw,last,iwork,work)
      ELSE
        CALL qags(toint1,0.0,1.0/Rf,1.0E-5,1.0E-5,SS1,SS1err,Neval,
     &            ierr1,Nlimit,Nlenw,last,iwork,work)
        CALL qags(toint1,1.0/Rf,0.5,1.0E-5,1.0E-5,SS2,SS2err,Neval,
     &            ierr2,Nlimit,Nlenw,last,iwork,work)
        CALL qagi(toint1,0.5,1,1.0E-5,1.0E-5,SS3,SS3err,Neval,
     &            ierr3,Nlimit,Nlenw,last,iwork,work)
      END IF

      var_numerical = (sigma8/sigma8_norm) * 
     &                   SQRT((SS1 + SS2 + SS3)/(2.0*pi*pi))

      END

c*********************************************************************

      REAL FUNCTION var_spline(M)
c--------------------------------------------------------------------
c
c Mass variance s(M), computed using spline interpolation
c 
c [M] = h^{-1} Msun
c 
c--------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL  M,logM,SS1
      
c---

      logM = ALOG10(M)
      CALL splint(vectorM,vectorS,vectorS2,Nsigma,logM,SS1)
      var_spline = SS1

      END

c**********************************************************************

      REAL FUNCTION dlnSdlnM(xM)
c----------------------------------------------------------------------
c
c  The logarithmic derivative dlnS/dlnM(M) where S(M) = variance(M)**2
c
c----------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL      eps
      PARAMETER (eps=0.15)

      REAL     xM,M1,M2,sig1,sig2

      REAL     variance
      EXTERNAL variance

c---

      M1 = (1.0-eps) * xM
      M2 = (1.0+eps) * xM

      sig1 = (variance(M1))**2
      sig2 = (variance(M2))**2

      dlnSdlnM = (ALOG(sig2)-ALOG(sig1)) / (ALOG(M2)-ALOG(M1))

      END

c**********************************************************************

      REAL*8 FUNCTION f2(x)
c--------------------------------------------------------------------
c
c Auxialiary function used in computation of growth rate
c
c--------------------------------------------------------------------

      IMPLICIT NONE
       
      REAL*8    x

c---

      f2 = DSQRT(x**3.0d0 + 2.0d0)/(x**1.5d0)

      END

c**********************************************************************

      REAL*8 FUNCTION f3(x)
c--------------------------------------------------------------------
c
c Auxialiary function used in computation of growth rate
c
c--------------------------------------------------------------------

      IMPLICIT NONE

      REAL     ff3,SS
      REAL*8   x

      EXTERNAL ff3

c---

      CALL QROMB(ff3,0.0,SNGL(x),SS)

      f3 = DBLE(SS)

      END 

c**********************************************************************
   
      REAL FUNCTION ff3(x)
c--------------------------------------------------------------------
c
c Auxialiary function used in computation of growth rate
c
c--------------------------------------------------------------------

      IMPLICIT NONE

      REAL     x

c---

      ff3 = (x/(x**3.0 + 2.0))**1.5

      END 

c*********************************************************************

      REAL FUNCTION toint1(xk)
c--------------------------------------------------------------------
c
c Integrating this function yields the mass variance (TH filter only)
c
c--------------------------------------------------------------------

      INCLUDE 'paramfile.h'

      REAL     xk,wf,x

      REAL  Rf
      COMMON /filtering/ Rf

      REAL     power_spec,XEXP
      EXTERNAL power_spec,XEXP

c---

c---compute filter at wavelength xk

      x = xk * Rf
      wf = 3.0 * (SIN(x) - x * COS(x)) / x**3

c---and compute integrand

      toint1 = power_spec(xk) * wf**2 * xk**2

      END

c*********************************************************************

      SUBROUTINE qromb(func,a,b,ss)
      INTEGER JMAX,JMAXP,K,KM
      REAL a,b,func,ss,EPS
      EXTERNAL func
      PARAMETER (EPS=1.e-6, JMAX=20, JMAXP=JMAX+1, K=5, KM=K-1)
      INTEGER j
      REAL dss,h(JMAXP),s(JMAXP)
      h(1)=1.
      do 11 j=1,JMAX
        call trapzd(func,a,b,s(j),j)
        if (j.ge.K) then
          call polint(h(j-KM),s(j-KM),K,0.,ss,dss)
          if (abs(dss).le.EPS*abs(ss)) return
        endif
        s(j+1)=s(j)
        h(j+1)=0.25*h(j)
11    continue
      CALL Terminate('too many steps in qromb')
      END

c---

      SUBROUTINE polint(xa,ya,n,x,y,dy)
      INTEGER n,NMAX
      REAL dy,x,y,xa(n),ya(n)
      PARAMETER (NMAX=10)
      INTEGER i,m,ns
      REAL den,dif,dift,ho,hp,w,c(NMAX),d(NMAX)
      ns=1
      dif=abs(x-xa(1))
      do 11 i=1,n
        dift=abs(x-xa(i))
        if (dift.lt.dif) then
          ns=i
          dif=dift
        endif
        c(i)=ya(i)
        d(i)=ya(i)
11    continue
      y=ya(ns)
      ns=ns-1
      do 13 m=1,n-1
        do 12 i=1,n-m
          ho=xa(i)-x
          hp=xa(i+m)-x
          w=c(i+1)-d(i)
          den=ho-hp
          if(den.eq.0.) CALL Terminate('failure in polint')
          den=w/den
          d(i)=hp*den
          c(i)=ho*den
12      continue
        if (2*ns.lt.n-m)then
          dy=c(ns+1)
        else
          dy=d(ns)
          ns=ns-1
        endif
        y=y+dy
13    continue
      return
      END

c---

      SUBROUTINE trapzd(func,a,b,s,n)
      INTEGER n
      REAL a,b,s,func
      EXTERNAL func
      INTEGER it,j
      REAL del,sum,tnm,x
      if (n.eq.1) then
        s=0.5*(b-a)*(func(a)+func(b))
      else
        it=2**(n-2)
        tnm=it
        del=(b-a)/tnm
        x=a+0.5*del
        sum=0.
        do 11 j=1,it
          sum=sum+func(x)
          x=x+del
11      continue
        s=0.5*(s+(b-a)*sum/tnm)
      endif
      return
      END

c---

      SUBROUTINE splint(xa,ya,y2a,n,x,y)
      INTEGER n
      REAL x,y,xa(n),y2a(n),ya(n)
      INTEGER k,khi,klo
      REAL a,b,h
      klo=1
      khi=n
1     if (khi-klo.gt.1) then
        k=(khi+klo)/2
        if(xa(k).gt.x)then
          khi=k
        else
          klo=k
        endif
      goto 1
      endif
      h=xa(khi)-xa(klo)
      if (h.eq.0.) CALL Terminate('bad xa input in splint')
      a=(xa(khi)-x)/h
      b=(x-xa(klo))/h
      y=a*ya(klo)+b*ya(khi)+((a**3-a)*y2a(klo)+(b**3-b)*y2a(khi))*(h**
     *2)/6.
      return
      END
      
c---

      SUBROUTINE spline(x,y,n,yp1,ypn,y2)
      INTEGER n,NMAX
      REAL yp1,ypn,x(n),y(n),y2(n)
      PARAMETER (NMAX=10000)
      INTEGER i,k
      REAL p,qn,sig,un,u(NMAX)
      if (yp1.gt..99e30) then
        y2(1)=0.
        u(1)=0.
      else
        y2(1)=-0.5
        u(1)=(3./(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)
      endif
      do 11 i=2,n-1
        sig=(x(i)-x(i-1))/(x(i+1)-x(i-1))
        p=sig*y2(i-1)+2.
        y2(i)=(sig-1.)/p
        u(i)=(6.*((y(i+1)-y(i))/(x(i+
     *1)-x(i))-(y(i)-y(i-1))/(x(i)-x(i-1)))/(x(i+1)-x(i-1))-sig*
     *u(i-1))/p
11    continue
      if (ypn.gt..99e30) then
        qn=0.
        un=0.
      else
        qn=0.5
        un=(3./(x(n)-x(n-1)))*(ypn-(y(n)-y(n-1))/(x(n)-x(n-1)))
      endif
      y2(n)=(un-qn*u(n-1))/(qn*y2(n-1)+1.)
      do 12 k=n-1,1,-1
        y2(k)=y2(k)*y2(k+1)+u(k)
12    continue
      return
      END



