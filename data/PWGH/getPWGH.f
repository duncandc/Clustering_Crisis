      PROGRAM get_PWGH
c**********************************************************************
c  Program to compute mass accretion history and potential well growth
c  histories for dark matter haloes. In addition, program computes
c  the concentration parameter of the main progenitor halo as function
c  of redshift.
c
c  Frank van den Bosch                                  Yale, Jun 2014     
c**********************************************************************
      
      INCLUDE 'paramfile.h'
            
      INTEGER   i,j,k,ianswer
      REAL      z,t,xlgz,xlgpsi,xlgpsi_min,xlgpsi_max,psi
      REAL      VmaxVvir0,Vmaxrat,VmaxVvir,c0,cc
      REAL      M,Afac,Bfac,Hfac,fac1,fac2,s1
      CHARACTER outfile*30
      
      REAL      M0,z0,s0,dc0,delta_dc
      COMMON /fpars/ M0,z0,s0,dc0,delta_dc
            
      REAL     zriddr,find_psi,variance,Delta_collapse,get_conc
      REAL     lookbacktime,time,dlnDdlnt,dlnSdlnM,XEXP,dzdlnt
      EXTERNAL zriddr,find_psi,variance,Delta_collapse,get_conc
      EXTERNAL lookbacktime,time,dlnDdlnt,dlnSdlnM,XEXP,dzdlnt

c**********************************************************************

c---read global and cosmological parameters

      CALL read_parameters

c---decide whether you want to compute MEDIAN or AVERAGE
c   MAHs and PWGHs; set parameters of universal MAH accordingly
                     
      WRITE(*,*)' Do you want medians (0) or averages (1)?'
      READ(*,*)ianswer
      WRITE(*,*)' '
      
      IF (ianswer.NE.0.AND.ianswer.NE.1) THEN
        CALL Terminate('Invalid choice') 
      ELSE
        IF (ianswer.EQ.0) THEN
          apar1 = 1.9278
          apar2 = 0.4241
          apar3 = 0.7684
          apar4 = 0.1481
          apar5 = 0.3096
          outfile = 'PWGH_median.dat'
          WRITE(*,*)'   >>> Computing median MAH & PWGH <<<'
        ELSE
          apar1 = 3.2954
          apar2 = 0.1975
          apar3 = 0.7542
          apar4 = 0.0898
          apar5 = 0.4415
          outfile = 'PWGH_average.dat'
          WRITE(*,*)'   >>> Computing average MAH & PWGH <<<'
        END IF
      END IF
          
c---initialize new cosmological parameters

      CALL init_cosmology

c---initialize and store mass variance

      CALL init_variance   

c---set time and redshift sampling; we sample log(1+z) linearly

      DO j=1,Nz
        xlgz = ALOG10(1.0+z0) + 
     &       FLOAT(j-1)/FLOAT(Nz-1) * (1.5 - ALOG10(1.0+z0))
        xlgz = xlgz + 0.00004343
        zz(j) = 10.0**xlgz - 1.0
        tt(j) = lookbacktime(zz(j))
      END DO
              
c---initialize variance and critical collapse density at z=z0

      s0 = variance(M0)**2
      dc0 = Delta_collapse(z0)
        
c---loop over redshift and compute MAH

      xlgpsi = 0.0
      DO j=1,Nz
        z = zz(j)
        delta_dc = Delta_collapse(z) - dc0          

        xlgpsi_max = xlgpsi
        xlgpsi_min = xlgpsi - 1.0
        xlgpsi = zriddr(find_psi,xlgpsi_min,xlgpsi_max,1.0E-5)
        xlgMAH(j) = xlgpsi
          
        psi = 10.0**xlgpsi
        M = psi * M0
        t = time(z) * xH_0_recip
        s1 = variance(M)**2

        fac1 = Delta_collapse(z)/delta_dc - 0.004
        fac2 = (s1/(s1-s0)) - 0.152
        Afac = dlnDdlnt(z) * fac1
        Bfac = 0.5 * ABS(dlnSdlnM(M)) * fac2
        Hfac = (apar2*apar3*ALOG10(EXP(1.0))) / (1.0-apar2*xlgpsi) +
     &           (apar4*apar5*psi**apar4) / (1.0-psi**apar4)
        acc_rate(j) = M/(t*1.0E+09) * (Afac/(Hfac + Bfac))

      END DO

c---using the MAH, we now compute Vmax and concentration along the MAH

      c0 = get_conc(1.0,0.0)
      VmaxVvir0 = ALOG10(0.465 * SQRT(c0/(ALOG(1.0+c0)-c0/(1.0+c0))))      

      OPEN(10,file=outfile,status='UNKNOWN')
      DO j=1,Nz
        z = zz(j)
        psi = 10.0**xlgMAH(j)
        CALL get_Vmax_rat(psi,z,Vmaxrat,cc)
        VmaxVvir = Vmaxrat + VmaxVvir0
        WRITE(10,74)j,zz(j),tt(j),xlgMAH(j),Vmaxrat,VmaxVvir,cc,
     &                acc_rate(j)
      END DO     
      CLOSE(10)

c---      

 74   FORMAT(I5,2X,F9.4,2X,F7.4,2X,7(E12.5,2X))
 
      STOP
      END
      
c**********************************************************************

      REAL FUNCTION find_psi(xlgpsi)
c----------------------------------------------------------------------
c
c  Root of this function gives the scaled redshift for the new mass
c
c----------------------------------------------------------------------
 
      INCLUDE 'paramfile.h'

      REAL     xlgpsi,psi,s1,omega_fid,omega,Gcorr
      
      REAL      M0,z0,s0,dc0,delta_dc
      COMMON /fpars/ M0,z0,s0,dc0,delta_dc
            
      REAL     variance
      EXTERNAL variance
      
c---

      psi = 10.0**xlgpsi

      s1 = variance(psi*M0)**2
      
      omega_fid = apar1 * (1.0-apar2*xlgpsi)**apar3 * 
     &            (1.0-psi**apar4)**apar5

      omega = delta_dc/SQRT(s1 - s0)
      
      Gcorr = 0.57 * (s1/s0)**0.19 * (dc0/SQRT(s0))**(-0.01)
      Gcorr = Gcorr**0.4
       
      find_psi = omega*Gcorr - omega_fid
            
      END

c**********************************************************************

      SUBROUTINE get_Vmax_rat(f,z,Vmaxrat,c1)
c----------------------------------------------------------------------
c
c  Copmute Vmax(z)/Vmax(z=0) for halo of mass fM at redshift z
c  Here M is defined as the halo mass at z=0
c
c----------------------------------------------------------------------
 
      INCLUDE 'paramfile.h'

      REAL   f,M,z,fac1,fac2,c0,c1
      REAL   Vvirrat,VmaxVvir0,VmaxVvir1,Vmaxrat
         
      REAL     xH,Delta_crit,get_conc
      EXTERNAL xH,Delta_crit,get_conc
      
c---

      c0 = get_conc(1.0,0.0)
      c1 = get_conc(f,z)
      
      fac1 = (xH(z)/xH_0)**(1.0/3.0)
      fac2 = (Delta_crit(z)/Delta_crit(0.0))**(1.0/6.0)
      Vvirrat = f**(1.0/3.0) * fac1 * fac2
      
      VmaxVvir0 = SQRT(c0 / (ALOG(1.0+c0) - c0/(1.0+c0)))
      VmaxVvir1 = SQRT(c1 / (ALOG(1.0+c1) - c1/(1.0+c1)))
               
      Vmaxrat = ALOG10((VmaxVvir1/VmaxVvir0) * Vvirrat)
      
      RETURN
      END 
       
c**********************************************************************

      REAL FUNCTION get_conc(Mfrac,z)
c----------------------------------------------------------------------
c
c  Copmute concentration for halo of mass Mfrac*M0 at redshift z.
c  We use the modified version of Zhao+09.
c
c----------------------------------------------------------------------
 
      INCLUDE 'paramfile.h'

      REAL    Mfrac,z,xlgpsi,tnow,tfour,zfour
    
      REAL     time
      EXTERNAL time
               
c---

      xlgpsi = ALOG10(0.04 * Mfrac)
      IF (xlgpsi.LT.xlgMAH(Nz)) THEN
        get_conc = 4.0
      ELSE
        CALL linintpol(xlgMAH,zz,Nz,xlgpsi,zfour)
        tnow  = time(z) 
        tfour = time(zfour)
        get_conc = 4.0 * (1.0 + (tnow/(3.4*tfour))**6.5)**(1.0/8.0)
      END IF
      
      END
      
c**********************************************************************

      REAL FUNCTION dzdlnt(z)
c----------------------------------------------------------------------
c
c  The derivative dz/dlnt at given z
c
c----------------------------------------------------------------------
 
      INCLUDE 'paramfile.h'

      REAL   eps
      PARAMETER  (eps=0.1)

      REAL   z,z1,z2,t1,t2
      
      EXTERNAL time
      REAL     time      

c---

      z1 = (1.0-eps) * z
      z2 = (1.0+eps) * z
      z1 = MAX(z1,0.0)
      
      t1 = time(z1) * xH_0_recip
      t2 = time(z2) * xH_0_recip
      
      dzdlnt = (z2-z1) / (ALOG(t2) - ALOG(t1))
      
      END 
       
c**********************************************************************

      SUBROUTINE read_parameters
c----------------------------------------------------------------------
c
c  Subroutine to read in global parameters
c
c----------------------------------------------------------------------
 
      INCLUDE 'paramfile.h'

      REAL      M0,z0,s0,dc0,delta_dc
      COMMON /fpars/ M0,z0,s0,dc0,delta_dc

c---

c---read parameters of cosmological parameters

      WRITE(*,*)' Give Omega_{m,0} : '
      READ(*,*)omega_0
      WRITE(*,*)' '

      WRITE(*,*)' Give h (=H_0/100) : '
      READ(*,*)xhubble
      WRITE(*,*)' '

      WRITE(*,*)' Give sigma_8 : '
      READ(*,*)sigma8
      WRITE(*,*)' '

      WRITE(*,*)' Give nspec (Harrisson-Zeldovich = 1.0)'
      READ(*,*)nspec
      WRITE(*,*)' '

      WRITE(*,*)' Give Omega_b h^2 : '
      READ(*,*)omega_b_h2
      WRITE(*,*)' '

c---compute others

      omega_lambda = 1.0 - omega_0
      omega_b = omega_b_h2/(xhubble**2.0)

c---read halo mass and redshift

      WRITE(*,*)' Give halo mass at redshift z_0 (in Msun/h)'
      READ(*,*)M0
      WRITE(*,*)' '
      
      WRITE(*,*)' Give redshift z_0'
      READ(*,*)z0
      WRITE(*,*)' '
              
c---write to screen
   
      WRITE(*,*)'-------------------------------------------'
      WRITE(*,77)'         Omega_{m,0} =',omega_0
      WRITE(*,77)'         Omega_{L,0} =',omega_lambda
      WRITE(*,77)'         Omega_{b,0} =',omega_b
      WRITE(*,77)'  H_0/(100 km/s/Mpc) =',xhubble  
      WRITE(*,77)'             sigma_8 =',sigma8
      WRITE(*,77)'                 n_s =',nspec      
      WRITE(*,77)'            log[M_0] =',ALOG10(M0)
      WRITE(*,77)'                 z_0 =',z0
      WRITE(*,*)'-------------------------------------------'
      WRITE(*,*)' '

77    FORMAT(A24,1X,F7.4)

      RETURN
      END

c**********************************************************************

      SUBROUTINE linintpol(xa,ya,N,x,y)

      IMPLICIT NONE

      INTEGER N,j
      REAL    xa(N),ya(N),x,y,x1,x2,y1,y2

c---

      CALL locate(xa,N,x,j)
      x1 = xa(j)
      x2 = xa(j+1)
      y1 = ya(j)
      y2 = ya(j+1)

      y = y1 + ((x-x1)/(x2-x1)) * (y2-y1)

      RETURN
      END      

c**********************************************************************

      SUBROUTINE locate(xx,n,x,j)

      IMPLICIT NONE
      
      INTEGER j,n,jl,jm,ju
      REAL x,xx(n)

c---

      jl=0
      ju=n+1
10    IF (ju-jl.GT.1) THEN
        jm=(ju+jl)/2
        IF ((xx(n).GT.xx(1)).EQV.(x.GT.xx(jm))) THEN
          jl=jm
        ELSE
          ju=jm
        END IF
      GOTO 10
      END IF
      j=jl
      
      RETURN
      END

c**********************************************************************

      REAL FUNCTION XEXP(x)
c--------------------------------------------------------------------
c
c Auxialiary function to compute EXP(x)
c
c--------------------------------------------------------------------

      REAL    x

c---

      IF (x.LT.-40.0) THEN
        XEXP = 0.0
      ELSE
        XEXP = EXP(x)
      END IF

      END

c**********************************************************************

      SUBROUTINE Terminate(message)
c--------------------------------------------------------------------
c
c  Output error message and terminate program
c
c--------------------------------------------------------------------

      IMPLICIT NONE

      character  message*(*)

c---

      WRITE(*,'(A)')message

      STOP

      RETURN
      END

c**********************************************************************

      FUNCTION zriddr(func,x1,x2,xacc)
      INTEGER MAXIT
      REAL zriddr,x1,x2,xacc,func,UNUSED
      PARAMETER (MAXIT=60,UNUSED=-1.11E30)
      EXTERNAL func
      INTEGER j
      REAL fh,fl,fm,fnew,s,xh,xl,xm,xnew
      fl=func(x1)
      fh=func(x2)
      if((fl.gt.0..and.fh.lt.0.).or.(fl.lt.0..and.fh.gt.0.))then
        xl=x1
        xh=x2
        zriddr=UNUSED
        do 11 j=1,MAXIT
          xm=0.5*(xl+xh)
          fm=func(xm)
          s=sqrt(fm**2-fl*fh)
          if(s.eq.0.)return
          xnew=xm+(xm-xl)*(sign(1.,fl-fh)*fm/s)
          if (abs(xnew-zriddr).le.xacc) return
          zriddr=xnew
          fnew=func(zriddr)
          if (fnew.eq.0.) return
          if(sign(fm,fnew).ne.fm) then
            xl=xm
            fl=fm
            xh=zriddr
            fh=fnew
          else if(sign(fl,fnew).ne.fl) then
            xh=zriddr
            fh=fnew
          else if(sign(fh,fnew).ne.fh) then
            xl=zriddr
            fl=fnew
          else
            CALL Terminate('never get here in zriddr')
          endif
          if(abs(xh-xl).le.xacc) return
11      continue
        CALL Terminate('zriddr exceed maximum iterations')
      else if (fl.eq.0.) then
        zriddr=x1
      else if (fh.eq.0.) then
        zriddr=x2
      else 
        WRITE(*,*)' ALARM IN ZRIDDR ',x1,x2,fl,fh
        CALL Terminate('root must be bracketed in zriddr')
      endif
      return
      END

