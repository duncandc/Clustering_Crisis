c**********************************************************************
c           NUMERICAL ROUTINES FROM QUADPACK
c
c  http://www.netlib.org/quadpack/readme
c
c The following routines have been included below:
c
c   qags.f
c   qagse.f
c   qag.f
c   qage.f
c   qagi.f
c   qawf.f
c   qng.f
c   qawfe.f
c   qagie.f  
c   qawoe.f  
c   qcheb.f
c   qk15i.f
c   qpsrt.f
c   qc25f.f
c   qelg.f 
c   qk15w.f
c   qwgtf.f
c   qk15.f
c   qk21.f
c   qk31.f
c   qk41.f
c   qk51.f
c   qk61.f
c
c**********************************************************************

      subroutine qags(f,a,b,epsabs,epsrel,result,abserr,neval,ier,
     *   limit,lenw,last,iwork,work)
c***begin prologue  qags
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a1a1
c***keywords  automatic integrator, general-purpose,
c             (end-point) singularities, extrapolation,
c             globally adaptive
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & prog. div. - k.u.leuven
c***purpose  the routine calculates an approximation result to a given
c            definite integral  i = integral of f over (a,b),
c            hopefully satisfying following claim for accuracy
c            abs(i-result).le.max(epsabs,epsrel*abs(i)).
c***description
c
c        computation of a definite integral
c        standard fortran subroutine
c        real version
c
c
c        parameters
c         on entry
c            f      - real
c                     function subprogram defining the integrand
c                     function f(x). the actual name for f needs to be
c                     declared e x t e r n a l in the driver program.
c
c            a      - real
c                     lower limit of integration
c
c            b      - real
c                     upper limit of integration
c
c            epsabs - real
c                     absolute accuracy requested
c            epsrel - real
c                     relative accuracy requested
c                     if  epsabs.le.0
c                     and epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
c                     the routine will end with ier = 6.
c
c         on return
c            result - real
c                     approximation to the integral
c
c            abserr - real
c                     estimate of the modulus of the absolute error,
c                     which should equal or exceed abs(i-result)
c
c            neval  - integer
c                     number of integrand evaluations
c
c            ier    - integer
c                     ier = 0 normal and reliable termination of the
c                             routine. it is assumed that the requested
c                             accuracy has been achieved.
c                     ier.gt.0 abnormal termination of the routine
c                             the estimates for integral and error are
c                             less reliable. it is assumed that the
c                             requested accuracy has not been achieved.
c            error messages
c                     ier = 1 maximum number of subdivisions allowed
c                             has been achieved. one can allow more sub-
c                             divisions by increasing the value of limit
c                             (and taking the according dimension
c                             adjustments into account. however, if
c                             this yields no improvement it is advised
c                             to analyze the integrand in order to
c                             determine the integration difficulties. if
c                             the position of a local difficulty can be
c                             determined (e.g. singularity,
c                             discontinuity within the interval) one
c                             will probably gain from splitting up the
c                             interval at this point and calling the
c                             integrator on the subranges. if possible,
c                             an appropriate special-purpose integrator
c                             should be used, which is designed for
c                             handling the type of difficulty involved.
c                         = 2 the occurrence of roundoff error is detec-
c                             ted, which prevents the requested
c                             tolerance from being achieved.
c                             the error may be under-estimated.
c                         = 3 extremely bad integrand behaviour
c                             occurs at some points of the integration
c                             interval.
c                         = 4 the algorithm does not converge.
c                             roundoff error is detected in the
c                             extrapolation table. it is presumed that
c                             the requested tolerance cannot be
c                             achieved, and that the returned result is
c                             the best which can be obtained.
c                         = 5 the integral is probably divergent, or
c                             slowly convergent. it must be noted that
c                             divergence can occur with any other value
c                             of ier.
c                         = 6 the input is invalid, because
c                             (epsabs.le.0 and
c                              epsrel.lt.max(50*rel.mach.acc.,0.5d-28)
c                             or limit.lt.1 or lenw.lt.limit*4.
c                             result, abserr, neval, last are set to
c                             zero.except when limit or lenw is invalid,
c                             iwork(1), work(limit*2+1) and
c                             work(limit*3+1) are set to zero, work(1)
c                             is set to a and work(limit+1) to b.
c
c         dimensioning parameters
c            limit - integer
c                    dimensioning parameter for iwork
c                    limit determines the maximum number of subintervals
c                    in the partition of the given integration interval
c                    (a,b), limit.ge.1.
c                    if limit.lt.1, the routine will end with ier = 6.
c
c            lenw  - integer
c                    dimensioning parameter for work
c                    lenw must be at least limit*4.
c                    if lenw.lt.limit*4, the routine will end
c                    with ier = 6.
c
c            last  - integer
c                    on return, last equals the number of subintervals
c                    produced in the subdivision process, detemines the
c                    number of significant elements actually in the work
c                    arrays.
c
c         work arrays
c            iwork - integer
c                    vector of dimension at least limit, the first k
c                    elements of which contain pointers
c                    to the error estimates over the subintervals
c                    such that work(limit*3+iwork(1)),... ,
c                    work(limit*3+iwork(k)) form a decreasing
c                    sequence, with k = last if last.le.(limit/2+2),
c                    and k = limit+1-last otherwise
c
c            work  - real
c                    vector of dimension at least lenw
c                    on return
c                    work(1), ..., work(last) contain the left
c                     end-points of the subintervals in the
c                     partition of (a,b),
c                    work(limit+1), ..., work(limit+last) contain
c                     the right end-points,
c                    work(limit*2+1), ..., work(limit*2+last) contain
c                     the integral approximations over the subintervals,
c                    work(limit*3+1), ..., work(limit*3+last)
c                     contain the error estimates.
c
c
c
c***references  (none)
c***routines called  qagse,xerror
c***end prologue  qags
c
c
      real a,abserr,b,epsabs,epsrel,f,result,work
      integer ier,iwork,lenw,limit,lvl,l1,l2,l3,neval
c
      dimension iwork(limit),work(lenw)
c
      external f
c
c         check validity of limit and lenw.
c
c***first executable statement  qags
      ier = 6
      neval = 0
      last = 0
      result = 0.0e+00
      abserr = 0.0e+00
      if(limit.lt.1.or.lenw.lt.limit*4) go to 10
c
c         prepare call for qagse.
c
      l1 = limit+1
      l2 = limit+l1
      l3 = limit+l2
c
      call qagse(f,a,b,epsabs,epsrel,limit,result,abserr,neval,
     *  ier,work(1),work(l1),work(l2),work(l3),iwork,last)
c
c         call error handler if necessary.
c
      lvl = 0
10    if(ier.eq.6) lvl = 1
c      if(ier.ne.0) WRITE(*,*)'ABNORMAL ERROR RETURN qags ',ier
      return
      end

c**********************************************************************

      subroutine qagse(f,a,b,epsabs,epsrel,limit,result,abserr,neval,
     *   ier,alist,blist,rlist,elist,iord,last)
c***begin prologue  qagse
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a1a1
c***keywords  automatic integrator, general-purpose,
c             (end point) singularities, extrapolation,
c             globally adaptive
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  the routine calculates an approximation result to a given
c            definite integral i = integral of f over (a,b),
c            hopefully satisfying following claim for accuracy
c            abs(i-result).le.max(epsabs,epsrel*abs(i)).
c***description
c
c        computation of a definite integral
c        standard fortran subroutine
c        real version
c
c        parameters
c         on entry
c            f      - real
c                     function subprogram defining the integrand
c                     function f(x). the actual name for f needs to be
c                     declared e x t e r n a l in the driver program.
c
c            a      - real
c                     lower limit of integration
c
c            b      - real
c                     upper limit of integration
c
c            epsabs - real
c                     absolute accuracy requested
c            epsrel - real
c                     relative accuracy requested
c                     if  epsabs.le.0
c                     and epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
c                     the routine will end with ier = 6.
c
c            limit  - integer
c                     gives an upperbound on the number of subintervals
c                     in the partition of (a,b)
c
c         on return
c            result - real
c                     approximation to the integral
c
c            abserr - real
c                     estimate of the modulus of the absolute error,
c                     which should equal or exceed abs(i-result)
c
c            neval  - integer
c                     number of integrand evaluations
c
c            ier    - integer
c                     ier = 0 normal and reliable termination of the
c                             routine. it is assumed that the requested
c                             accuracy has been achieved.
c                     ier.gt.0 abnormal termination of the routine
c                             the estimates for integral and error are
c                             less reliable. it is assumed that the
c                             requested accuracy has not been achieved.
c            error messages
c                         = 1 maximum number of subdivisions allowed
c                             has been achieved. one can allow more sub-
c                             divisions by increasing the value of limit
c                             (and taking the according dimension
c                             adjustments into account). however, if
c                             this yields no improvement it is advised
c                             to analyze the integrand in order to
c                             determine the integration difficulties. if
c                             the position of a local difficulty can be
c                             determined (e.g. singularity,
c                             discontinuity within the interval) one
c                             will probably gain from splitting up the
c                             interval at this point and calling the
c                             integrator on the subranges. if possible,
c                             an appropriate special-purpose integrator
c                             should be used, which is designed for
c                             handling the type of difficulty involved.
c                         = 2 the occurrence of roundoff error is detec-
c                             ted, which prevents the requested
c                             tolerance from being achieved.
c                             the error may be under-estimated.
c                         = 3 extremely bad integrand behaviour
c                             occurs at some points of the integration
c                             interval.
c                         = 4 the algorithm does not converge.
c                             roundoff error is detected in the
c                             extrapolation table.
c                             it is presumed that the requested
c                             tolerance cannot be achieved, and that the
c                             returned result is the best which can be
c                             obtained.
c                         = 5 the integral is probably divergent, or
c                             slowly convergent. it must be noted that
c                             divergence can occur with any other value
c                             of ier.
c                         = 6 the input is invalid, because
c                             epsabs.le.0 and
c                             epsrel.lt.max(50*rel.mach.acc.,0.5d-28).
c                             result, abserr, neval, last, rlist(1),
c                             iord(1) and elist(1) are set to zero.
c                             alist(1) and blist(1) are set to a and b
c                             respectively.
c
c            alist  - real
c                     vector of dimension at least limit, the first
c                      last  elements of which are the left end points
c                     of the subintervals in the partition of the
c                     given integration range (a,b)
c
c            blist  - real
c                     vector of dimension at least limit, the first
c                      last  elements of which are the right end points
c                     of the subintervals in the partition of the given
c                     integration range (a,b)
c
c            rlist  - real
c                     vector of dimension at least limit, the first
c                      last  elements of which are the integral
c                     approximations on the subintervals
c
c            elist  - real
c                     vector of dimension at least limit, the first
c                      last  elements of which are the moduli of the
c                     absolute error estimates on the subintervals
c
c            iord   - integer
c                     vector of dimension at least limit, the first k
c                     elements of which are pointers to the
c                     error estimates over the subintervals,
c                     such that elist(iord(1)), ..., elist(iord(k))
c                     form a decreasing sequence, with k = last
c                     if last.le.(limit/2+2), and k = limit+1-last
c                     otherwise
c
c            last   - integer
c                     number of subintervals actually produced in the
c                     subdivision process
c
c***references  (none)
c***routines called  qelg,qk21,qpsrt,r1mach
c***end prologue  qagse
c
      real a,abseps,abserr,alist,area,area1,area12,area2,a1,
     *  a2,b,blist,b1,b2,correc,defabs,defab1,defab2,r1mach,
     *  dres,elist,epmach,epsabs,epsrel,erlarg,erlast,errbnd,
     *  errmax,error1,error2,erro12,errsum,ertest,f,oflow,resabs,
     *  reseps,result,res3la,rlist,rlist2,small,uflow
      integer id,ier,ierro,iord,iroff1,iroff2,iroff3,jupbnd,k,ksgn,
     *  ktmin,last,limit,maxerr,neval,nres,nrmax,numrl2
      logical extrap,noext
c
      dimension alist(limit),blist(limit),elist(limit),iord(limit),
     * res3la(3),rlist(limit),rlist2(52)
c
      external f
c
c            the dimension of rlist2 is determined by the value of
c            limexp in subroutine qelg (rlist2 should be of dimension
c            (limexp+2) at least).
c
c            list of major variables
c            -----------------------
c
c           alist     - list of left end points of all subintervals
c                       considered up to now
c           blist     - list of right end points of all subintervals
c                       considered up to now
c           rlist(i)  - approximation to the integral over
c                       (alist(i),blist(i))
c           rlist2    - array of dimension at least limexp+2
c                       containing the part of the epsilon table
c                       which is still needed for further computations
c           elist(i)  - error estimate applying to rlist(i)
c           maxerr    - pointer to the interval with largest error
c                       estimate
c           errmax    - elist(maxerr)
c           erlast    - error on the interval currently subdivided
c                       (before that subdivision has taken place)
c           area      - sum of the integrals over the subintervals
c           errsum    - sum of the errors over the subintervals
c           errbnd    - requested accuracy max(epsabs,epsrel*
c                       abs(result))
c           *****1    - variable for the left interval
c           *****2    - variable for the right interval
c           last      - index for subdivision
c           nres      - number of calls to the extrapolation routine
c           numrl2    - number of elements currently in rlist2. if an
c                       appropriate approximation to the compounded
c                       integral has been obtained it is put in
c                       rlist2(numrl2) after numrl2 has been increased
c                       by one.
c           small     - length of the smallest interval considered
c                       up to now, multiplied by 1.5
c           erlarg    - sum of the errors over the intervals larger
c                       than the smallest interval considered up to now
c           extrap    - logical variable denoting that the routine
c                       is attempting to perform extrapolation
c                       i.e. before subdividing the smallest interval
c                       we try to decrease the value of erlarg.
c           noext     - logical variable denoting that extrapolation
c                       is no longer allowed (true value)
c
c            machine dependent constants
c            ---------------------------
c
c           epmach is the largest relative spacing.
c           uflow is the smallest positive magnitude.
c           oflow is the largest positive magnitude.
c
c***first executable statement  qagse
      epmach = r1mach(4)
c
c            test on validity of parameters
c            ------------------------------
      ier = 0
      neval = 0
      last = 0
      result = 0.0e+00
      abserr = 0.0e+00
      alist(1) = a
      blist(1) = b
      rlist(1) = 0.0e+00
      elist(1) = 0.0e+00
      if(epsabs.le.0.0e+00.and.epsrel.lt.amax1(0.5e+02*epmach,0.5e-14))
     *   ier = 6
      if(ier.eq.6) go to 999
c
c           first approximation to the integral
c           -----------------------------------
c
      uflow = r1mach(1)
      oflow = r1mach(2)
      ierro = 0
      call qk21(f,a,b,result,abserr,defabs,resabs)
c
c           test on accuracy.
c
      dres = abs(result)
      errbnd = amax1(epsabs,epsrel*dres)
      last = 1
      rlist(1) = result
      elist(1) = abserr
      iord(1) = 1
      if(abserr.le.1.0e+02*epmach*defabs.and.abserr.gt.
     *  errbnd) ier = 2
      if(limit.eq.1) ier = 1
      if(ier.ne.0.or.(abserr.le.errbnd.and.abserr.ne.resabs).or.
     *  abserr.eq.0.0e+00) go to 140
c
c           initialization
c           --------------
c
      rlist2(1) = result
      errmax = abserr
      maxerr = 1
      area = result
      errsum = abserr
      abserr = oflow
      nrmax = 1
      nres = 0
      numrl2 = 2
      ktmin = 0
      extrap = .false.
      noext = .false.
      iroff1 = 0
      iroff2 = 0
      iroff3 = 0
      ksgn = -1
      if(dres.ge.(0.1e+01-0.5e+02*epmach)*defabs) ksgn = 1
c
c           main do-loop
c           ------------
c
      do 90 last = 2,limit
c
c           bisect the subinterval with the nrmax-th largest
c           error estimate.
c
        a1 = alist(maxerr)
        b1 = 0.5e+00*(alist(maxerr)+blist(maxerr))
        a2 = b1
        b2 = blist(maxerr)
        erlast = errmax
        call qk21(f,a1,b1,area1,error1,resabs,defab1)
        call qk21(f,a2,b2,area2,error2,resabs,defab2)
c
c           improve previous approximations to integral
c           and error and test for accuracy.
c
        area12 = area1+area2
        erro12 = error1+error2
        errsum = errsum+erro12-errmax
        area = area+area12-rlist(maxerr)
        if(defab1.eq.error1.or.defab2.eq.error2) go to 15
        if(abs(rlist(maxerr)-area12).gt.0.1e-04*abs(area12)
     *  .or.erro12.lt.0.99e+00*errmax) go to 10
        if(extrap) iroff2 = iroff2+1
        if(.not.extrap) iroff1 = iroff1+1
   10   if(last.gt.10.and.erro12.gt.errmax) iroff3 = iroff3+1
   15   rlist(maxerr) = area1
        rlist(last) = area2
        errbnd = amax1(epsabs,epsrel*abs(area))
c
c           test for roundoff error and eventually
c           set error flag.
c
        if(iroff1+iroff2.ge.10.or.iroff3.ge.20) ier = 2
        if(iroff2.ge.5) ierro = 3
c
c           set error flag in the case that the number of
c           subintervals equals limit.
c
        if(last.eq.limit) ier = 1
c
c           set error flag in the case of bad integrand behaviour
c           at a point of the integration range.
c
        if(amax1(abs(a1),abs(b2)).le.(0.1e+01+0.1e+03*epmach)*
     *  (abs(a2)+0.1e+04*uflow)) ier = 4
c
c           append the newly-created intervals to the list.
c
        if(error2.gt.error1) go to 20
        alist(last) = a2
        blist(maxerr) = b1
        blist(last) = b2
        elist(maxerr) = error1
        elist(last) = error2
        go to 30
   20   alist(maxerr) = a2
        alist(last) = a1
        blist(last) = b1
        rlist(maxerr) = area2
        rlist(last) = area1
        elist(maxerr) = error2
        elist(last) = error1
c
c           call subroutine qpsrt to maintain the descending ordering
c           in the list of error estimates and select the
c           subinterval with nrmax-th largest error estimate (to be
c           bisected next).
c
   30   call qpsrt(limit,last,maxerr,errmax,elist,iord,nrmax)
c ***jump out of do-loop
        if(errsum.le.errbnd) go to 115
c ***jump out of do-loop
        if(ier.ne.0) go to 100
        if(last.eq.2) go to 80
        if(noext) go to 90
        erlarg = erlarg-erlast
        if(abs(b1-a1).gt.small) erlarg = erlarg+erro12
        if(extrap) go to 40
c
c           test whether the interval to be bisected next is the
c           smallest interval.
c
        if(abs(blist(maxerr)-alist(maxerr)).gt.small) go to 90
        extrap = .true.
        nrmax = 2
   40   if(ierro.eq.3.or.erlarg.le.ertest) go to 60
c
c           the smallest interval has the largest error.
c           before bisecting decrease the sum of the errors
c           over the larger intervals (erlarg) and perform
c           extrapolation.
c
        id = nrmax
        jupbnd = last
        if(last.gt.(2+limit/2)) jupbnd = limit+3-last
        do 50 k = id,jupbnd
          maxerr = iord(nrmax)
          errmax = elist(maxerr)
c ***jump out of do-loop
          if(abs(blist(maxerr)-alist(maxerr)).gt.small) go to 90
          nrmax = nrmax+1
   50   continue
c
c           perform extrapolation.
c
   60   numrl2 = numrl2+1
        rlist2(numrl2) = area
        call qelg(numrl2,rlist2,reseps,abseps,res3la,nres)
        ktmin = ktmin+1
        if(ktmin.gt.5.and.abserr.lt.0.1e-02*errsum) ier = 5
        if(abseps.ge.abserr) go to 70
        ktmin = 0
        abserr = abseps
        result = reseps
        correc = erlarg
        ertest = amax1(epsabs,epsrel*abs(reseps))
c ***jump out of do-loop
        if(abserr.le.ertest) go to 100
c
c           prepare bisection of the smallest interval.
c
   70   if(numrl2.eq.1) noext = .true.
        if(ier.eq.5) go to 100
        maxerr = iord(1)
        errmax = elist(maxerr)
        nrmax = 1
        extrap = .false.
        small = small*0.5e+00
        erlarg = errsum
        go to 90
   80   small = abs(b-a)*0.375e+00
        erlarg = errsum
        ertest = errbnd
        rlist2(2) = area
   90 continue
c
c           set final result and error estimate.
c           ------------------------------------
c
  100 if(abserr.eq.oflow) go to 115
      if(ier+ierro.eq.0) go to 110
      if(ierro.eq.3) abserr = abserr+correc
      if(ier.eq.0) ier = 3
      if(result.ne.0.0e+00.and.area.ne.0.0e+00) go to 105
      if(abserr.gt.errsum) go to 115
      if(area.eq.0.0e+00) go to 130
      go to 110
  105 if(abserr/abs(result).gt.errsum/abs(area)) go to 115
c
c           test on divergence.
c
  110 if(ksgn.eq.(-1).and.amax1(abs(result),abs(area)).le.
     * defabs*0.1e-01) go to 130
      if(0.1e-01.gt.(result/area).or.(result/area).gt.0.1e+03
     * .or.errsum.gt.abs(area)) ier = 6
      go to 130
c
c           compute global integral sum.
c
  115 result = 0.0e+00
      do 120 k = 1,last
         result = result+rlist(k)
  120 continue
      abserr = errsum
  130 if(ier.gt.2) ier = ier-1
  140 neval = 42*last-21
  999 return
      end

c**********************************************************************


      subroutine qag(f,a,b,epsabs,epsrel,key,result,abserr,neval,ier,
     *    limit,lenw,last,iwork,work)
c***begin prologue  qag
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a1a1
c***keywords  automatic integrator, general-purpose,
c             integrand examinator, globally adaptive,
c             gauss-kronrod
c***author  piessens,robert,appl. math. & progr. div - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  the routine calculates an approximation result to a given
c            definite integral i = integral of f over (a,b),
c            hopefully satisfying following claim for accuracy
c            abs(i-result)le.max(epsabs,epsrel*abs(i)).
c***description
c
c        computation of a definite integral
c        standard fortran subroutine
c        real version
c
c            f      - real
c                     function subprogam defining the integrand
c                     function f(x). the actual name for f needs to be
c                     declared e x t e r n a l in the driver program.
c
c            a      - real
c                     lower limit of integration
c
c            b      - real
c                     upper limit of integration
c
c            epsabs - real
c                     absolute accuracy requested
c            epsrel - real
c                     relative accuracy requested
c                     if  epsabs.le.0
c                     and epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
c                     the routine will end with ier = 6.
c
c            key    - integer
c                     key for choice of local integration rule
c                     a gauss-kronrod pair is used with
c                       7 - 15 points if key.lt.2,
c                      10 - 21 points if key = 2,
c                      15 - 31 points if key = 3,
c                      20 - 41 points if key = 4,
c                      25 - 51 points if key = 5,
c                      30 - 61 points if key.gt.5.
c
c         on return
c            result - real
c                     approximation to the integral
c
c            abserr - real
c                     estimate of the modulus of the absolute error,
c                     which should equal or exceed abs(i-result)
c
c            neval  - integer
c                     number of integrand evaluations
c
c            ier    - integer
c                     ier = 0 normal and reliable termination of the
c                             routine. it is assumed that the requested
c                             accuracy has been achieved.
c                     ier.gt.0 abnormal termination of the routine
c                             the estimates for result and error are
c                             less reliable. it is assumed that the
c                             requested accuracy has not been achieved.
c                      error messages
c                     ier = 1 maximum number of subdivisions allowed
c                             has been achieved. one can allow more
c                             subdivisions by increasing the value of
c                             limit (and taking the according dimension
c                             adjustments into account). however, if
c                             this yield no improvement it is advised
c                             to analyze the integrand in order to
c                             determine the integration difficulaties.
c                             if the position of a local difficulty can
c                             be determined (i.e.singularity,
c                             discontinuity within the interval) one
c                             will probably gain from splitting up the
c                             interval at this point and calling the
c                             integrator on the subranges. if possible,
c                             an appropriate special-purpose integrator
c                             should be used which is designed for
c                             handling the type of difficulty involved.
c                         = 2 the occurrence of roundoff error is
c                             detected, which prevents the requested
c                             tolerance from being achieved.
c                         = 3 extremely bad integrand behaviour occurs
c                             at some points of the integration
c                             interval.
c                         = 6 the input is invalid, because
c                             (epsabs.le.0 and
c                              epsrel.lt.max(50*rel.mach.acc.,0.5d-28))
c                             or limit.lt.1 or lenw.lt.limit*4.
c                             result, abserr, neval, last are set
c                             to zero.
c                             except when lenw is invalid, iwork(1),
c                             work(limit*2+1) and work(limit*3+1) are
c                             set to zero, work(1) is set to a and
c                             work(limit+1) to b.
c
c         dimensioning parameters
c            limit - integer
c                    dimensioning parameter for iwork
c                    limit determines the maximum number of subintervals
c                    in the partition of the given integration interval
c                    (a,b), limit.ge.1.
c                    if limit.lt.1, the routine will end with ier = 6.
c
c            lenw  - integer
c                    dimensioning parameter for work
c                    lenw must be at least limit*4.
c                    if lenw.lt.limit*4, the routine will end with
c                    ier = 6.
c
c            last  - integer
c                    on return, last equals the number of subintervals
c                    produced in the subdivision process, which
c                    determines the number of significant elements
c                    actually in the work arrays.
c
c         work arrays
c            iwork - integer
c                    vector of dimension at least limit, the first k
c                    elements of which contain pointers to the error
c                    estimates over the subintervals, such that
c                    work(limit*3+iwork(1)),... , work(limit*3+iwork(k))
c                    form a decreasing sequence with k = last if
c                    last.le.(limit/2+2), and k = limit+1-last otherwise
c
c            work  - real
c                    vector of dimension at least lenw
c                    on return
c                    work(1), ..., work(last) contain the left end
c                    points of the subintervals in the partition of
c                     (a,b),
c                    work(limit+1), ..., work(limit+last) contain the
c                     right end points,
c                    work(limit*2+1), ..., work(limit*2+last) contain
c                     the integral approximations over the subintervals,
c                    work(limit*3+1), ..., work(limit*3+last) contain
c                     the error estimates.
c
c***references  (none)
c***routines called  qage,xerror
c***end prologue  qag
c
      real a,abserr,b,epsabs,epsrel,f,result,work
      integer ier,iwork,key,lenw,limit,lvl,l1,l2,l3,neval
c
      dimension iwork(limit),work(lenw)
c
      external f
c
c         check validity of lenw.
c
c***first executable statement  qag
      ier = 6
      neval = 0
      last = 0
      result = 0.0e+00
      abserr = 0.0e+00
      if(limit.lt.1.or.lenw.lt.limit*4) go to 10
c
c         prepare call for qage.
c
      l1 = limit+1
      l2 = limit+l1
      l3 = limit+l2
c
      call qage(f,a,b,epsabs,epsrel,key,limit,result,abserr,neval,
     *  ier,work(1),work(l1),work(l2),work(l3),iwork,last)
c
c         call error handler if necessary.
c
      lvl = 0
10    if(ier.eq.6) lvl = 1
c      if(ier.ne.0) WRITE(*,*)'ABNORMAL ERROR RETURN qag ',ier
      return
      end

c**********************************************************************

      subroutine qage(f,a,b,epsabs,epsrel,key,limit,result,abserr,
     *   neval,ier,alist,blist,rlist,elist,iord,last)
c***begin prologue  qage
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a1a1
c***keywords  automatic integrator, general-purpose,
c             integrand examinator, globally adaptive,
c             gauss-kronrod
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  the routine calculates an approximation result to a given
c            definite integral   i = integral of f over (a,b),
c            hopefully satisfying following claim for accuracy
c            abs(i-reslt).le.max(epsabs,epsrel*abs(i)).
c***description
c
c        computation of a definite integral
c        standard fortran subroutine
c        real version
c
c        parameters
c         on entry
c            f      - real
c                     function subprogram defining the integrand
c                     function f(x). the actual name for f needs to be
c                     declared e x t e r n a l in the driver program.
c
c            a      - real
c                     lower limit of integration
c
c            b      - real
c                     upper limit of integration
c
c            epsabs - real
c                     absolute accuracy requested
c            epsrel - real
c                     relative accuracy requested
c                     if  epsabs.le.0
c                     and epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
c                     the routine will end with ier = 6.
c
c            key    - integer
c                     key for choice of local integration rule
c                     a gauss-kronrod pair is used with
c                          7 - 15 points if key.lt.2,
c                         10 - 21 points if key = 2,
c                         15 - 31 points if key = 3,
c                         20 - 41 points if key = 4,
c                         25 - 51 points if key = 5,
c                         30 - 61 points if key.gt.5.
c
c            limit  - integer
c                     gives an upperbound on the number of subintervals
c                     in the partition of (a,b), limit.ge.1.
c
c         on return
c            result - real
c                     approximation to the integral
c
c            abserr - real
c                     estimate of the modulus of the absolute error,
c                     which should equal or exceed abs(i-result)
c
c            neval  - integer
c                     number of integrand evaluations
c
c            ier    - integer
c                     ier = 0 normal and reliable termination of the
c                             routine. it is assumed that the requested
c                             accuracy has been achieved.
c                     ier.gt.0 abnormal termination of the routine
c                             the estimates for result and error are
c                             less reliable. it is assumed that the
c                             requested accuracy has not been achieved.
c            error messages
c                     ier = 1 maximum number of subdivisions allowed
c                             has been achieved. one can allow more
c                             subdivisions by increasing the value
c                             of limit.
c                             however, if this yields no improvement it
c                             is rather advised to analyze the integrand
c                             in order to determine the integration
c                             difficulties. if the position of a local
c                             difficulty can be determined(e.g.
c                             singularity, discontinuity within the
c                             interval) one will probably gain from
c                             splitting up the interval at this point
c                             and calling the integrator on the
c                             subranges. if possible, an appropriate
c                             special-purpose integrator should be used
c                             which is designed for handling the type of
c                             difficulty involved.
c                         = 2 the occurrence of roundoff error is
c                             detected, which prevents the requested
c                             tolerance from being achieved.
c                         = 3 extremely bad integrand behaviour occurs
c                             at some points of the integration
c                             interval.
c                         = 6 the input is invalid, because
c                             (epsabs.le.0 and
c                              epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
c                             result, abserr, neval, last, rlist(1) ,
c                             elist(1) and iord(1) are set to zero.
c                             alist(1) and blist(1) are set to a and b
c                             respectively.
c
c            alist   - real
c                      vector of dimension at least limit, the first
c                       last  elements of which are the left
c                      end points of the subintervals in the partition
c                      of the given integration range (a,b)
c
c            blist   - real
c                      vector of dimension at least limit, the first
c                       last  elements of which are the right
c                      end points of the subintervals in the partition
c                      of the given integration range (a,b)
c
c            rlist   - real
c                      vector of dimension at least limit, the first
c                       last  elements of which are the
c                      integral approximations on the subintervals
c
c            elist   - real
c                      vector of dimension at least limit, the first
c                       last  elements of which are the moduli of the
c                      absolute error estimates on the subintervals
c
c            iord    - integer
c                      vector of dimension at least limit, the first k
c                      elements of which are pointers to the
c                      error estimates over the subintervals,
c                      such that elist(iord(1)), ...,
c                      elist(iord(k)) form a decreasing sequence,
c                      with k = last if last.le.(limit/2+2), and
c                      k = limit+1-last otherwise
c
c            last    - integer
c                      number of subintervals actually produced in the
c                      subdivision process
c
c***references  (none)
c***routines called  qk15,qk21,qk31,qk41,qk51,qk61,qpsrt,r1mach
c***end prologue  qage
c
      real a,abserr,alist,area,area1,area12,area2,a1,a2,b,blist,
     *  b1,b2,defabs,defab1,defab2,r1mach,elist,epmach,
     *  epsabs,epsrel,errbnd,errmax,error1,error2,erro12,errsum,f,
     *  resabs,result,rlist,uflow
      integer ier,iord,iroff1,iroff2,k,key,keyf,last,
     *  limit,maxerr,neval,nrmax
c
      dimension alist(limit),blist(limit),elist(limit),iord(limit),
     *  rlist(limit)
c
      external f
c
c            list of major variables
c            -----------------------
c
c           alist     - list of left end points of all subintervals
c                       considered up to now
c           blist     - list of right end points of all subintervals
c                       considered up to now
c           rlist(i)  - approximation to the integral over
c                      (alist(i),blist(i))
c           elist(i)  - error estimate applying to rlist(i)
c           maxerr    - pointer to the interval with largest
c                       error estimate
c           errmax    - elist(maxerr)
c           area      - sum of the integrals over the subintervals
c           errsum    - sum of the errors over the subintervals
c           errbnd    - requested accuracy max(epsabs,epsrel*
c                       abs(result))
c           *****1    - variable for the left subinterval
c           *****2    - variable for the right subinterval
c           last      - index for subdivision
c
c
c           machine dependent constants
c           ---------------------------
c
c           epmach  is the largest relative spacing.
c           uflow  is the smallest positive magnitude.
c
c***first executable statement  qage
      epmach = r1mach(4)
      uflow = r1mach(1)
c
c           test on validity of parameters
c           ------------------------------
c
      ier = 0
      neval = 0
      last = 0
      result = 0.0e+00
      abserr = 0.0e+00
      alist(1) = a
      blist(1) = b
      rlist(1) = 0.0e+00
      elist(1) = 0.0e+00
      iord(1) = 0
      if(epsabs.le.0.0e+00.and.
     *  epsrel.lt.amax1(0.5e+02*epmach,0.5e-14)) ier = 6
      if(ier.eq.6) go to 999
c
c           first approximation to the integral
c           -----------------------------------
c
      keyf = key
      if(key.le.0) keyf = 1
      if(key.ge.7) keyf = 6
      neval = 0
      if(keyf.eq.1) call qk15(f,a,b,result,abserr,defabs,resabs)
      if(keyf.eq.2) call qk21(f,a,b,result,abserr,defabs,resabs)
      if(keyf.eq.3) call qk31(f,a,b,result,abserr,defabs,resabs)
      if(keyf.eq.4) call qk41(f,a,b,result,abserr,defabs,resabs)
      if(keyf.eq.5) call qk51(f,a,b,result,abserr,defabs,resabs)
      if(keyf.eq.6) call qk61(f,a,b,result,abserr,defabs,resabs)
      last = 1
      rlist(1) = result
      elist(1) = abserr
      iord(1) = 1
c
c           test on accuracy.
c
      errbnd = amax1(epsabs,epsrel*abs(result))
      if(abserr.le.0.5e+02*epmach*defabs.and.abserr.gt.
     *  errbnd) ier = 2
      if(limit.eq.1) ier = 1
      if(ier.ne.0.or.(abserr.le.errbnd.and.abserr.ne.resabs)
     *  .or.abserr.eq.0.0e+00) go to 60
c
c           initialization
c           --------------
c
c
      errmax = abserr
      maxerr = 1
      area = result
      errsum = abserr
      nrmax = 1
      iroff1 = 0
      iroff2 = 0
c
c           main do-loop
c           ------------
c
      do 30 last = 2,limit
c
c           bisect the subinterval with the largest error estimate.
c
        a1 = alist(maxerr)
        b1 = 0.5e+00*(alist(maxerr)+blist(maxerr))
        a2 = b1
        b2 = blist(maxerr)
        if(keyf.eq.1) call qk15(f,a1,b1,area1,error1,resabs,defab1)
        if(keyf.eq.2) call qk21(f,a1,b1,area1,error1,resabs,defab1)
        if(keyf.eq.3) call qk31(f,a1,b1,area1,error1,resabs,defab1)
        if(keyf.eq.4) call qk41(f,a1,b1,area1,error1,resabs,defab1)
        if(keyf.eq.5) call qk51(f,a1,b1,area1,error1,resabs,defab1)
        if(keyf.eq.6) call qk61(f,a1,b1,area1,error1,resabs,defab1)
        if(keyf.eq.1) call qk15(f,a2,b2,area2,error2,resabs,defab2)
        if(keyf.eq.2) call qk21(f,a2,b2,area2,error2,resabs,defab2)
        if(keyf.eq.3) call qk31(f,a2,b2,area2,error2,resabs,defab2)
        if(keyf.eq.4) call qk41(f,a2,b2,area2,error2,resabs,defab2)
        if(keyf.eq.5) call qk51(f,a2,b2,area2,error2,resabs,defab2)
        if(keyf.eq.6) call qk61(f,a2,b2,area2,error2,resabs,defab2)
c
c           improve previous approximations to integral
c           and error and test for accuracy.
c
        neval = neval+1
        area12 = area1+area2
        erro12 = error1+error2
        errsum = errsum+erro12-errmax
        area = area+area12-rlist(maxerr)
        if(defab1.eq.error1.or.defab2.eq.error2) go to 5
        if(abs(rlist(maxerr)-area12).le.0.1e-04*abs(area12)
     *  .and.erro12.ge.0.99e+00*errmax) iroff1 = iroff1+1
        if(last.gt.10.and.erro12.gt.errmax) iroff2 = iroff2+1
    5   rlist(maxerr) = area1
        rlist(last) = area2
        errbnd = amax1(epsabs,epsrel*abs(area))
        if(errsum.le.errbnd) go to 8
c
c           test for roundoff error and eventually
c           set error flag.
c
        if(iroff1.ge.6.or.iroff2.ge.20) ier = 2
c
c           set error flag in the case that the number of
c           subintervals equals limit.
c
        if(last.eq.limit) ier = 1
c
c           set error flag in the case of bad integrand behaviour
c           at a point of the integration range.
c
        if(amax1(abs(a1),abs(b2)).le.(0.1e+01+0.1e+03*
     *  epmach)*(abs(a2)+0.1e+04*uflow)) ier = 3
c
c           append the newly-created intervals to the list.
c
    8   if(error2.gt.error1) go to 10
        alist(last) = a2
        blist(maxerr) = b1
        blist(last) = b2
        elist(maxerr) = error1
        elist(last) = error2
        go to 20
   10   alist(maxerr) = a2
        alist(last) = a1
        blist(last) = b1
        rlist(maxerr) = area2
        rlist(last) = area1
        elist(maxerr) = error2
        elist(last) = error1
c
c           call subroutine qpsrt to maintain the descending ordering
c           in the list of error estimates and select the
c           subinterval with the largest error estimate (to be
c           bisected next).
c
   20   call qpsrt(limit,last,maxerr,errmax,elist,iord,nrmax)
c ***jump out of do-loop
        if(ier.ne.0.or.errsum.le.errbnd) go to 40
   30 continue
c
c           compute final result.
c           ---------------------
c
   40 result = 0.0e+00
      do 50 k=1,last
        result = result+rlist(k)
   50 continue
      abserr = errsum
   60 if(keyf.ne.1) neval = (10*keyf+1)*(2*neval+1)
      if(keyf.eq.1) neval = 30*neval+15
  999 return
      end

c**********************************************************************

      subroutine qagi(f,bound,inf,epsabs,epsrel,result,abserr,neval,
     *   ier,limit,lenw,last,iwork,work)
c***begin prologue  qagi
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a3a1,h2a4a1
c***keywords  automatic integrator, infinite intervals,
c             general-purpose, transformation, extrapolation,
c             globally adaptive
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. -k.u.leuven
c***purpose  the routine calculates an approximation result to a given
c            integral   i = integral of f over (bound,+infinity)
c                    or i = integral of f over (-infinity,bound)
c                    or i = integral of f over (-infinity,+infinity)
c            hopefully satisfying following claim for accuracy
c            abs(i-result).le.max(epsabs,epsrel*abs(i)).
c***description
c
c        integration over infinite intervals
c        standard fortran subroutine
c
c        parameters
c         on entry
c            f      - real
c                     function subprogram defining the integrand
c                     function f(x). the actual name for f needs to be
c                     declared e x t e r n a l in the driver program.
c
c            bound  - real
c                     finite bound of integration range
c                     (has no meaning if interval is doubly-infinite)
c
c            inf    - integer
c                     indicating the kind of integration range involved
c                     inf = 1 corresponds to  (bound,+infinity),
c                     inf = -1            to  (-infinity,bound),
c                     inf = 2             to (-infinity,+infinity).
c
c            epsabs - real
c                     absolute accuracy requested
c            epsrel - real
c                     relative accuracy requested
c                     if  epsabs.le.0
c                     and epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
c                     the routine will end with ier = 6.
c
c
c         on return
c            result - real
c                     approximation to the integral
c
c            abserr - real
c                     estimate of the modulus of the absolute error,
c                     which should equal or exceed abs(i-result)
c
c            neval  - integer
c                     number of integrand evaluations
c
c            ier    - integer
c                     ier = 0 normal and reliable termination of the
c                             routine. it is assumed that the requested
c                             accuracy has been achieved.
c                   - ier.gt.0 abnormal termination of the routine. the
c                             estimates for result and error are less
c                             reliable. it is assumed that the requested
c                             accuracy has not been achieved.
c            error messages
c                     ier = 1 maximum number of subdivisions allowed
c                             has been achieved. one can allow more
c                             subdivisions by increasing the value of
c                             limit (and taking the according dimension
c                             adjustments into account). however, if
c                             this yields no improvement it is advised
c                             to analyze the integrand in order to
c                             determine the integration difficulties. if
c                             the position of a local difficulty can be
c                             determined (e.g. singularity,
c                             discontinuity within the interval) one
c                             will probably gain from splitting up the
c                             interval at this point and calling the
c                             integrator on the subranges. if possible,
c                             an appropriate special-purpose integrator
c                             should be used, which is designed for
c                             handling the type of difficulty involved.
c                         = 2 the occurrence of roundoff error is
c                             detected, which prevents the requested
c                             tolerance from being achieved.
c                             the error may be under-estimated.
c                         = 3 extremely bad integrand behaviour occurs
c                             at some points of the integration
c                             interval.
c                         = 4 the algorithm does not converge.
c                             roundoff error is detected in the
c                             extrapolation table.
c                             it is assumed that the requested tolerance
c                             cannot be achieved, and that the returned
c                             result is the best which can be obtained.
c                         = 5 the integral is probably divergent, or
c                             slowly convergent. it must be noted that
c                             divergence can occur with any other value
c                             of ier.
c                         = 6 the input is invalid, because
c                             (epsabs.le.0 and
c                              epsrel.lt.max(50*rel.mach.acc.,0.5d-28))
c                              or limit.lt.1 or leniw.lt.limit*4.
c                             result, abserr, neval, last are set to
c                             zero. exept when limit or leniw is
c                             invalid, iwork(1), work(limit*2+1) and
c                             work(limit*3+1) are set to zero, work(1)
c                             is set to a and work(limit+1) to b.
c
c         dimensioning parameters
c            limit - integer
c                    dimensioning parameter for iwork
c                    limit determines the maximum number of subintervals
c                    in the partition of the given integration interval
c                    (a,b), limit.ge.1.
c                    if limit.lt.1, the routine will end with ier = 6.
c
c            lenw  - integer
c                    dimensioning parameter for work
c                    lenw must be at least limit*4.
c                    if lenw.lt.limit*4, the routine will end
c                    with ier = 6.
c
c            last  - integer
c                    on return, last equals the number of subintervals
c                    produced in the subdivision process, which
c                    determines the number of significant elements
c                    actually in the work arrays.
c
c         work arrays
c            iwork - integer
c                    vector of dimension at least limit, the first
c                    k elements of which contain pointers
c                    to the error estimates over the subintervals,
c                    such that work(limit*3+iwork(1)),... ,
c                    work(limit*3+iwork(k)) form a decreasing
c                    sequence, with k = last if last.le.(limit/2+2), and
c                    k = limit+1-last otherwise
c
c            work  - real
c                    vector of dimension at least lenw
c                    on return
c                    work(1), ..., work(last) contain the left
c                     end points of the subintervals in the
c                     partition of (a,b),
c                    work(limit+1), ..., work(limit+last) contain
c                     the right end points,
c                    work(limit*2+1), ...,work(limit*2+last) contain the
c                     integral approximations over the subintervals,
c                    work(limit*3+1), ..., work(limit*3)
c                     contain the error estimates.
c***references  (none)
c***routines called  qagie,xerror
c***end prologue  qagi
c
      real   abserr,  epsabs,epsrel,f,result,work
      integer ier,iwork,    lenw,limit,lvl,l1,l2,l3,neval
c
      dimension iwork(limit),work(lenw)
c
      external f
c
c         check validity of limit and lenw.
c
c***first executable statement  qagi
      ier = 6
      neval = 0
      last = 0
      result = 0.0e+00
      abserr = 0.0e+00
      if(limit.lt.1.or.lenw.lt.limit*4) go to 10
c
c         prepare call for qagie.
c
      l1 = limit+1
      l2 = limit+l1
      l3 = limit+l2
c
      call qagie(f,bound,inf,epsabs,epsrel,limit,result,abserr,
     *  neval,ier,work(1),work(l1),work(l2),work(l3),iwork,last)
c
c         call error handler if necessary.
c
      lvl = 0
10    if(ier.eq.6) lvl = 1
ccc      if(ier.ne.0) WRITE(*,*)'ABNORMAL ERROR RETURN qagi',ier
      return
      end

c**********************************************************************

      subroutine qawf(f,a,omega,integr,epsabs,result,abserr,neval,ier,
     *   limlst,lst,leniw,maxp1,lenw,iwork,work)
c***begin prologue  qawf
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a3a1
c***keywords  automatic integrator, special-purpose,fourier
c             integral, integration between zeros with dqawoe,
c             convergence acceleration with dqext
c***author  piessens,robert ,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math & progr. div. - k.u.leuven
c***purpose  the routine calculates an approximation result to a given
c            fourier integral
c            i = integral of f(x)*w(x) over (a,infinity)
c            where w(x) = cos(omega*x) or w(x) = sin(omega*x).
c            hopefully satisfying following claim for accuracy
c            abs(i-result).le.epsabs.
c***description
c
c        computation of fourier integrals
c        standard fortran subroutine
c        real version
c
c
c        parameters
c         on entry
c            f      - real
c                     function subprogram defining the integrand
c                     function f(x). the actual name for f needs to be
c                     declared e x t e r n a l in the driver program.
c
c            a      - real
c                     lower limit of integration
c
c            omega  - real
c                     parameter in the integrand weight function
c
c            integr - integer
c                     indicates which of the weight functions is used
c                     integr = 1      w(x) = cos(omega*x)
c                     integr = 2      w(x) = sin(omega*x)
c                     if integr.ne.1.and.integr.ne.2, the routine
c                     will end with ier = 6.
c
c            epsabs - real
c                     absolute accuracy requested, epsabs.gt.0.
c                     if epsabs.le.0, the routine will end with ier = 6.
c
c         on return
c            result - real
c                     approximation to the integral
c
c            abserr - real
c                     estimate of the modulus of the absolute error,
c                     which should equal or exceed abs(i-result)
c
c            neval  - integer
c                     number of integrand evaluations
c
c            ier    - integer
c                     ier = 0 normal and reliable termination of the
c                             routine. it is assumed that the requested
c                             accuracy has been achieved.
c                     ier.gt.0 abnormal termination of the routine.
c                             the estimates for integral and error are
c                             less reliable. it is assumed that the
c                             requested accuracy has not been achieved.
c            error messages
c                    if omega.ne.0
c                     ier = 1 maximum number of cycles allowed
c                             has been achieved, i.e. of subintervals
c                             (a+(k-1)c,a+kc) where
c                             c = (2*int(abs(omega))+1)*pi/abs(omega),
c                             for k = 1, 2, ..., lst.
c                             one can allow more cycles by increasing
c                             the value of limlst (and taking the
c                             according dimension adjustments into
c                             account). examine the array iwork which
c                             contains the error flags on the cycles, in
c                             order to look for eventual local
c                             integration difficulties.
c                             if the position of a local difficulty
c                             can be determined (e.g. singularity,
c                             discontinuity within the interval) one
c                             will probably gain from splitting up the
c                             interval at this point and calling
c                             appropriate integrators on the subranges.
c                         = 4 the extrapolation table constructed for
c                             convergence accelaration of the series
c                             formed by the integral contributions over
c                             the cycles, does not converge to within
c                             the requested accuracy.
c                             as in the case of ier = 1, it is advised
c                             to examine the array iwork which contains
c                             the error flags on the cycles.
c                         = 6 the input is invalid because
c                             (integr.ne.1 and integr.ne.2) or
c                              epsabs.le.0 or limlst.lt.1 or
c                              leniw.lt.(limlst+2) or maxp1.lt.1 or
c                              lenw.lt.(leniw*2+maxp1*25).
c                              result, abserr, neval, lst are set to
c                              zero.
c                         = 7 bad integrand behaviour occurs within
c                             one or more of the cycles. location and
c                             type of the difficulty involved can be
c                             determined from the first lst elements of
c                             vector iwork.  here lst is the number of
c                             cycles actually needed (see below).
c                             iwork(k) = 1 the maximum number of
c                                          subdivisions (=(leniw-limlst)
c                                          /2) has been achieved on the
c                                          k th cycle.
c                                      = 2 occurrence of roundoff error
c                                          is detected and prevents the
c                                          tolerance imposed on the k th
c                                          cycle, from being achieved
c                                          on this cycle.
c                                      = 3 extremely bad integrand
c                                          behaviour occurs at some
c                                          points of the k th cycle.
c                                      = 4 the integration procedure
c                                          over the k th cycle does
c                                          not converge (to within the
c                                          required accuracy) due to
c                                          roundoff in the extrapolation
c                                          procedure invoked on this
c                                          cycle. it is assumed that the
c                                          result on this interval is
c                                          the best which can be
c                                          obtained.
c                                      = 5 the integral over the k th
c                                          cycle is probably divergent
c                                          or slowly convergent. it must
c                                          be noted that divergence can
c                                          occur with any other value of
c                                          iwork(k).
c                    if omega = 0 and integr = 1,
c                    the integral is calculated by means of dqagie,
c                    and ier = iwork(1) (with meaning as described
c                    for iwork(k),k = 1).
c
c         dimensioning parameters
c            limlst - integer
c                     limlst gives an upper bound on the number of
c                     cycles, limlst.ge.3.
c                     if limlst.lt.3, the routine will end with ier = 6.
c
c            lst    - integer
c                     on return, lst indicates the number of cycles
c                     actually needed for the integration.
c                     if omega = 0, then lst is set to 1.
c
c            leniw  - integer
c                     dimensioning parameter for iwork. on entry,
c                     (leniw-limlst)/2 equals the maximum number of
c                     subintervals allowed in the partition of each
c                     cycle, leniw.ge.(limlst+2).
c                     if leniw.lt.(limlst+2), the routine will end with
c                     ier = 6.
c
c            maxp1  - integer
c                     maxp1 gives an upper bound on the number of
c                     chebyshev moments which can be stored, i.e. for
c                     the intervals of lengths abs(b-a)*2**(-l),
c                     l = 0,1, ..., maxp1-2, maxp1.ge.1.
c                     if maxp1.lt.1, the routine will end with ier = 6.
c            lenw   - integer
c                     dimensioning parameter for work
c                     lenw must be at least leniw*2+maxp1*25.
c                     if lenw.lt.(leniw*2+maxp1*25), the routine will
c                     end with ier = 6.
c
c         work arrays
c            iwork  - integer
c                     vector of dimension at least leniw
c                     on return, iwork(k) for k = 1, 2, ..., lst
c                     contain the error flags on the cycles.
c
c            work   - real
c                     vector of dimension at least
c                     on return,
c                     work(1), ..., work(lst) contain the integral
c                      approximations over the cycles,
c                     work(limlst+1), ..., work(limlst+lst) contain
c                      the error extimates over the cycles.
c                     further elements of work have no specific
c                     meaning for the user.
c
c***references  (none)
c***routines called  qawfe
c***end prologue  qawf
c
       real a,abserr,epsabs,f,omega,result,work
       integer ier,integr,leniw,limit,limlst,lvl,lst,l1,l2,l3,l4,l5,l6,
     *  maxp1,neval
c
       dimension iwork(leniw),work(lenw)
c
       external f
c
c         check validity of limlst, leniw, maxp1 and lenw.
c
c***first executable statement  qawf
      ier = 6
      neval = 0
      last = 0
      result = 0.0e+00
      abserr = 0.0e+00
      if(limlst.lt.3.or.leniw.lt.(limlst+2).or.maxp1.lt.1.or.lenw.lt.
     *   (leniw*2+maxp1*25)) go to 10
c
c         prepare call for qawfe
c
      limit = (leniw-limlst)/2
      l1 = limlst+1
      l2 = limlst+l1
      l3 = limit+l2
      l4 = limit+l3
      l5 = limit+l4
      l6 = limit+l5
      ll2 = limit+l1
      call qawfe(f,a,omega,integr,epsabs,limlst,limit,maxp1,result,
     *  abserr,neval,ier,work(1),work(l1),iwork(1),lst,work(l2),
     *  work(l3),work(l4),work(l5),iwork(l1),iwork(ll2),work(l6))
c
c         call error handler if necessary
c
      lvl = 0
10    if(ier.eq.6) lvl = 1
ccc      if(ier.ne.0) WRITE(*,*)'ABNORMAL ERROR RETURN qawf',ier
      return
      end


c**********************************************************************

      subroutine qawfe(f,a,omega,integr,epsabs,limlst,limit,maxp1,
     *   result,abserr,neval,ier,rslst,erlst,ierlst,lst,alist,blist,
     *   rlist,elist,iord,nnlog,chebmo)
c***begin prologue  qawfe
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a3a1
c***keywords  automatic integrator, special-purpose,
c             fourier integrals,
c             integration between zeros with dqawoe,
c             convergence acceleration with dqelg
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           dedoncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  the routine calculates an approximation result to a
c            given fourier integal
c            i = integral of f(x)*w(x) over (a,infinity)
c             where w(x) = cos(omega*x) or w(x) = sin(omega*x),
c            hopefully satisfying following claim for accuracy
c            abs(i-result).le.epsabs.
c***description
c
c        computation of fourier integrals
c        standard fortran subroutine
c        real version
c
c        parameters
c         on entry
c            f      - real
c                     function subprogram defining the integrand
c                     function f(x). the actual name for f needs to
c                     be declared e x t e r n a l in the driver program.
c
c            a      - real
c                     lower limit of integration
c
c            omega  - real
c                     parameter in the weight function
c
c            integr - integer
c                     indicates which weight function is used
c                     integr = 1      w(x) = cos(omega*x)
c                     integr = 2      w(x) = sin(omega*x)
c                     if integr.ne.1.and.integr.ne.2, the routine will
c                     end with ier = 6.
c
c            epsabs - real
c                     absolute accuracy requested, epsabs.gt.0
c                     if epsabs.le.0, the routine will end with ier = 6.
c
c            limlst - integer
c                     limlst gives an upper bound on the number of
c                     cycles, limlst.ge.1.
c                     if limlst.lt.3, the routine will end with ier = 6.
c
c            limit  - integer
c                     gives an upper bound on the number of subintervals
c                     allowed in the partition of each cycle, limit.ge.1
c                     each cycle, limit.ge.1.
c
c            maxp1  - integer
c                     gives an upper bound on the number of
c                     chebyshev moments which can be stored, i.e.
c                     for the intervals of lengths abs(b-a)*2**(-l),
c                     l=0,1, ..., maxp1-2, maxp1.ge.1
c
c         on return
c            result - real
c                     approximation to the integral x
c
c            abserr - real
c                     estimate of the modulus of the absolute error,
c                     which should equal or exceed abs(i-result)
c
c            neval  - integer
c                     number of integrand evaluations
c
c            ier    - ier = 0 normal and reliable termination of
c                             the routine. it is assumed that the
c                             requested accuracy has been achieved.
c                     ier.gt.0 abnormal termination of the routine. the
c                             estimates for integral and error are less
c                             reliable. it is assumed that the requested
c                             accuracy has not been achieved.
c            error messages
c                    if omega.ne.0
c                     ier = 1 maximum number of  cycles  allowed
c                             has been achieved., i.e. of subintervals
c                             (a+(k-1)c,a+kc) where
c                             c = (2*int(abs(omega))+1)*pi/abs(omega),
c                             for k = 1, 2, ..., lst.
c                             one can allow more cycles by increasing
c                             the value of limlst (and taking the
c                             according dimension adjustments into
c                             account).
c                             examine the array iwork which contains
c                             the error flags on the cycles, in order to
c                             look for eventual local integration
c                             difficulties. if the position of a local
c                             difficulty can be determined (e.g.
c                             singularity, discontinuity within the
c                             interval) one will probably gain from
c                             splitting up the interval at this point
c                             and calling appropriate integrators on
c                             the subranges.
c                         = 4 the extrapolation table constructed for
c                             convergence acceleration of the series
c                             formed by the integral contributions over
c                             the cycles, does not converge to within
c                             the requested accuracy. as in the case of
c                             ier = 1, it is advised to examine the
c                             array iwork which contains the error
c                             flags on the cycles.
c                         = 6 the input is invalid because
c                             (integr.ne.1 and integr.ne.2) or
c                              epsabs.le.0 or limlst.lt.3.
c                              result, abserr, neval, lst are set
c                              to zero.
c                         = 7 bad integrand behaviour occurs within one
c                             or more of the cycles. location and type
c                             of the difficulty involved can be
c                             determined from the vector ierlst. here
c                             lst is the number of cycles actually
c                             needed (see below).
c                             ierlst(k) = 1 the maximum number of
c                                           subdivisions (= limit) has
c                                           been achieved on the k th
c                                           cycle.
c                                       = 2 occurrence of roundoff error
c                                           is detected and prevents the
c                                           tolerance imposed on the
c                                           k th cycle, from being
c                                           achieved.
c                                       = 3 extremely bad integrand
c                                           behaviour occurs at some
c                                           points of the k th cycle.
c                                       = 4 the integration procedure
c                                           over the k th cycle does
c                                           not converge (to within the
c                                           required accuracy) due to
c                                           roundoff in the
c                                           extrapolation procedure
c                                           invoked on this cycle. it
c                                           is assumed that the result
c                                           on this interval is the
c                                           best which can be obtained.
c                                       = 5 the integral over the k th
c                                           cycle is probably divergent
c                                           or slowly convergent. it
c                                           must be noted that
c                                           divergence can occur with
c                                           any other value of
c                                           ierlst(k).
c                    if omega = 0 and integr = 1,
c                    the integral is calculated by means of dqagie
c                    and ier = ierlst(1) (with meaning as described
c                    for ierlst(k), k = 1).
c
c            rslst  - real
c                     vector of dimension at least limlst
c                     rslst(k) contains the integral contribution
c                     over the interval (a+(k-1)c,a+kc) where
c                     c = (2*int(abs(omega))+1)*pi/abs(omega),
c                     k = 1, 2, ..., lst.
c                     note that, if omega = 0, rslst(1) contains
c                     the value of the integral over (a,infinity).
c
c            erlst  - real
c                     vector of dimension at least limlst
c                     erlst(k) contains the error estimate corresponding
c                     with rslst(k).
c
c            ierlst - integer
c                     vector of dimension at least limlst
c                     ierlst(k) contains the error flag corresponding
c                     with rslst(k). for the meaning of the local error
c                     flags see description of output parameter ier.
c
c            lst    - integer
c                     number of subintervals needed for the integration
c                     if omega = 0 then lst is set to 1.
c
c            alist, blist, rlist, elist - real
c                     vector of dimension at least limit,
c
c            iord, nnlog - integer
c                     vector of dimension at least limit, providing
c                     space for the quantities needed in the subdivision
c                     process of each cycle
c
c            chebmo - real
c                     array of dimension at least (maxp1,25), providing
c                     space for the chebyshev moments needed within the
c                     cycles
c
c***references  (none)
c***routines called  qagie,qawoe,qelg,r1mach
c***end prologue  qawfe
c
      real a,abseps,abserr,alist,blist,chebmo,correc,cycle,
     *  c1,c2,dl,dla,drl,elist,ep,eps,epsa,epsabs,erlst,
     *  errsum,fact,omega,p,pi,p1,psum,reseps,result,res3la,rlist,rslst
     *  ,r1mach,uflow
      integer ier,ierlst,integr,iord,ktmin,l,lst,limit,ll,maxp1,
     *    nev,neval,nnlog,nres,numrl2
c
      dimension alist(limit),blist(limit),chebmo(maxp1,25),elist(limit),
     *  erlst(limlst),ierlst(limlst),iord(limit),nnlog(limit),psum(52),
     *  res3la(3),rlist(limit),rslst(limlst)
c
      external f
c
c
c            the dimension of  psum  is determined by the value of
c            limexp in subroutine qelg (psum must be
c            of dimension (limexp+2) at least).
c
c           list of major variables
c           -----------------------
c
c           c1, c2    - end points of subinterval (of length
c                       cycle)
c           cycle     - (2*int(abs(omega))+1)*pi/abs(omega)
c           psum      - vector of dimension at least (limexp+2)
c                       (see routine qelg)
c                       psum contains the part of the epsilon
c                       table which is still needed for further
c                       computations.
c                       each element of psum is a partial sum of
c                       the series which should sum to the value of
c                       the integral.
c           errsum    - sum of error estimates over the
c                       subintervals, calculated cumulatively
c           epsa      - absolute tolerance requested over current
c                       subinterval
c           chebmo    - array containing the modified chebyshev
c                       moments (see also routine qc25f)
c
      data p/0.9e+00/,pi/0.31415926535897932e+01/
c
c           test on validity of parameters
c           ------------------------------
c
c***first executable statement  qawfe
      result = 0.0e+00
      abserr = 0.0e+00
      neval = 0
      lst = 0
      ier = 0
      if((integr.ne.1.and.integr.ne.2).or.epsabs.le.0.0e+00.or.
     *  limlst.lt.3) ier = 6
      if(ier.eq.6) go to 999
      if(omega.ne.0.0e+00) go to 10
c
c           integration by qagie if omega is zero
c           --------------------------------------
c
      if(integr.eq.1) call qagie(f,0.0e+00,1,epsabs,0.0e+00,limit,
     *  result,abserr,neval,ier,alist,blist,rlist,elist,iord,last)
      rslst(1) = result
      erlst(1) = abserr
      ierlst(1) = ier
      lst = 1
      go to 999
c
c           initializations
c           ---------------
c
   10 l = abs(omega)
      dl = 2*l+1
      cycle = dl*pi/abs(omega)
      ier = 0
      ktmin = 0
      neval = 0
      numrl2 = 0
      nres = 0
      c1 = a
      c2 = cycle+a
      p1 = 0.1e+01-p
      eps = epsabs
      uflow = r1mach(1)
      if(epsabs.gt.uflow/p1) eps = epsabs*p1
      ep = eps
      fact = 0.1e+01
      correc = 0.0e+00
      abserr = 0.0e+00
      errsum = 0.0e+00
c
c           main do-loop
c           ------------
c
      do 50 lst = 1,limlst
c
c           integrate over current subinterval.
c
        dla = lst
        epsa = eps*fact
        call qawoe(f,c1,c2,omega,integr,epsa,0.0e+00,limit,lst,maxp1,
     *  rslst(lst),erlst(lst),nev,ierlst(lst),last,alist,blist,rlist,
     *  elist,iord,nnlog,momcom,chebmo)
        neval = neval+nev
        fact = fact*p
        errsum = errsum+erlst(lst)
        drl = 0.5e+02*abs(rslst(lst))
c
c           test on accuracy with partial sum
c
        if(errsum+drl.le.epsabs.and.lst.ge.6) go to 80
        correc = amax1(correc,erlst(lst))
        if(ierlst(lst).ne.0) eps = amax1(ep,correc*p1)
        if(ierlst(lst).ne.0) ier = 7
        if(ier.eq.7.and.(errsum+drl).le.correc*0.1e+02.and.
     *  lst.gt.5) go to 80
        numrl2 = numrl2+1
        if(lst.gt.1) go to 20
        psum(1) = rslst(1)
        go to 40
   20   psum(numrl2) = psum(ll)+rslst(lst)
        if(lst.eq.2) go to 40
c
c           test on maximum number of subintervals
c
        if(lst.eq.limlst) ier = 1
c
c           perform new extrapolation
c
        call qelg(numrl2,psum,reseps,abseps,res3la,nres)
c
c           test whether extrapolated result is influenced by
c           roundoff
c
        ktmin = ktmin+1
        if(ktmin.ge.15.and.abserr.le.0.1e-02*(errsum+drl)) ier = 4
        if(abseps.gt.abserr.and.lst.ne.3) go to 30
        abserr = abseps
        result = reseps
        ktmin = 0
c
c           if ier is not 0, check whether direct result (partial
c           sum) or extrapolated result yields the best integral
c           approximation
c
        if((abserr+0.1e+02*correc).le.epsabs.or.
     *  (abserr.le.epsabs.and.0.1e+02*correc.ge.epsabs)) go to 60
   30   if(ier.ne.0.and.ier.ne.7) go to 60
   40   ll = numrl2
        c1 = c2
        c2 = c2+cycle
   50 continue
c
c         set final result and error estimate
c         -----------------------------------
c
   60 abserr = abserr+0.1e+02*correc
      if(ier.eq.0) go to 999
      if(result.ne.0.0e+00.and.psum(numrl2).ne.0.0e+00) go to 70
      if(abserr.gt.errsum) go to 80
      if(psum(numrl2).eq.0.0e+00) go to 999
   70 if(abserr/abs(result).gt.(errsum+drl)/abs(psum(numrl2)))
     *  go to 80
      if(ier.ge.1.and.ier.ne.7) abserr = abserr+drl
      go to 999
   80 result = psum(numrl2)
      abserr = errsum+drl
  999 return
      end

c**********************************************************************

      subroutine qagie(f,bound,inf,epsabs,epsrel,limit,result,abserr,
     *   neval,ier,alist,blist,rlist,elist,iord,last)
c***begin prologue  qagie
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a3a1,h2a4a1
c***keywords  automatic integrator, infinite intervals,
c             general-purpose, transformation, extrapolation,
c             globally adaptive
c***author  piessens,robert,appl. math & progr. div - k.u.leuven
c           de doncker,elise,appl. math & progr. div - k.u.leuven
c***purpose  the routine calculates an approximation result to a given
c            integral   i = integral of f over (bound,+infinity)
c                    or i = integral of f over (-infinity,bound)
c                    or i = integral of f over (-infinity,+infinity),
c                    hopefully satisfying following claim for accuracy
c                    abs(i-result).le.max(epsabs,epsrel*abs(i))
c***description
c
c integration over infinite intervals
c standard fortran subroutine
c
c            f      - real
c                     function subprogram defining the integrand
c                     function f(x). the actual name for f needs to be
c                     declared e x t e r n a l in the driver program.
c
c            bound  - real
c                     finite bound of integration range
c                     (has no meaning if interval is doubly-infinite)
c
c            inf    - real
c                     indicating the kind of integration range involved
c                     inf = 1 corresponds to  (bound,+infinity),
c                     inf = -1            to  (-infinity,bound),
c                     inf = 2             to (-infinity,+infinity).
c
c            epsabs - real
c                     absolute accuracy requested
c            epsrel - real
c                     relative accuracy requested
c                     if  epsabs.le.0
c                     and epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
c                     the routine will end with ier = 6.
c
c            limit  - integer
c                     gives an upper bound on the number of subintervals
c                     in the partition of (a,b), limit.ge.1
c
c         on return
c            result - real
c                     approximation to the integral
c
c            abserr - real
c                     estimate of the modulus of the absolute error,
c                     which should equal or exceed abs(i-result)
c
c            neval  - integer
c                     number of integrand evaluations
c
c            ier    - integer
c                     ier = 0 normal and reliable termination of the
c                             routine. it is assumed that the requested
c                             accuracy has been achieved.
c                   - ier.gt.0 abnormal termination of the routine. the
c                             estimates for result and error are less
c                             reliable. it is assumed that the requested
c                             accuracy has not been achieved.
c            error messages
c                     ier = 1 maximum number of subdivisions allowed
c                             has been achieved. one can allow more
c                             subdivisions by increasing the value of
c                             limit (and taking the according dimension
c                             adjustments into account). however,if
c                             this yields no improvement it is advised
c                             to analyze the integrand in order to
c                             determine the integration difficulties.
c                             if the position of a local difficulty can
c                             be determined (e.g. singularity,
c                             discontinuity within the interval) one
c                             will probably gain from splitting up the
c                             interval at this point and calling the
c                             integrator on the subranges. if possible,
c                             an appropriate special-purpose integrator
c                             should be used, which is designed for
c                             handling the type of difficulty involved.
c                         = 2 the occurrence of roundoff error is
c                             detected, which prevents the requested
c                             tolerance from being achieved.
c                             the error may be under-estimated.
c                         = 3 extremely bad integrand behaviour occurs
c                             at some points of the integration
c                             interval.
c                         = 4 the algorithm does not converge.
c                             roundoff error is detected in the
c                             extrapolation table.
c                             it is assumed that the requested tolerance
c                             cannot be achieved, and that the returned
c                             result is the best which can be obtained.
c                         = 5 the integral is probably divergent, or
c                             slowly convergent. it must be noted that
c                             divergence can occur with any other value
c                             of ier.
c                         = 6 the input is invalid, because
c                             (epsabs.le.0 and
c                              epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
c                             result, abserr, neval, last, rlist(1),
c                             elist(1) and iord(1) are set to zero.
c                             alist(1) and blist(1) are set to 0
c                             and 1 respectively.
c
c            alist  - real
c                     vector of dimension at least limit, the first
c                      last  elements of which are the left
c                     end points of the subintervals in the partition
c                     of the transformed integration range (0,1).
c
c            blist  - real
c                     vector of dimension at least limit, the first
c                      last  elements of which are the right
c                     end points of the subintervals in the partition
c                     of the transformed integration range (0,1).
c
c            rlist  - real
c                     vector of dimension at least limit, the first
c                      last  elements of which are the integral
c                     approximations on the subintervals
c
c            elist  - real
c                     vector of dimension at least limit,  the first
c                     last elements of which are the moduli of the
c                     absolute error estimates on the subintervals
c
c            iord   - integer
c                     vector of dimension limit, the first k
c                     elements of which are pointers to the
c                     error estimates over the subintervals,
c                     such that elist(iord(1)), ..., elist(iord(k))
c                     form a decreasing sequence, with k = last
c                     if last.le.(limit/2+2), and k = limit+1-last
c                     otherwise
c
c            last   - integer
c                     number of subintervals actually produced
c                     in the subdivision process
c
c***references  (none)
c***routines called  qelg,qk15i,qpsrt,r1mach
c***end prologue  qagie
c
      real abseps,abserr,alist,area,area1,area12,area2,a1,
     *  a2,blist,boun,bound,b1,b2,correc,defabs,defab1,defab2,
     *  dres,r1mach,elist,epmach,epsabs,epsrel,erlarg,erlast,
     *  errbnd,errmax,error1,error2,erro12,errsum,ertest,f,oflow,resabs,
     *  reseps,result,res3la,rlist,rlist2,small,uflow
      integer id,ier,ierro,inf,iord,iroff1,iroff2,iroff3,jupbnd,k,ksgn,
     *  ktmin,last,limit,maxerr,neval,nres,nrmax,numrl2
      logical extrap,noext
c
      dimension alist(limit),blist(limit),elist(limit),iord(limit),
     *  res3la(3),rlist(limit),rlist2(52)
c
      external f
c
c            the dimension of rlist2 is determined by the value of
c            limexp in subroutine qelg.
c
c
c            list of major variables
c            -----------------------
c
c           alist     - list of left end points of all subintervals
c                       considered up to now
c           blist     - list of right end points of all subintervals
c                       considered up to now
c           rlist(i)  - approximation to the integral over
c                       (alist(i),blist(i))
c           rlist2    - array of dimension at least (limexp+2),
c                       containing the part of the epsilon table
c                       wich is still needed for further computations
c           elist(i)  - error estimate applying to rlist(i)
c           maxerr    - pointer to the interval with largest error
c                       estimate
c           errmax    - elist(maxerr)
c           erlast    - error on the interval currently subdivided
c                       (before that subdivision has taken place)
c           area      - sum of the integrals over the subintervals
c           errsum    - sum of the errors over the subintervals
c           errbnd    - requested accuracy max(epsabs,epsrel*
c                       abs(result))
c           *****1    - variable for the left subinterval
c           *****2    - variable for the right subinterval
c           last      - index for subdivision
c           nres      - number of calls to the extrapolation routine
c           numrl2    - number of elements currently in rlist2. if an
c                       appropriate approximation to the compounded
c                       integral has been obtained, it is put in
c                       rlist2(numrl2) after numrl2 has been increased
c                       by one.
c           small     - length of the smallest interval considered up
c                       to now, multiplied by 1.5
c           erlarg    - sum of the errors over the intervals larger
c                       than the smallest interval considered up to now
c           extrap    - logical variable denoting that the routine
c                       is attempting to perform extrapolation. i.e.
c                       before subdividing the smallest interval we
c                       try to decrease the value of erlarg.
c           noext     - logical variable denoting that extrapolation
c                       is no longer allowed (true-value)
c
c            machine dependent constants
c            ---------------------------
c
c           epmach is the largest relative spacing.
c           uflow is the smallest positive magnitude.
c           oflow is the largest positive magnitude.
c
       epmach = r1mach(4)
c
c           test on validity of parameters
c           -----------------------------
c
c***first executable statement  qagie
      ier = 0
      neval = 0
      last = 0
      result = 0.0e+00
      abserr = 0.0e+00
      alist(1) = 0.0e+00
      blist(1) = 0.1e+01
      rlist(1) = 0.0e+00
      elist(1) = 0.0e+00
      iord(1) = 0
      if(epsabs.le.0.0e+00.and.epsrel.lt.amax1(0.5e+02*epmach,0.5e-14))
     *  ier = 6
      if(ier.eq.6) go to 999
c
c
c           first approximation to the integral
c           -----------------------------------
c
c           determine the interval to be mapped onto (0,1).
c           if inf = 2 the integral is computed as i = i1+i2, where
c           i1 = integral of f over (-infinity,0),
c           i2 = integral of f over (0,+infinity).
c
      boun = bound
      if(inf.eq.2) boun = 0.0e+00
      call qk15i(f,boun,inf,0.0e+00,0.1e+01,result,abserr,
     *  defabs,resabs)
c
c           test on accuracy
c
      last = 1
      rlist(1) = result
      elist(1) = abserr
      iord(1) = 1
      dres = abs(result)
      errbnd = amax1(epsabs,epsrel*dres)
      if(abserr.le.1.0e+02*epmach*defabs.and.abserr.gt.
     *  errbnd) ier = 2
      if(limit.eq.1) ier = 1
      if(ier.ne.0.or.(abserr.le.errbnd.and.abserr.ne.resabs).or.
     *  abserr.eq.0.0e+00) go to 130
c
c           initialization
c           --------------
c
      uflow = r1mach(1)
      oflow = r1mach(2)
      rlist2(1) = result
      errmax = abserr
      maxerr = 1
      area = result
      errsum = abserr
      abserr = oflow
      nrmax = 1
      nres = 0
      ktmin = 0
      numrl2 = 2
      extrap = .false.
      noext = .false.
      ierro = 0
      iroff1 = 0
      iroff2 = 0
      iroff3 = 0
      ksgn = -1
      if(dres.ge.(0.1e+01-0.5e+02*epmach)*defabs) ksgn = 1
c
c           main do-loop
c           ------------
c
      do 90 last = 2,limit
c
c           bisect the subinterval with nrmax-th largest
c           error estimate.
c
        a1 = alist(maxerr)
        b1 = 0.5e+00*(alist(maxerr)+blist(maxerr))
        a2 = b1
        b2 = blist(maxerr)
        erlast = errmax
        call qk15i(f,boun,inf,a1,b1,area1,error1,resabs,defab1)
        call qk15i(f,boun,inf,a2,b2,area2,error2,resabs,defab2)
c
c           improve previous approximations to integral
c           and error and test for accuracy.
c
        area12 = area1+area2
        erro12 = error1+error2
        errsum = errsum+erro12-errmax
        area = area+area12-rlist(maxerr)
        if(defab1.eq.error1.or.defab2.eq.error2)go to 15
        if(abs(rlist(maxerr)-area12).gt.0.1e-04*abs(area12)
     *  .or.erro12.lt.0.99e+00*errmax) go to 10
        if(extrap) iroff2 = iroff2+1
        if(.not.extrap) iroff1 = iroff1+1
   10   if(last.gt.10.and.erro12.gt.errmax) iroff3 = iroff3+1
   15   rlist(maxerr) = area1
        rlist(last) = area2
        errbnd = amax1(epsabs,epsrel*abs(area))
c
c           test for roundoff error and eventually
c           set error flag.
c
        if(iroff1+iroff2.ge.10.or.iroff3.ge.20) ier = 2
        if(iroff2.ge.5) ierro = 3
c
c           set error flag in the case that the number of
c           subintervals equals limit.
c
        if(last.eq.limit) ier = 1
c
c           set error flag in the case of bad integrand behaviour
c           at some points of the integration range.
c
        if(amax1(abs(a1),abs(b2)).le.(0.1e+01+0.1e+03*epmach)*
     *  (abs(a2)+0.1e+04*uflow)) ier = 4
c
c           append the newly-created intervals to the list.
c
        if(error2.gt.error1) go to 20
        alist(last) = a2
        blist(maxerr) = b1
        blist(last) = b2
        elist(maxerr) = error1
        elist(last) = error2
        go to 30
   20   alist(maxerr) = a2
        alist(last) = a1
        blist(last) = b1
        rlist(maxerr) = area2
        rlist(last) = area1
        elist(maxerr) = error2
        elist(last) = error1
c
c           call subroutine qpsrt to maintain the descending ordering
c           in the list of error estimates and select the
c           subinterval with nrmax-th largest error estimate (to be
c           bisected next).
c
   30   call qpsrt(limit,last,maxerr,errmax,elist,iord,nrmax)
        if(errsum.le.errbnd) go to 115
        if(ier.ne.0) go to 100
        if(last.eq.2) go to 80
        if(noext) go to 90
        erlarg = erlarg-erlast
        if(abs(b1-a1).gt.small) erlarg = erlarg+erro12
        if(extrap) go to 40
c
c           test whether the interval to be bisected next is the
c           smallest interval.
c
        if(abs(blist(maxerr)-alist(maxerr)).gt.small) go to 90
        extrap = .true.
        nrmax = 2
   40   if(ierro.eq.3.or.erlarg.le.ertest) go to 60
c
c           the smallest interval has the largest error.
c           before bisecting decrease the sum of the errors
c           over the larger intervals (erlarg) and perform
c           extrapolation.
c
        id = nrmax
        jupbnd = last
        if(last.gt.(2+limit/2)) jupbnd = limit+3-last
        do 50 k = id,jupbnd
          maxerr = iord(nrmax)
          errmax = elist(maxerr)
          if(abs(blist(maxerr)-alist(maxerr)).gt.small) go to 90
          nrmax = nrmax+1
   50   continue
c
c           perform extrapolation.
c
   60   numrl2 = numrl2+1
        rlist2(numrl2) = area
        call qelg(numrl2,rlist2,reseps,abseps,res3la,nres)
        ktmin = ktmin+1
        if(ktmin.gt.5.and.abserr.lt.0.1e-02*errsum) ier = 5
        if(abseps.ge.abserr) go to 70
        ktmin = 0
        abserr = abseps
        result = reseps
        correc = erlarg
        ertest = amax1(epsabs,epsrel*abs(reseps))
        if(abserr.le.ertest) go to 100
c
c            prepare bisection of the smallest interval.
c
   70   if(numrl2.eq.1) noext = .true.
        if(ier.eq.5) go to 100
        maxerr = iord(1)
        errmax = elist(maxerr)
        nrmax = 1
        extrap = .false.
        small = small*0.5e+00
        erlarg = errsum
        go to 90
   80   small = 0.375e+00
        erlarg = errsum
        ertest = errbnd
        rlist2(2) = area
   90 continue
c
c           set final result and error estimate.
c           ------------------------------------
c
  100 if(abserr.eq.oflow) go to 115
      if((ier+ierro).eq.0) go to 110
      if(ierro.eq.3) abserr = abserr+correc
      if(ier.eq.0) ier = 3
      if(result.ne.0.0e+00.and.area.ne.0.0e+00)go to 105
      if(abserr.gt.errsum)go to 115
      if(area.eq.0.0e+00) go to 130
      go to 110
  105 if(abserr/abs(result).gt.errsum/abs(area))go to 115
c
c           test on divergence
c
  110 if(ksgn.eq.(-1).and.amax1(abs(result),abs(area)).le.
     * defabs*0.1e-01) go to 130
      if(0.1e-01.gt.(result/area).or.(result/area).gt.0.1e+03.
     *or.errsum.gt.abs(area)) ier = 6
      go to 130
c
c           compute global integral sum.
c
  115 result = 0.0e+00
      do 120 k = 1,last
        result = result+rlist(k)
  120 continue
      abserr = errsum
  130 neval = 30*last-15
      if(inf.eq.2) neval = 2*neval
      if(ier.gt.2) ier=ier-1
  999 return
      end

c******************************************************************

      subroutine qng(f,a,b,epsabs,epsrel,result,abserr,neval,ier)
c***begin prologue  qng
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a1a1
c***keywords  automatic integrator, smooth integrand,
c             non-adaptive, gauss-kronrod(patterson)
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl math & progr. div. - k.u.leuven
c           kahaner,david,nbs - modified (2/82)
c***purpose  the routine calculates an approximation result to a
c            given definite integral i = integral of f over (a,b),
c            hopefully satisfying following claim for accuracy
c            abs(i-result).le.max(epsabs,epsrel*abs(i)).
c***description
c
c non-adaptive integration
c standard fortran subroutine
c real version
c
c           f      - real version
c                    function subprogram defining the integrand function
c                    f(x). the actual name for f needs to be declared
c                    e x t e r n a l in the driver program.
c
c           a      - real version
c                    lower limit of integration
c
c           b      - real version
c                    upper limit of integration
c
c           epsabs - real
c                    absolute accuracy requested
c           epsrel - real
c                    relative accuracy requested
c                    if  epsabs.le.0
c                    and epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
c                    the routine will end with ier = 6.
c
c         on return
c           result - real
c                    approximation to the integral i
c                    result is obtained by applying the 21-point
c                    gauss-kronrod rule (res21) obtained by optimal
c                    addition of abscissae to the 10-point gauss rule
c                    (res10), or by applying the 43-point rule (res43)
c                    obtained by optimal addition of abscissae to the
c                    21-point gauss-kronrod rule, or by applying the
c                    87-point rule (res87) obtained by optimal addition
c                    of abscissae to the 43-point rule.
c
c           abserr - real
c                    estimate of the modulus of the absolute error,
c                    which should equal or exceed abs(i-result)
c
c           neval  - integer
c                    number of integrand evaluations
c
c           ier    - ier = 0 normal and reliable termination of the
c                            routine. it is assumed that the requested
c                            accuracy has been achieved.
c                    ier.gt.0 abnormal termination of the routine. it is
c                            assumed that the requested accuracy has
c                            not been achieved.
c           error messages
c                    ier = 1 the maximum number of steps has been
c                            executed. the integral is probably too
c                            difficult to be calculated by dqng.
c                        = 6 the input is invalid, because
c                            epsabs.le.0 and
c                            epsrel.lt.max(50*rel.mach.acc.,0.5d-28).
c                            result, abserr and neval are set to zero.
c
c***references  (none)
c***routines called  r1mach,xerror
c***end prologue  qng
c
      real a,absc,abserr,b,centr,dhlgth,epmach,epsabs,epsrel,f,fcentr,
     *  fval,fval1,fval2,fv1,fv2,fv3,fv4,hlgth,result,res10,res21,res43,
     *  res87,resabs,resasc,reskh,r1mach,savfun,uflow,w10,w21a,w43a,
     *  w43b,w87a,w87b,x1,x2,x3,x4
      integer ier,ipx,k,l,neval
      external f
c
      dimension fv1(5),fv2(5),fv3(5),fv4(5),x1(5),x2(5),x3(11),x4(22),
     *  w10(5),w21a(5),w21b(6),w43a(10),w43b(12),w87a(21),w87b(23),
     *  savfun(21)
c
c           the following data statements contain the
c           abscissae and weights of the integration rules used.
c
c           x1      abscissae common to the 10-, 21-, 43-
c                   and 87-point rule
c           x2      abscissae common to the 21-, 43- and
c                   87-point rule
c           x3      abscissae common to the 43- and 87-point
c                   rule
c           x4      abscissae of the 87-point rule
c           w10     weights of the 10-point formula
c           w21a    weights of the 21-point formula for
c                   abscissae x1
c           w21b    weights of the 21-point formula for
c                   abscissae x2
c           w43a    weights of the 43-point formula for
c                   abscissae x1, x3
c           w43b    weights of the 43-point formula for
c                   abscissae x3
c           w87a    weights of the 87-point formula for
c                   abscissae x1, x2, x3
c           w87b    weights of the 87-point formula for
c                   abscissae x4
c
      data x1(1),x1(2),x1(3),x1(4),x1(5)/
     *     0.9739065285171717e+00,     0.8650633666889845e+00,
     *     0.6794095682990244e+00,     0.4333953941292472e+00,
     *     0.1488743389816312e+00/
      data x2(1),x2(2),x2(3),x2(4),x2(5)/
     *     0.9956571630258081e+00,     0.9301574913557082e+00,
     *     0.7808177265864169e+00,     0.5627571346686047e+00,
     *     0.2943928627014602e+00/
      data x3(1),x3(2),x3(3),x3(4),x3(5),x3(6),x3(7),x3(8),
     *  x3(9),x3(10),x3(11)/
     *     0.9993333609019321e+00,     0.9874334029080889e+00,
     *     0.9548079348142663e+00,     0.9001486957483283e+00,
     *     0.8251983149831142e+00,     0.7321483889893050e+00,
     *     0.6228479705377252e+00,     0.4994795740710565e+00,
     *     0.3649016613465808e+00,     0.2222549197766013e+00,
     *     0.7465061746138332e-01/
      data x4(1),x4(2),x4(3),x4(4),x4(5),x4(6),x4(7),x4(8),x4(9),
     *  x4(10),x4(11),x4(12),x4(13),x4(14),x4(15),x4(16),x4(17),x4(18),
     *  x4(19),x4(20),x4(21),x4(22)/   0.9999029772627292e+00,
     *     0.9979898959866787e+00,     0.9921754978606872e+00,
     *     0.9813581635727128e+00,     0.9650576238583846e+00,
     *     0.9431676131336706e+00,     0.9158064146855072e+00,
     *     0.8832216577713165e+00,     0.8457107484624157e+00,
     *     0.8035576580352310e+00,     0.7570057306854956e+00,
     *     0.7062732097873218e+00,     0.6515894665011779e+00,
     *     0.5932233740579611e+00,     0.5314936059708319e+00,
     *     0.4667636230420228e+00,     0.3994248478592188e+00,
     *     0.3298748771061883e+00,     0.2585035592021616e+00,
     *     0.1856953965683467e+00,     0.1118422131799075e+00,
     *     0.3735212339461987e-01/
      data w10(1),w10(2),w10(3),w10(4),w10(5)/
     *     0.6667134430868814e-01,     0.1494513491505806e+00,
     *     0.2190863625159820e+00,     0.2692667193099964e+00,
     *     0.2955242247147529e+00/
      data w21a(1),w21a(2),w21a(3),w21a(4),w21a(5)/
     *     0.3255816230796473e-01,     0.7503967481091995e-01,
     *     0.1093871588022976e+00,     0.1347092173114733e+00,
     *     0.1477391049013385e+00/
      data w21b(1),w21b(2),w21b(3),w21b(4),w21b(5),w21b(6)/
     *     0.1169463886737187e-01,     0.5475589657435200e-01,
     *     0.9312545458369761e-01,     0.1234919762620659e+00,
     *     0.1427759385770601e+00,     0.1494455540029169e+00/
      data w43a(1),w43a(2),w43a(3),w43a(4),w43a(5),w43a(6),w43a(7),
     *  w43a(8),w43a(9),w43a(10)/      0.1629673428966656e-01,
     *     0.3752287612086950e-01,     0.5469490205825544e-01,
     *     0.6735541460947809e-01,     0.7387019963239395e-01,
     *     0.5768556059769796e-02,     0.2737189059324884e-01,
     *     0.4656082691042883e-01,     0.6174499520144256e-01,
     *     0.7138726726869340e-01/
      data w43b(1),w43b(2),w43b(3),w43b(4),w43b(5),w43b(6),
     *  w43b(7),w43b(8),w43b(9),w43b(10),w43b(11),w43b(12)/
     *     0.1844477640212414e-02,     0.1079868958589165e-01,
     *     0.2189536386779543e-01,     0.3259746397534569e-01,
     *     0.4216313793519181e-01,     0.5074193960018458e-01,
     *     0.5837939554261925e-01,     0.6474640495144589e-01,
     *     0.6956619791235648e-01,     0.7282444147183321e-01,
     *     0.7450775101417512e-01,     0.7472214751740301e-01/
      data w87a(1),w87a(2),w87a(3),w87a(4),w87a(5),w87a(6),
     *  w87a(7),w87a(8),w87a(9),w87a(10),w87a(11),w87a(12),
     *  w87a(13),w87a(14),w87a(15),w87a(16),w87a(17),w87a(18),
     *  w87a(19),w87a(20),w87a(21)/
     *     0.8148377384149173e-02,     0.1876143820156282e-01,
     *     0.2734745105005229e-01,     0.3367770731163793e-01,
     *     0.3693509982042791e-01,     0.2884872430211531e-02,
     *     0.1368594602271270e-01,     0.2328041350288831e-01,
     *     0.3087249761171336e-01,     0.3569363363941877e-01,
     *     0.9152833452022414e-03,     0.5399280219300471e-02,
     *     0.1094767960111893e-01,     0.1629873169678734e-01,
     *     0.2108156888920384e-01,     0.2537096976925383e-01,
     *     0.2918969775647575e-01,     0.3237320246720279e-01,
     *     0.3478309895036514e-01,     0.3641222073135179e-01,
     *     0.3725387550304771e-01/
      data w87b(1),w87b(2),w87b(3),w87b(4),w87b(5),w87b(6),w87b(7),
     *  w87b(8),w87b(9),w87b(10),w87b(11),w87b(12),w87b(13),w87b(14),
     *  w87b(15),w87b(16),w87b(17),w87b(18),w87b(19),w87b(20),
     *  w87b(21),w87b(22),w87b(23)/    0.2741455637620724e-03,
     *     0.1807124155057943e-02,     0.4096869282759165e-02,
     *     0.6758290051847379e-02,     0.9549957672201647e-02,
     *     0.1232944765224485e-01,     0.1501044734638895e-01,
     *     0.1754896798624319e-01,     0.1993803778644089e-01,
     *     0.2219493596101229e-01,     0.2433914712600081e-01,
     *     0.2637450541483921e-01,     0.2828691078877120e-01,
     *     0.3005258112809270e-01,     0.3164675137143993e-01,
     *     0.3305041341997850e-01,     0.3425509970422606e-01,
     *     0.3526241266015668e-01,     0.3607698962288870e-01,
     *     0.3669860449845609e-01,     0.3712054926983258e-01,
     *     0.3733422875193504e-01,     0.3736107376267902e-01/
c
c           list of major variables
c           -----------------------
c
c           centr  - mid point of the integration interval
c           hlgth  - half-length of the integration interval
c           fcentr - function value at mid point
c           absc   - abscissa
c           fval   - function value
c           savfun - array of function values which
c                    have already been computed
c           res10  - 10-point gauss result
c           res21  - 21-point kronrod result
c           res43  - 43-point result
c           res87  - 87-point result
c           resabs - approximation to the integral of abs(f)
c           resasc - approximation to the integral of abs(f-i/(b-a))
c
c           machine dependent constants
c           ---------------------------
c
c           epmach is the largest relative spacing.
c           uflow is the smallest positive magnitude.
c
c***first executable statement  qng
      epmach = r1mach(4)
      uflow = r1mach(1)
c
c           test on validity of parameters
c           ------------------------------
c
      result = 0.0e+00
      abserr = 0.0e+00
      neval = 0
      ier = 6
      if(epsabs.le.0.0e+00.and.epsrel.lt.amax1(0.5e-14,0.5e+02*epmach))
     *  go to 80
      hlgth = 0.5e+00*(b-a)
      dhlgth = abs(hlgth)
      centr = 0.5e+00*(b+a)
      fcentr = f(centr)
      neval = 21
      ier = 1
c
c          compute the integral using the 10- and 21-point formula.
c
      do 70 l = 1,3
      go to (5,25,45),l
    5 res10 = 0.0e+00
      res21 = w21b(6)*fcentr
      resabs = w21b(6)*abs(fcentr)
      do 10 k=1,5
        absc = hlgth*x1(k)
        fval1 = f(centr+absc)
        fval2 = f(centr-absc)
        fval = fval1+fval2
        res10 = res10+w10(k)*fval
        res21 = res21+w21a(k)*fval
        resabs = resabs+w21a(k)*(abs(fval1)+abs(fval2))
        savfun(k) = fval
        fv1(k) = fval1
        fv2(k) = fval2
   10 continue
      ipx = 5
      do 15 k=1,5
        ipx = ipx+1
        absc = hlgth*x2(k)
        fval1 = f(centr+absc)
        fval2 = f(centr-absc)
        fval = fval1+fval2
        res21 = res21+w21b(k)*fval
        resabs = resabs+w21b(k)*(abs(fval1)+abs(fval2))
        savfun(ipx) = fval
        fv3(k) = fval1
        fv4(k) = fval2
   15 continue
c
c          test for convergence.
c
      result = res21*hlgth
      resabs = resabs*dhlgth
      reskh = 0.5e+00*res21
      resasc = w21b(6)*abs(fcentr-reskh)
      do 20 k = 1,5
        resasc = resasc+w21a(k)*(abs(fv1(k)-reskh)+abs(fv2(k)-reskh))
     *                  +w21b(k)*(abs(fv3(k)-reskh)+abs(fv4(k)-reskh))
   20 continue
      abserr = abs((res21-res10)*hlgth)
      resasc = resasc*dhlgth
      go to 65
c
c          compute the integral using the 43-point formula.
c
   25 res43 = w43b(12)*fcentr
      neval = 43
      do 30 k=1,10
        res43 = res43+savfun(k)*w43a(k)
   30 continue
      do 40 k=1,11
        ipx = ipx+1
        absc = hlgth*x3(k)
        fval = f(absc+centr)+f(centr-absc)
        res43 = res43+fval*w43b(k)
        savfun(ipx) = fval
   40 continue
c
c          test for convergence.
c
      result = res43*hlgth
      abserr = abs((res43-res21)*hlgth)
      go to 65
c
c          compute the integral using the 87-point formula.
c
   45 res87 = w87b(23)*fcentr
      neval = 87
      do 50 k=1,21
        res87 = res87+savfun(k)*w87a(k)
   50 continue
      do 60 k=1,22
        absc = hlgth*x4(k)
        res87 = res87+w87b(k)*(f(absc+centr)+f(centr-absc))
   60 continue
      result = res87*hlgth
      abserr = abs((res87-res43)*hlgth)
   65 if(resasc.ne.0.0e+00.and.abserr.ne.0.0e+00)
     *  abserr = resasc*amin1(0.1e+01,
     *  (0.2e+03*abserr/resasc)**1.5e+00)
      if (resabs.gt.uflow/(0.5e+02*epmach)) abserr = amax1
     *  ((epmach*0.5e+02)*resabs,abserr)
      if (abserr.le.amax1(epsabs,epsrel*abs(result))) ier = 0
c ***jump out of do-loop
      if (ier.eq.0) go to 999
   70 continue
cccc   80 WRITE(*,*)'ABNORMAL ERROR RETURN qng',ier
   80 continue
  999 return
      end

c**********************************************************************

      subroutine qawoe (f,a,b,omega,integr,epsabs,epsrel,limit,icall,
     *  maxp1,result,abserr,neval,ier,last,alist,blist,rlist,elist,iord,
     *   nnlog,momcom,chebmo)
c***begin prologue  qawoe
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a2a1
c***keywords  automatic integrator, special-purpose,
c             integrand with oscillatory cos or sin factor,
c             clenshaw-curtis method, (end point) singularities,
c             extrapolation, globally adaptive
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  the routine calculates an approximation result to a given
c            definite integral
c            i = integral of f(x)*w(x) over (a,b)
c            where w(x) = cos(omega*x) or w(x) = sin(omega*x),
c            hopefully satisfying following claim for accuracy
c            abs(i-result).le.max(epsabs,epsrel*abs(i)).
c***description
c
c        computation of oscillatory integrals
c        standard fortran subroutine
c        real version
c
c        parameters
c         on entry
c            f      - real
c                     function subprogram defining the integrand
c                     function f(x). the actual name for f needs to be
c                     declared e x t e r n a l in the driver program.
c
c            a      - real
c                     lower limit of integration
c
c            b      - real
c                     upper limit of integration
c
c            omega  - real
c                     parameter in the integrand weight function
c
c            integr - integer
c                     indicates which of the weight functions is to be
c                     used
c                     integr = 1      w(x) = cos(omega*x)
c                     integr = 2      w(x) = sin(omega*x)
c                     if integr.ne.1 and integr.ne.2, the routine
c                     will end with ier = 6.
c
c            epsabs - real
c                     absolute accuracy requested
c            epsrel - real
c                     relative accuracy requested
c                     if  epsabs.le.0
c                     and epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
c                     the routine will end with ier = 6.
c
c            limit  - integer
c                     gives an upper bound on the number of subdivisions
c                     in the partition of (a,b), limit.ge.1.
c
c            icall  - integer
c                     if dqawoe is to be used only once, icall must
c                     be set to 1.  assume that during this call, the
c                     chebyshev moments (for clenshaw-curtis integration
c                     of degree 24) have been computed for intervals of
c                     lenghts (abs(b-a))*2**(-l), l=0,1,2,...momcom-1.
c                     if icall.gt.1 this means that dqawoe has been
c                     called twice or more on intervals of the same
c                     length abs(b-a). the chebyshev moments already
c                     computed are then re-used in subsequent calls.
c                     if icall.lt.1, the routine will end with ier = 6.
c
c            maxp1  - integer
c                     gives an upper bound on the number of chebyshev
c                     moments which can be stored, i.e. for the
c                     intervals of lenghts abs(b-a)*2**(-l),
c                     l=0,1, ..., maxp1-2, maxp1.ge.1.
c                     if maxp1.lt.1, the routine will end with ier = 6.
c
c         on return
c            result - real
c                     approximation to the integral
c
c            abserr - real
c                     estimate of the modulus of the absolute error,
c                     which should equal or exceed abs(i-result)
c
c            neval  - integer
c                     number of integrand evaluations
c
c            ier    - integer
c                     ier = 0 normal and reliable termination of the
c                             routine. it is assumed that the
c                             requested accuracy has been achieved.
c                   - ier.gt.0 abnormal termination of the routine.
c                             the estimates for integral and error are
c                             less reliable. it is assumed that the
c                             requested accuracy has not been achieved.
c            error messages
c                     ier = 1 maximum number of subdivisions allowed
c                             has been achieved. one can allow more
c                             subdivisions by increasing the value of
c                             limit (and taking according dimension
c                             adjustments into account). however, if
c                             this yields no improvement it is advised
c                             to analyze the integrand, in order to
c                             determine the integration difficulties.
c                             if the position of a local difficulty can
c                             be determined (e.g. singularity,
c                             discontinuity within the interval) one
c                             will probably gain from splitting up the
c                             interval at this point and calling the
c                             integrator on the subranges. if possible,
c                             an appropriate special-purpose integrator
c                             should be used which is designed for
c                             handling the type of difficulty involved.
c                         = 2 the occurrence of roundoff error is
c                             detected, which prevents the requested
c                             tolerance from being achieved.
c                             the error may be under-estimated.
c                         = 3 extremely bad integrand behaviour occurs
c                             at some points of the integration
c                             interval.
c                         = 4 the algorithm does not converge.
c                             roundoff error is detected in the
c                             extrapolation table.
c                             it is presumed that the requested
c                             tolerance cannot be achieved due to
c                             roundoff in the extrapolation table,
c                             and that the returned result is the
c                             best which can be obtained.
c                         = 5 the integral is probably divergent, or
c                             slowly convergent. it must be noted that
c                             divergence can occur with any other value
c                             of ier.gt.0.
c                         = 6 the input is invalid, because
c                             (epsabs.le.0 and
c                              epsrel.lt.max(50*rel.mach.acc.,0.5d-28))
c                             or (integr.ne.1 and integr.ne.2) or
c                             icall.lt.1 or maxp1.lt.1.
c                             result, abserr, neval, last, rlist(1),
c                             elist(1), iord(1) and nnlog(1) are set
c                             to zero. alist(1) and blist(1) are set
c                             to a and b respectively.
c
c            last  -  integer
c                     on return, last equals the number of
c                     subintervals produces in the subdivision
c                     process, which determines the number of
c                     significant elements actually in the
c                     work arrays.
c            alist  - real
c                     vector of dimension at least limit, the first
c                      last  elements of which are the left
c                     end points of the subintervals in the partition
c                     of the given integration range (a,b)
c
c            blist  - real
c                     vector of dimension at least limit, the first
c                      last  elements of which are the right
c                     end points of the subintervals in the partition
c                     of the given integration range (a,b)
c
c            rlist  - real
c                     vector of dimension at least limit, the first
c                      last  elements of which are the integral
c                     approximations on the subintervals
c
c            elist  - real
c                     vector of dimension at least limit, the first
c                      last  elements of which are the moduli of the
c                     absolute error estimates on the subintervals
c
c            iord   - integer
c                     vector of dimension at least limit, the first k
c                     elements of which are pointers to the error
c                     estimates over the subintervals,
c                     such that elist(iord(1)), ...,
c                     elist(iord(k)) form a decreasing sequence, with
c                     k = last if last.le.(limit/2+2), and
c                     k = limit+1-last otherwise.
c
c            nnlog  - integer
c                     vector of dimension at least limit, containing the
c                     subdivision levels of the subintervals, i.e.
c                     iwork(i) = l means that the subinterval
c                     numbered i is of length abs(b-a)*2**(1-l)
c
c         on entry and return
c            momcom - integer
c                     indicating that the chebyshev moments
c                     have been computed for intervals of lengths
c                     (abs(b-a))*2**(-l), l=0,1,2, ..., momcom-1,
c                     momcom.lt.maxp1
c
c            chebmo - real
c                     array of dimension (maxp1,25) containing the
c                     chebyshev moments
c
c***references  (none)
c***routines called  qc25f,qelg,qpsrt,r1mach
c***end prologue  qawoe
c
      real a,abseps,abserr,alist,area,area1,area12,area2,a1,
     *  a2,b,blist,b1,b2,chebmo,correc,defab1,defab2,defabs,
     *  domega,r1mach,dres,elist,epmach,epsabs,epsrel,erlarg,
     *  erlast,errbnd,errmax,error1,erro12,error2,errsum,ertest,f,oflow,
     *  omega,resabs,reseps,result,res3la,rlist,rlist2,small,uflow,width
      integer icall,id,ier,ierro,integr,iord,iroff1,iroff2,iroff3,
     *  jupbnd,k,ksgn,ktmin,last,limit,maxerr,maxp1,momcom,nev,
     *  neval,nnlog,nres,nrmax,nrmom,numrl2
      logical extrap,noext,extall
c
      dimension alist(limit),blist(limit),rlist(limit),elist(limit),
     *  iord(limit),rlist2(52),res3la(3),chebmo(maxp1,25),nnlog(limit)
c
      external f
c
c            the dimension of rlist2 is determined by  the value of
c            limexp in subroutine qelg (rlist2 should be of
c            dimension (limexp+2) at least).
c
c            list of major variables
c            -----------------------
c
c           alist     - list of left end points of all subintervals
c                       considered up to now
c           blist     - list of right end points of all subintervals
c                       considered up to now
c           rlist(i)  - approximation to the integral over
c                       (alist(i),blist(i))
c           rlist2    - array of dimension at least limexp+2
c                       containing the part of the epsilon table
c                       which is still needed for further computations
c           elist(i)  - error estimate applying to rlist(i)
c           maxerr    - pointer to the interval with largest
c                       error estimate
c           errmax    - elist(maxerr)
c           erlast    - error on the interval currently subdivided
c           area      - sum of the integrals over the subintervals
c           errsum    - sum of the errors over the subintervals
c           errbnd    - requested accuracy max(epsabs,epsrel*
c                       abs(result))
c           *****1    - variable for the left subinterval
c           *****2    - variable for the right subinterval
c           last      - index for subdivision
c           nres      - number of calls to the extrapolation routine
c           numrl2    - number of elements in rlist2. if an appropriate
c                       approximation to the compounded integral has
c                       been obtained it is put in rlist2(numrl2) after
c                       numrl2 has been increased by one
c           small     - length of the smallest interval considered
c                       up to now, multiplied by 1.5
c           erlarg    - sum of the errors over the intervals larger
c                       than the smallest interval considered up to now
c           extrap    - logical variable denoting that the routine is
c                       attempting to perform extrapolation, i.e. before
c                       subdividing the smallest interval we try to
c                       decrease the value of erlarg
c           noext     - logical variable denoting that extrapolation
c                       is no longer allowed (true value)
c
c            machine dependent constants
c            ---------------------------
c
c           epmach is the largest relative spacing.
c           uflow is the smallest positive magnitude.
c           oflow is the largest positive magnitude.
c
c***first executable statement  qawoe
      epmach = r1mach(4)
c
c         test on validity of parameters
c         ------------------------------
c
      ier = 0
      neval = 0
      last = 0
      result = 0.0e+00
      abserr = 0.0e+00
      alist(1) = a
      blist(1) = b
      rlist(1) = 0.0e+00
      elist(1) = 0.0e+00
      iord(1) = 0
      nnlog(1) = 0
      if((integr.ne.1.and.integr.ne.2).or.(epsabs.le.0.0e+00.and.
     *  epsrel.lt.amax1(0.5e+02*epmach,0.5e-14)).or.icall.lt.1.or.
     *  maxp1.lt.1) ier = 6
      if(ier.eq.6) go to 999
c
c           first approximation to the integral
c           -----------------------------------
c
      domega = abs(omega)
      nrmom = 0
      if (icall.gt.1) go to 5
      momcom = 0
    5 call qc25f(f,a,b,domega,integr,nrmom,maxp1,0,result,abserr,
     *  neval,defabs,resabs,momcom,chebmo)
c
c           test on accuracy.
c
      dres = abs(result)
      errbnd = amax1(epsabs,epsrel*dres)
      rlist(1) = result
      elist(1) = abserr
      iord(1) = 1
      if(abserr.le.0.1e+03*epmach*defabs.and.abserr.gt.
     *  errbnd) ier = 2
      if(limit.eq.1) ier = 1
      if(ier.ne.0.or.abserr.le.errbnd) go to 200
c
c           initializations
c           ---------------
c
      uflow = r1mach(1)
      oflow = r1mach(2)
      errmax = abserr
      maxerr = 1
      area = result
      errsum = abserr
      abserr = oflow
      nrmax = 1
      extrap = .false.
      noext = .false.
      ierro = 0
      iroff1 = 0
      iroff2 = 0
      iroff3 = 0
      ktmin = 0
      small = abs(b-a)*0.75e+00
      nres = 0
      numrl2 = 0
      extall = .false.
      if(0.5e+00*abs(b-a)*domega.gt.0.2e+01) go to 10
      numrl2 = 1
      extall = .true.
      rlist2(1) = result
   10 if(0.25e+00*abs(b-a)*domega.le.0.2e+01) extall = .true.
      ksgn = -1
      if(dres.ge.(0.1e+01-0.5e+02*epmach)*defabs) ksgn = 1
c
c           main do-loop
c           ------------
c
      do 140 last = 2,limit
c
c           bisect the subinterval with the nrmax-th largest
c           error estimate.
c
        nrmom = nnlog(maxerr)+1
        a1 = alist(maxerr)
        b1 = 0.5e+00*(alist(maxerr)+blist(maxerr))
        a2 = b1
        b2 = blist(maxerr)
        erlast = errmax
        call qc25f(f,a1,b1,domega,integr,nrmom,maxp1,0,
     *  area1,error1,nev,resabs,defab1,momcom,chebmo)
        neval = neval+nev
        call qc25f(f,a2,b2,domega,integr,nrmom,maxp1,1,
     *  area2,error2,nev,resabs,defab2,momcom,chebmo)
        neval = neval+nev
c
c           improve previous approximations to integral
c           and error and test for accuracy.
c
        area12 = area1+area2
        erro12 = error1+error2
        errsum = errsum+erro12-errmax
        area = area+area12-rlist(maxerr)
        if(defab1.eq.error1.or.defab2.eq.error2) go to 25
        if(abs(rlist(maxerr)-area12).gt.0.1e-04*abs(area12)
     *  .or.erro12.lt.0.99e+00*errmax) go to 20
        if(extrap) iroff2 = iroff2+1
        if(.not.extrap) iroff1 = iroff1+1
   20   if(last.gt.10.and.erro12.gt.errmax) iroff3 = iroff3+1
   25   rlist(maxerr) = area1
        rlist(last) = area2
        nnlog(maxerr) = nrmom
        nnlog(last) = nrmom
        errbnd = amax1(epsabs,epsrel*abs(area))
c
c           test for roundoff error and eventually
c           set error flag
c
        if(iroff1+iroff2.ge.10.or.iroff3.ge.20) ier = 2
        if(iroff2.ge.5) ierro = 3
c
c           set error flag in the case that the number of
c           subintervals equals limit.
c
        if(last.eq.limit) ier = 1
c
c           set error flag in the case of bad integrand behaviour
c           at a point of the integration range.
c
        if(amax1(abs(a1),abs(b2)).le.(0.1e+01+0.1e+03*epmach)
     *  *(abs(a2)+0.1e+04*uflow)) ier = 4
c
c           append the newly-created intervals to the list.
c
        if(error2.gt.error1) go to 30
        alist(last) = a2
        blist(maxerr) = b1
        blist(last) = b2
        elist(maxerr) = error1
        elist(last) = error2
        go to 40
   30   alist(maxerr) = a2
        alist(last) = a1
        blist(last) = b1
        rlist(maxerr) = area2
        rlist(last) = area1
        elist(maxerr) = error2
        elist(last) = error1
c
c           call subroutine qpsrt to maintain the descending ordering
c           in the list of error estimates and select the
c           subinterval with nrmax-th largest error estimate (to be
c           bisected next).
c
   40   call qpsrt(limit,last,maxerr,errmax,elist,iord,nrmax)
c ***jump out of do-loop
      if(errsum.le.errbnd) go to 170
      if(ier.ne.0) go to 150
        if(last.eq.2.and.extall) go to 120
        if(noext) go to 140
        if(.not.extall) go to 50
        erlarg = erlarg-erlast
        if(abs(b1-a1).gt.small) erlarg = erlarg+erro12
        if(extrap) go to 70
c
c           test whether the interval to be bisected next is the
c           smallest interval.
c
   50   width = abs(blist(maxerr)-alist(maxerr))
        if(width.gt.small) go to 140
        if(extall) go to 60
c
c           test whether we can start with the extrapolation
c           procedure (we do this if we integrate over the
c           next interval with use of a gauss-kronrod rule - see
c           subroutine qc25f).
c
        small = small*0.5e+00
        if(0.25e+00*width*domega.gt.0.2e+01) go to 140
        extall = .true.
        go to 130
   60   extrap = .true.
        nrmax = 2
   70   if(ierro.eq.3.or.erlarg.le.ertest) go to 90
c
c           the smallest interval has the largest error.
c           before bisecting decrease the sum of the errors
c           over the larger intervals (erlarg) and perform
c           extrapolation.
c
        jupbnd = last
        if (last.gt.(limit/2+2)) jupbnd = limit+3-last
        id = nrmax
        do 80 k = id,jupbnd
          maxerr = iord(nrmax)
          errmax = elist(maxerr)
          if(abs(blist(maxerr)-alist(maxerr)).gt.small) go to 140
          nrmax = nrmax+1
   80   continue
c
c           perform extrapolation.
c
   90   numrl2 = numrl2+1
        rlist2(numrl2) = area
        if(numrl2.lt.3) go to 110
        call qelg(numrl2,rlist2,reseps,abseps,res3la,nres)
        ktmin = ktmin+1
        if(ktmin.gt.5.and.abserr.lt.0.1e-02*errsum) ier = 5
        if(abseps.ge.abserr) go to 100
        ktmin = 0
        abserr = abseps
        result = reseps
        correc = erlarg
        ertest = amax1(epsabs,epsrel*abs(reseps))
c ***jump out of do-loop
        if(abserr.le.ertest) go to 150
c
c           prepare bisection of the smallest interval.
c
  100   if(numrl2.eq.1) noext = .true.
        if(ier.eq.5) go to 150
  110   maxerr = iord(1)
        errmax = elist(maxerr)
        nrmax = 1
        extrap = .false.
        small = small*0.5e+00
        erlarg = errsum
        go to 140
  120   small = small*0.5e+00
        numrl2 = numrl2+1
        rlist2(numrl2) = area
  130   ertest = errbnd
        erlarg = errsum
  140 continue
c
c           set the final result.
c           ---------------------
c
  150 if(abserr.eq.oflow.or.nres.eq.0) go to 170
      if(ier+ierro.eq.0) go to 165
      if(ierro.eq.3) abserr = abserr+correc
      if(ier.eq.0) ier = 3
      if(result.ne.0.0e+00.and.area.ne.0.0e+00) go to 160
      if(abserr.gt.errsum) go to 170
      if(area.eq.0.0e+00) go to 190
      go to 165
  160 if(abserr/abs(result).gt.errsum/abs(area)) go to 170
c
c           test on divergence.
c
  165 if(ksgn.eq.(-1).and.amax1(abs(result),abs(area)).le.
     * defabs*0.1e-01) go to 190
      if(0.1e-01.gt.(result/area).or.(result/area).gt.0.1e+03
     * .or.errsum.ge.abs(area)) ier = 6
      go to 190
c
c           compute global integral sum.
c
  170 result = 0.0e+00
      do 180 k=1,last
        result = result+rlist(k)
  180 continue
      abserr = errsum
  190 if (ier.gt.2) ier=ier-1
  200 if (integr.eq.2.and.omega.lt.0.0e+00) result=-result
  999 return
      end

c**********************************************************************

      subroutine qcheb(x,fval,cheb12,cheb24)
c***begin prologue  qcheb
c***refer to  qc25c,qc25f,qc25s
c***routines called  (none)
c***revision date  830518   (yymmdd)
c***keywords  chebyshev series expansion, fast fourier transform
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  this routine computes the chebyshev series expansion
c            of degrees 12 and 24 of a function using a
c            fast fourier transform method
c            f(x) = sum(k=1,..,13) (cheb12(k)*t(k-1,x)),
c            f(x) = sum(k=1,..,25) (cheb24(k)*t(k-1,x)),
c            where t(k,x) is the chebyshev polynomial of degree k.
c***description
c
c        chebyshev series expansion
c        standard fortran subroutine
c        real version
c
c        parameters
c          on entry
c           x      - real
c                    vector of dimension 11 containing the
c                    values cos(k*pi/24), k = 1, ..., 11
c
c           fval   - real
c                    vector of dimension 25 containing the
c                    function values at the points
c                    (b+a+(b-a)*cos(k*pi/24))/2, k = 0, ...,24,
c                    where (a,b) is the approximation interval.
c                    fval(1) and fval(25) are divided by two
c                    (these values are destroyed at output).
c
c          on return
c           cheb12 - real
c                    vector of dimension 13 containing the
c                    chebyshev coefficients for degree 12
c
c           cheb24 - real
c                    vector of dimension 25 containing the
c                    chebyshev coefficients for degree 24
c
c***end prologue  qcheb
c
      real alam,alam1,alam2,cheb12,cheb24,
     *  fval,part1,part2,part3,v,x
      integer i,j
c
      dimension cheb12(13),cheb24(25),fval(25),v(12),x(11)
c
c***first executable statement  qcheb
      do 10 i=1,12
        j = 26-i
        v(i) = fval(i)-fval(j)
        fval(i) = fval(i)+fval(j)
   10 continue
      alam1 = v(1)-v(9)
      alam2 = x(6)*(v(3)-v(7)-v(11))
      cheb12(4) = alam1+alam2
      cheb12(10) = alam1-alam2
      alam1 = v(2)-v(8)-v(10)
      alam2 = v(4)-v(6)-v(12)
      alam = x(3)*alam1+x(9)*alam2
      cheb24(4) = cheb12(4)+alam
      cheb24(22) = cheb12(4)-alam
      alam = x(9)*alam1-x(3)*alam2
      cheb24(10) = cheb12(10)+alam
      cheb24(16) = cheb12(10)-alam
      part1 = x(4)*v(5)
      part2 = x(8)*v(9)
      part3 = x(6)*v(7)
      alam1 = v(1)+part1+part2
      alam2 = x(2)*v(3)+part3+x(10)*v(11)
      cheb12(2) = alam1+alam2
      cheb12(12) = alam1-alam2
      alam = x(1)*v(2)+x(3)*v(4)+x(5)*v(6)+x(7)*v(8)
     *  +x(9)*v(10)+x(11)*v(12)
      cheb24(2) = cheb12(2)+alam
      cheb24(24) = cheb12(2)-alam
      alam = x(11)*v(2)-x(9)*v(4)+x(7)*v(6)-x(5)*v(8)
     *  +x(3)*v(10)-x(1)*v(12)
      cheb24(12) = cheb12(12)+alam
      cheb24(14) = cheb12(12)-alam
      alam1 = v(1)-part1+part2
      alam2 = x(10)*v(3)-part3+x(2)*v(11)
      cheb12(6) = alam1+alam2
      cheb12(8) = alam1-alam2
      alam = x(5)*v(2)-x(9)*v(4)-x(1)*v(6)
     *  -x(11)*v(8)+x(3)*v(10)+x(7)*v(12)
      cheb24(6) = cheb12(6)+alam
      cheb24(20) = cheb12(6)-alam
      alam = x(7)*v(2)-x(3)*v(4)-x(11)*v(6)+x(1)*v(8)
     *  -x(9)*v(10)-x(5)*v(12)
      cheb24(8) = cheb12(8)+alam
      cheb24(18) = cheb12(8)-alam
      do 20 i=1,6
        j = 14-i
        v(i) = fval(i)-fval(j)
        fval(i) = fval(i)+fval(j)
   20 continue
      alam1 = v(1)+x(8)*v(5)
      alam2 = x(4)*v(3)
      cheb12(3) = alam1+alam2
      cheb12(11) = alam1-alam2
      cheb12(7) = v(1)-v(5)
      alam = x(2)*v(2)+x(6)*v(4)+x(10)*v(6)
      cheb24(3) = cheb12(3)+alam
      cheb24(23) = cheb12(3)-alam
      alam = x(6)*(v(2)-v(4)-v(6))
      cheb24(7) = cheb12(7)+alam
      cheb24(19) = cheb12(7)-alam
      alam = x(10)*v(2)-x(6)*v(4)+x(2)*v(6)
      cheb24(11) = cheb12(11)+alam
      cheb24(15) = cheb12(11)-alam
      do 30 i=1,3
        j = 8-i
        v(i) = fval(i)-fval(j)
        fval(i) = fval(i)+fval(j)
   30 continue
      cheb12(5) = v(1)+x(8)*v(3)
      cheb12(9) = fval(1)-x(8)*fval(3)
      alam = x(4)*v(2)
      cheb24(5) = cheb12(5)+alam
      cheb24(21) = cheb12(5)-alam
      alam = x(8)*fval(2)-fval(4)
      cheb24(9) = cheb12(9)+alam
      cheb24(17) = cheb12(9)-alam
      cheb12(1) = fval(1)+fval(3)
      alam = fval(2)+fval(4)
      cheb24(1) = cheb12(1)+alam
      cheb24(25) = cheb12(1)-alam
      cheb12(13) = v(1)-v(3)
      cheb24(13) = cheb12(13)
      alam = 0.1e+01/0.6e+01
      do 40 i=2,12
        cheb12(i) = cheb12(i)*alam
   40 continue
      alam = 0.5e+00*alam
      cheb12(1) = cheb12(1)*alam
      cheb12(13) = cheb12(13)*alam
      do 50 i=2,24
        cheb24(i) = cheb24(i)*alam
   50 continue
      cheb24(1) = 0.5e+00*alam*cheb24(1)
      cheb24(25) = 0.5e+00*alam*cheb24(25)
      return
      end

c**********************************************************************

      subroutine qk15i(f,boun,inf,a,b,result,abserr,resabs,resasc)
c***begin prologue  qk15i
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a3a2,h2a4a2
c***keywords  15-point transformed gauss-kronrod rules
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  the original (infinite integration range is mapped
c            onto the interval (0,1) and (a,b) is a part of (0,1).
c            it is the purpose to compute
c            i = integral of transformed integrand over (a,b),
c            j = integral of abs(transformed integrand) over (a,b).
c***description
c
c           integration rule
c           standard fortran subroutine
c           real version
c
c           parameters
c            on entry
c              f      - real
c                       fuction subprogram defining the integrand
c                       function f(x). the actual name for f needs to be
c                       declared e x t e r n a l in the calling program.
c
c              boun   - real
c                       finite bound of original integration
c                       range (set to zero if inf = +2)
c
c              inf    - integer
c                       if inf = -1, the original interval is
c                                   (-infinity,bound),
c                       if inf = +1, the original interval is
c                                   (bound,+infinity),
c                       if inf = +2, the original interval is
c                                   (-infinity,+infinity) and
c                       the integral is computed as the sum of two
c                       integrals, one over (-infinity,0) and one over
c                       (0,+infinity).
c
c              a      - real
c                       lower limit for integration over subrange
c                       of (0,1)
c
c              b      - real
c                       upper limit for integration over subrange
c                       of (0,1)
c
c            on return
c              result - real
c                       approximation to the integral i
c                       result is computed by applying the 15-point
c                       kronrod rule(resk) obtained by optimal addition
c                       of abscissae to the 7-point gauss rule(resg).
c
c              abserr - real
c                       estimate of the modulus of the absolute error,
c                       which should equal or exceed abs(i-result)
c
c              resabs - real
c                       approximation to the integral j
c
c              resasc - real
c                       approximation to the integral of
c                       abs((transformed integrand)-i/(b-a)) over (a,b)
c
c***references  (none)
c***routines called  r1mach
c***end prologue  qk15i
c
      real a,absc,absc1,absc2,abserr,b,boun,centr,
     *  dinf,r1mach,epmach,f,fc,fsum,fval1,fval2,fv1,
     *  fv2,hlgth,resabs,resasc,resg,resk,reskh,result,tabsc1,tabsc2,
     *  uflow,wg,wgk,xgk
      integer inf,j,min0
      external f
c
      dimension fv1(7),fv2(7),xgk(8),wgk(8),wg(8)
c
c           the abscissae and weights are supplied for the interval
c           (-1,1).  because of symmetry only the positive abscissae and
c           their corresponding weights are given.
c
c           xgk    - abscissae of the 15-point kronrod rule
c                    xgk(2), xgk(4), ... abscissae of the 7-point
c                    gauss rule
c                    xgk(1), xgk(3), ...  abscissae which are optimally
c                    added to the 7-point gauss rule
c
c           wgk    - weights of the 15-point kronrod rule
c
c           wg     - weights of the 7-point gauss rule, corresponding
c                    to the abscissae xgk(2), xgk(4), ...
c                    wg(1), wg(3), ... are set to zero.
c
      data xgk(1),xgk(2),xgk(3),xgk(4),xgk(5),xgk(6),xgk(7),
     *  xgk(8)/
     *     0.9914553711208126e+00,     0.9491079123427585e+00,
     *     0.8648644233597691e+00,     0.7415311855993944e+00,
     *     0.5860872354676911e+00,     0.4058451513773972e+00,
     *     0.2077849550078985e+00,     0.0000000000000000e+00/
c
      data wgk(1),wgk(2),wgk(3),wgk(4),wgk(5),wgk(6),wgk(7),
     *  wgk(8)/
     *     0.2293532201052922e-01,     0.6309209262997855e-01,
     *     0.1047900103222502e+00,     0.1406532597155259e+00,
     *     0.1690047266392679e+00,     0.1903505780647854e+00,
     *     0.2044329400752989e+00,     0.2094821410847278e+00/
c
      data wg(1),wg(2),wg(3),wg(4),wg(5),wg(6),wg(7),wg(8)/
     *     0.0000000000000000e+00,     0.1294849661688697e+00,
     *     0.0000000000000000e+00,     0.2797053914892767e+00,
     *     0.0000000000000000e+00,     0.3818300505051189e+00,
     *     0.0000000000000000e+00,     0.4179591836734694e+00/
c
c
c           list of major variables
c           -----------------------
c
c           centr  - mid point of the interval
c           hlgth  - half-length of the interval
c           absc*  - abscissa
c           tabsc* - transformed abscissa
c           fval*  - function value
c           resg   - result of the 7-point gauss formula
c           resk   - result of the 15-point kronrod formula
c           reskh  - approximation to the mean value of the transformed
c                    integrand over (a,b), i.e. to i/(b-a)
c
c           machine dependent constants
c           ---------------------------
c
c           epmach is the largest relative spacing.
c           uflow is the smallest positive magnitude.
c
c***first executable statement  qk15i
      epmach = r1mach(4)
      uflow = r1mach(1)
      dinf = min0(1,inf)
c
      centr = 0.5e+00*(a+b)
      hlgth = 0.5e+00*(b-a)
      tabsc1 = boun+dinf*(0.1e+01-centr)/centr
      fval1 = f(tabsc1)
      if(inf.eq.2) fval1 = fval1+f(-tabsc1)
      fc = (fval1/centr)/centr
c
c           compute the 15-point kronrod approximation to
c           the integral, and estimate the error.
c
      resg = wg(8)*fc
      resk = wgk(8)*fc
      resabs = abs(resk)
      do 10 j=1,7
        absc = hlgth*xgk(j)
        absc1 = centr-absc
        absc2 = centr+absc
        tabsc1 = boun+dinf*(0.1e+01-absc1)/absc1
        tabsc2 = boun+dinf*(0.1e+01-absc2)/absc2
        fval1 = f(tabsc1)
        fval2 = f(tabsc2)
        if(inf.eq.2) fval1 = fval1+f(-tabsc1)
        if(inf.eq.2) fval2 = fval2+f(-tabsc2)
        fval1 = (fval1/absc1)/absc1
        fval2 = (fval2/absc2)/absc2
        fv1(j) = fval1
        fv2(j) = fval2
        fsum = fval1+fval2
        resg = resg+wg(j)*fsum
        resk = resk+wgk(j)*fsum
        resabs = resabs+wgk(j)*(abs(fval1)+abs(fval2))
   10 continue
      reskh = resk*0.5e+00
      resasc = wgk(8)*abs(fc-reskh)
      do 20 j=1,7
        resasc = resasc+wgk(j)*(abs(fv1(j)-reskh)+abs(fv2(j)-reskh))
   20 continue
      result = resk*hlgth
      resasc = resasc*hlgth
      resabs = resabs*hlgth
      abserr = abs((resk-resg)*hlgth)
      if(resasc.ne.0.0e+00.and.abserr.ne.0.e0) abserr = resasc*
     * amin1(0.1e+01,(0.2e+03*abserr/resasc)**1.5e+00)
      if(resabs.gt.uflow/(0.5e+02*epmach)) abserr = amax1
     * ((epmach*0.5e+02)*resabs,abserr)
      return
      end

c**********************************************************************

      subroutine qk15w(f,w,p1,p2,p3,p4,kp,a,b,result,abserr,
     *   resabs,resasc)
c***begin prologue  qk15w
c***date written   810101   (yymmdd)
c***revision date  830518   (mmddyy)
c***category no.  h2a2a2
c***keywords  15-point gauss-kronrod rules
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  to compute i = integral of f*w over (a,b), with error
c                           estimate
c                       j = integral of abs(f*w) over (a,b)
c***description
c
c           integration rules
c           standard fortran subroutine
c           real version
c
c           parameters
c             on entry
c              f      - real
c                       function subprogram defining the integrand
c                       function f(x). the actual name for f needs to be
c                       declared e x t e r n a l in the driver program.
c
c              w      - real
c                       function subprogram defining the integrand
c                       weight function w(x). the actual name for w
c                       needs to be declared e x t e r n a l in the
c                       calling program.
c
c              p1, p2, p3, p4 - real
c                       parameters in the weight function
c
c              kp     - integer
c                       key for indicating the type of weight function
c
c              a      - real
c                       lower limit of integration
c
c              b      - real
c                       upper limit of integration
c
c            on return
c              result - real
c                       approximation to the integral i
c                       result is computed by applying the 15-point
c                       kronrod rule (resk) obtained by optimal addition
c                       of abscissae to the 7-point gauss rule (resg).
c
c              abserr - real
c                       estimate of the modulus of the absolute error,
c                       which should equal or exceed abs(i-result)
c
c              resabs - real
c                       approximation to the integral of abs(f)
c
c              resasc - real
c                       approximation to the integral of abs(f-i/(b-a))
c
c***references  (none)
c***routines called  r1mach
c***end prologue  qk15w
c
      real a,absc,absc1,absc2,abserr,b,centr,dhlgth,
     *  r1mach,epmach,f,fc,fsum,fval1,fval2,fv1,fv2,
     *  hlgth,p1,p2,p3,p4,resabs,resasc,resg,resk,reskh,result,uflow,
     *  w,wg,wgk,xgk
      integer j,jtw,jtwm1,kp
      external f,w
c
      dimension fv1(7),fv2(7),xgk(8),wgk(8),wg(4)
c
c           the abscissae and weights are given for the interval (-1,1).
c           because of symmetry only the positive abscissae and their
c           corresponding weights are given.
c
c           xgk    - abscissae of the 15-point gauss-kronrod rule
c                    xgk(2), xgk(4), ... abscissae of the 7-point
c                    gauss rule
c                    xgk(1), xgk(3), ... abscissae which are optimally
c                    added to the 7-point gauss rule
c
c           wgk    - weights of the 15-point gauss-kronrod rule
c
c           wg     - weights of the 7-point gauss rule
c
      data xgk(1),xgk(2),xgk(3),xgk(4),xgk(5),xgk(6),xgk(7),
     *  xgk(8)/
     *     0.9914553711208126e+00,     0.9491079123427585e+00,
     *     0.8648644233597691e+00,     0.7415311855993944e+00,
     *     0.5860872354676911e+00,     0.4058451513773972e+00,
     *     0.2077849550078985e+00,     0.0000000000000000e+00/
c
      data wgk(1),wgk(2),wgk(3),wgk(4),wgk(5),wgk(6),wgk(7),
     *  wgk(8)/
     *     0.2293532201052922e-01,     0.6309209262997855e-01,
     *     0.1047900103222502e+00,     0.1406532597155259e+00,
     *     0.1690047266392679e+00,     0.1903505780647854e+00,
     *     0.2044329400752989e+00,     0.2094821410847278e+00/
c
      data wg(1),wg(2),wg(3),wg(4)/
     *     0.1294849661688697e+00,    0.2797053914892767e+00,
     *     0.3818300505051889e+00,    0.4179591836734694e+00/
c
c
c           list of major variables
c           -----------------------
c
c           centr  - mid point of the interval
c           hlgth  - half-length of the interval
c           absc*  - abscissa
c           fval*  - function value
c           resg   - result of the 7-point gauss formula
c           resk   - result of the 15-point kronrod formula
c           reskh  - approximation to the mean value of f*w over (a,b),
c                    i.e. to i/(b-a)
c
c           machine dependent constants
c           ---------------------------
c
c           epmach is the largest relative spacing.
c           uflow is the smallest positive magnitude.
c
c***first executable statement  qk15w
      epmach = r1mach(4)
      uflow = r1mach(1)
c
      centr = 0.5e+00*(a+b)
      hlgth = 0.5e+00*(b-a)
      dhlgth = abs(hlgth)
c
c           compute the 15-point kronrod approximation to the
c           integral, and estimate the error.
c
      fc = f(centr)*w(centr,p1,p2,p3,p4,kp)
      resg = wg(4)*fc
      resk = wgk(8)*fc
      resabs = abs(resk)
      do 10 j=1,3
        jtw = j*2
        absc = hlgth*xgk(jtw)
        absc1 = centr-absc
        absc2 = centr+absc
        fval1 = f(absc1)*w(absc1,p1,p2,p3,p4,kp)
        fval2 = f(absc2)*w(absc2,p1,p2,p3,p4,kp)
        fv1(jtw) = fval1
        fv2(jtw) = fval2
        fsum = fval1+fval2
        resg = resg+wg(j)*fsum
        resk = resk+wgk(jtw)*fsum
        resabs = resabs+wgk(jtw)*(abs(fval1)+abs(fval2))
   10 continue
      do 15 j=1,4
        jtwm1 = j*2-1
        absc = hlgth*xgk(jtwm1)
        absc1 = centr-absc
        absc2 = centr+absc
        fval1 = f(absc1)*w(absc1,p1,p2,p3,p4,kp)
        fval2 = f(absc2)*w(absc2,p1,p2,p3,p4,kp)
        fv1(jtwm1) = fval1
        fv2(jtwm1) = fval2
        fsum = fval1+fval2
        resk = resk+wgk(jtwm1)*fsum
        resabs = resabs+wgk(jtwm1)*(abs(fval1)+abs(fval2))
   15 continue
      reskh = resk*0.5e+00
      resasc = wgk(8)*abs(fc-reskh)
      do 20 j=1,7
        resasc = resasc+wgk(j)*(abs(fv1(j)-reskh)+abs(fv2(j)-reskh))
   20 continue
      result = resk*hlgth
      resabs = resabs*dhlgth
      resasc = resasc*dhlgth
      abserr = abs((resk-resg)*hlgth)
      if(resasc.ne.0.0e+00.and.abserr.ne.0.0e+00)
     *  abserr = resasc*amin1(0.1e+01,
     *  (0.2e+03*abserr/resasc)**1.5e+00)
      if(resabs.gt.uflow/(0.5e+02*epmach)) abserr = amax1((epmach*
     *  0.5e+02)*resabs,abserr)
      return
      end

c**********************************************************************

      subroutine qpsrt(limit,last,maxerr,ermax,elist,iord,nrmax)
c***begin prologue  qpsrt
c***refer to  qage,qagie,qagpe,qagse,qawce,qawse,qawoe
c***routines called  (none)
c***keywords  sequential sorting
c***description
c
c 1.        qpsrt
c           ordering routine
c              standard fortran subroutine
c              real version
c
c 2.        purpose
c              this routine maintains the descending ordering
c              in the list of the local error estimates resulting from
c              the interval subdivision process. at each call two error
c              estimates are inserted using the sequential search
c              method, top-down for the largest error estimate
c              and bottom-up for the smallest error estimate.
c
c 3.        calling sequence
c              call qpsrt(limit,last,maxerr,ermax,elist,iord,nrmax)
c
c           parameters (meaning at output)
c              limit  - integer
c                       maximum number of error estimates the list
c                       can contain
c
c              last   - integer
c                       number of error estimates currently
c                       in the list
c
c              maxerr - integer
c                       maxerr points to the nrmax-th largest error
c                       estimate currently in the list
c
c              ermax  - real
c                       nrmax-th largest error estimate
c                       ermax = elist(maxerr)
c
c              elist  - real
c                       vector of dimension last containing
c                       the error estimates
c
c              iord   - integer
c                       vector of dimension last, the first k
c                       elements of which contain pointers
c                       to the error estimates, such that
c                       elist(iord(1)),... , elist(iord(k))
c                       form a decreasing sequence, with
c                       k = last if last.le.(limit/2+2), and
c                       k = limit+1-last otherwise
c
c              nrmax  - integer
c                       maxerr = iord(nrmax)
c
c 4.        no subroutines or functions needed
c***end prologue  qpsrt
c
      real elist,ermax,errmax,errmin
      integer i,ibeg,ido,iord,isucc,j,jbnd,jupbn,k,last,limit,maxerr,
     *  nrmax
      dimension elist(last),iord(last)
c
c           check whether the list contains more than
c           two error estimates.
c
c***first executable statement  qpsrt
      if(last.gt.2) go to 10
      iord(1) = 1
      iord(2) = 2
      go to 90
c
c           this part of the routine is only executed
c           if, due to a difficult integrand, subdivision
c           increased the error estimate. in the normal case
c           the insert procedure should start after the
c           nrmax-th largest error estimate.
c
   10 errmax = elist(maxerr)
      if(nrmax.eq.1) go to 30
      ido = nrmax-1
      do 20 i = 1,ido
        isucc = iord(nrmax-1)
c ***jump out of do-loop
        if(errmax.le.elist(isucc)) go to 30
        iord(nrmax) = isucc
        nrmax = nrmax-1
   20    continue
c
c           compute the number of elements in the list to
c           be maintained in descending order. this number
c           depends on the number of subdivisions still
c           allowed.
c
   30 jupbn = last
      if(last.gt.(limit/2+2)) jupbn = limit+3-last
      errmin = elist(last)
c
c           insert errmax by traversing the list top-down,
c           starting comparison from the element elist(iord(nrmax+1)).
c
      jbnd = jupbn-1
      ibeg = nrmax+1
      if(ibeg.gt.jbnd) go to 50
      do 40 i=ibeg,jbnd
        isucc = iord(i)
c ***jump out of do-loop
        if(errmax.ge.elist(isucc)) go to 60
        iord(i-1) = isucc
   40 continue
   50 iord(jbnd) = maxerr
      iord(jupbn) = last
      go to 90
c
c           insert errmin by traversing the list bottom-up.
c
   60 iord(i-1) = maxerr
      k = jbnd
      do 70 j=i,jbnd
        isucc = iord(k)
c ***jump out of do-loop
        if(errmin.lt.elist(isucc)) go to 80
        iord(k+1) = isucc
        k = k-1
   70 continue
      iord(i) = last
      go to 90
   80 iord(k+1) = last
c
c           set maxerr and ermax.
c
   90 maxerr = iord(nrmax)
      ermax = elist(maxerr)
      return
      end

c**********************************************************************

      subroutine qc25f(f,a,b,omega,integr,nrmom,maxp1,ksave,result,
     *   abserr,neval,resabs,resasc,momcom,chebmo)
c***begin prologue  qc25f
c***date written   810101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a2a2
c***keywords  integration rules for functions with cos or sin
c             factor, clenshaw-curtis, gauss-kronrod
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  to compute the integral i=integral of f(x) over (a,b)
c            where w(x) = cos(omega*x) or (wx)=sin(omega*x)
c            and to compute j=integral of abs(f) over (a,b). for small
c            value of omega or small intervals (a,b) 15-point gauss-
c            kronrod rule used. otherwise generalized clenshaw-curtis us
c***description
c
c        integration rules for functions with cos or sin factor
c        standard fortran subroutine
c        real version
c
c        parameters
c         on entry
c           f      - real
c                    function subprogram defining the integrand
c                    function f(x). the actual name for f needs to
c                    be declared e x t e r n a l in the calling program.
c
c           a      - real
c                    lower limit of integration
c
c           b      - real
c                    upper limit of integration
c
c           omega  - real
c                    parameter in the weight function
c
c           integr - integer
c                    indicates which weight function is to be used
c                       integr = 1   w(x) = cos(omega*x)
c                       integr = 2   w(x) = sin(omega*x)
c
c           nrmom  - integer
c                    the length of interval (a,b) is equal to the length
c                    of the original integration interval divided by
c                    2**nrmom (we suppose that the routine is used in an
c                    adaptive integration process, otherwise set
c                    nrmom = 0). nrmom must be zero at the first call.
c
c           maxp1  - integer
c                    gives an upper bound on the number of chebyshev
c                    moments which can be stored, i.e. for the
c                    intervals of lengths abs(bb-aa)*2**(-l),
c                    l = 0,1,2, ..., maxp1-2.
c
c           ksave  - integer
c                    key which is one when the moments for the
c                    current interval have been computed
c
c         on return
c           result - real
c                    approximation to the integral i
c
c           abserr - real
c                    estimate of the modulus of the absolute
c                    error, which should equal or exceed abs(i-result)
c
c           neval  - integer
c                    number of integrand evaluations
c
c           resabs - real
c                    approximation to the integral j
c
c           resasc - real
c                    approximation to the integral of abs(f-i/(b-a))
c
c         on entry and return
c           momcom - integer
c                    for each interval length we need to compute the
c                    chebyshev moments. momcom counts the number of
c                    intervals for which these moments have already been
c                    computed. if nrmom.lt.momcom or ksave = 1, the
c                    chebyshev moments for the interval (a,b) have
c                    already been computed and stored, otherwise we
c                    compute them and we increase momcom.
c
c           chebmo - real
c                    array of dimension at least (maxp1,25) containing
c                    the modified chebyshev moments for the first momcom
c                    momcom interval lengths
c
c***references  (none)
c***routines called  qcheb,qk15w,qwgtf,r1mach,sgtsl
c***end prologue  qc25f
c
      real a,abserr,ac,an,an2,as,asap,ass,b,centr,chebmo,
     *  cheb12,cheb24,conc,cons,cospar,d,qwgtf,
     *  d1,r1mach,d2,estc,ests,f,fval,hlgth,oflow,omega,parint,par2,
     *  par22,p2,p3,p4,resabs,resasc,resc12,resc24,ress12,ress24,
     *  result,sinpar,v,x
      integer i,iers,integr,isym,j,k,ksave,m,maxp1,momcom,neval,
     *  noequ,noeq1,nrmom
c
      dimension chebmo(maxp1,25),cheb12(13),cheb24(25),d(25),d1(25),
     *  d2(25),fval(25),v(28),x(11)
c
      external f,qwgtf
c
c           the vector x contains the values cos(k*pi/24)
c           k = 1, ...,11, to be used for the chebyshev expansion of f
c
      data x(1),x(2),x(3),x(4),x(5),x(6),x(7),x(8),x(9),
     *  x(10),x(11)/
     *     0.9914448613738104e+00,     0.9659258262890683e+00,
     *     0.9238795325112868e+00,     0.8660254037844386e+00,
     *     0.7933533402912352e+00,     0.7071067811865475e+00,
     *     0.6087614290087206e+00,     0.5000000000000000e+00,
     *     0.3826834323650898e+00,     0.2588190451025208e+00,
     *     0.1305261922200516e+00/
c
c           list of major variables
c           -----------------------
c
c           centr  - mid point of the integration interval
c           hlgth  - half-length of the integration interval
c           fval   - value of the function f at the points
c                    (b-a)*0.5*cos(k*pi/12) + (b+a)*0.5,
c                    k = 0, ..., 24
c           cheb12 - coefficients of the chebyshev series expansion
c                    of degree 12, for the function f, in the
c                    interval (a,b)
c           cheb24 - coefficients of the chebyshev series expansion
c                    of degree 24, for the function f, in the
c                    interval (a,b)
c           resc12 - approximation to the integral of
c                    cos(0.5*(b-a)*omega*x)*f(0.5*(b-a)*x+0.5*(b+a))
c                    over (-1,+1), using the chebyshev series
c                    expansion of degree 12
c           resc24 - approximation to the same integral, using the
c                    chebyshev series expansion of degree 24
c           ress12 - the analogue of resc12 for the sine
c           ress24 - the analogue of resc24 for the sine
c
c
c           machine dependent constant
c           --------------------------
c
c           oflow is the largest positive magnitude.
c
c***first executable statement  qc25f
      oflow = r1mach(2)
c
      centr = 0.5e+00*(b+a)
      hlgth = 0.5e+00*(b-a)
      parint = omega*hlgth
c
c           compute the integral using the 15-point gauss-kronrod
c           formula if the value of the parameter in the integrand
c           is small.
c
      if(abs(parint).gt.0.2e+01) go to 10
      call qk15w(f,qwgtf,omega,p2,p3,p4,integr,a,b,result,
     *  abserr,resabs,resasc)
      neval = 15
      go to 170
c
c           compute the integral using the generalized clenshaw-
c           curtis method.
c
   10 conc = hlgth*cos(centr*omega)
      cons = hlgth*sin(centr*omega)
      resasc = oflow
      neval = 25
c
c           check whether the chebyshev moments for this interval
c           have already been computed.
c
      if(nrmom.lt.momcom.or.ksave.eq.1) go to 120
c
c           compute a new set of chebyshev moments.
c
      m = momcom+1
      par2 = parint*parint
      par22 = par2+0.2e+01
      sinpar = sin(parint)
      cospar = cos(parint)
c
c           compute the chebyshev moments with respect to cosine.
c
      v(1) = 0.2e+01*sinpar/parint
      v(2) = (0.8e+01*cospar+(par2+par2-0.8e+01)*sinpar/
     *  parint)/par2
      v(3) = (0.32e+02*(par2-0.12e+02)*cospar+(0.2e+01*
     *  ((par2-0.80e+02)*par2+0.192e+03)*sinpar)/
     *  parint)/(par2*par2)
      ac = 0.8e+01*cospar
      as = 0.24e+02*parint*sinpar
      if(abs(parint).gt.0.24e+02) go to 30
c
c           compute the chebyshev moments as the
c           solutions of a boundary value problem with 1
c           initial value (v(3)) and 1 end value (computed
c           using an asymptotic formula).
c
      noequ = 25
      noeq1 = noequ-1
      an = 0.6e+01
      do 20 k = 1,noeq1
        an2 = an*an
        d(k) = -0.2e+01*(an2-0.4e+01)*(par22-an2-an2)
        d2(k) = (an-0.1e+01)*(an-0.2e+01)*par2
        d1(k+1) = (an+0.3e+01)*(an+0.4e+01)*par2
        v(k+3) = as-(an2-0.4e+01)*ac
        an = an+0.2e+01
   20 continue
      an2 = an*an
      d(noequ) = -0.2e+01*(an2-0.4e+01)*(par22-an2-an2)
      v(noequ+3) = as-(an2-0.4e+01)*ac
      v(4) = v(4)-0.56e+02*par2*v(3)
      ass = parint*sinpar
      asap = (((((0.210e+03*par2-0.1e+01)*cospar-(0.105e+03*par2
     *  -0.63e+02)*ass)/an2-(0.1e+01-0.15e+02*par2)*cospar
     *  +0.15e+02*ass)/an2-cospar+0.3e+01*ass)/an2-cospar)/an2
      v(noequ+3) = v(noequ+3)-0.2e+01*asap*par2*(an-0.1e+01)*
     *   (an-0.2e+01)
c
c           solve the tridiagonal system by means of gaussian
c           elimination with partial pivoting.
c
      call sgtsl(noequ,d1,d,d2,v(4),iers)
      go to 50
c
c           compute the chebyshev moments by means of forward
c           recursion.
c
   30 an = 0.4e+01
      do 40 i = 4,13
        an2 = an*an
        v(i) = ((an2-0.4e+01)*(0.2e+01*(par22-an2-an2)*v(i-1)-ac)
     *  +as-par2*(an+0.1e+01)*(an+0.2e+01)*v(i-2))/
     *  (par2*(an-0.1e+01)*(an-0.2e+01))
        an = an+0.2e+01
   40 continue
   50 do 60 j = 1,13
        chebmo(m,2*j-1) = v(j)
   60 continue
c
c           compute the chebyshev moments with respect to sine.
c
      v(1) = 0.2e+01*(sinpar-parint*cospar)/par2
      v(2) = (0.18e+02-0.48e+02/par2)*sinpar/par2
     *  +(-0.2e+01+0.48e+02/par2)*cospar/parint
      ac = -0.24e+02*parint*cospar
      as = -0.8e+01*sinpar
      if(abs(parint).gt.0.24e+02) go to 80
c
c           compute the chebyshev moments as the
c           solutions of a boundary value problem with 1
c           initial value (v(2)) and 1 end value (computed
c           using an asymptotic formula).
c
      an = 0.5e+01
      do 70 k = 1,noeq1
        an2 = an*an
        d(k) = -0.2e+01*(an2-0.4e+01)*(par22-an2-an2)
        d2(k) = (an-0.1e+01)*(an-0.2e+01)*par2
        d1(k+1) = (an+0.3e+01)*(an+0.4e+01)*par2
        v(k+2) = ac+(an2-0.4e+01)*as
        an = an+0.2e+01
   70 continue
      an2 = an*an
      d(noequ) = -0.2e+01*(an2-0.4e+01)*(par22-an2-an2)
      v(noequ+2) = ac+(an2-0.4e+01)*as
      v(3) = v(3)-0.42e+02*par2*v(2)
      ass = parint*cospar
      asap = (((((0.105e+03*par2-0.63e+02)*ass+(0.210e+03*par2
     *  -0.1e+01)*sinpar)/an2+(0.15e+02*par2-0.1e+01)*sinpar-
     *  0.15e+02*ass)/an2-0.3e+01*ass-sinpar)/an2-sinpar)/an2
      v(noequ+2) = v(noequ+2)-0.2e+01*asap*par2*(an-0.1e+01)
     *  *(an-0.2e+01)
c
c           solve the tridiagonal system by means of gaussian
c           elimination with partial pivoting.
c
      call sgtsl(noequ,d1,d,d2,v(3),iers)
      go to 100
c
c           compute the chebyshev moments by means of
c           forward recursion.
c
   80 an = 0.3e+01
      do 90 i = 3,12
        an2 = an*an
        v(i) = ((an2-0.4e+01)*(0.2e+01*(par22-an2-an2)*v(i-1)+as)
     *  +ac-par2*(an+0.1e+01)*(an+0.2e+01)*v(i-2))
     *  /(par2*(an-0.1e+01)*(an-0.2e+01))
        an = an+0.2e+01
   90 continue
  100 do 110 j = 1,12
        chebmo(m,2*j) = v(j)
  110 continue
  120 if (nrmom.lt.momcom) m = nrmom+1
       if (momcom.lt.maxp1-1.and.nrmom.ge.momcom) momcom = momcom+1
c
c           compute the coefficients of the chebyshev expansions
c           of degrees 12 and 24 of the function f.
c
      fval(1) = 0.5e+00*f(centr+hlgth)
      fval(13) = f(centr)
      fval(25) = 0.5e+00*f(centr-hlgth)
      do 130 i = 2,12
        isym = 26-i
        fval(i) = f(hlgth*x(i-1)+centr)
        fval(isym) = f(centr-hlgth*x(i-1))
  130 continue
      call qcheb(x,fval,cheb12,cheb24)
c
c           compute the integral and error estimates.
c
      resc12 = cheb12(13)*chebmo(m,13)
      ress12 = 0.0e+00
      k = 11
      do 140 j = 1,6
        resc12 = resc12+cheb12(k)*chebmo(m,k)
        ress12 = ress12+cheb12(k+1)*chebmo(m,k+1)
        k = k-2
  140 continue
      resc24 = cheb24(25)*chebmo(m,25)
      ress24 = 0.0e+00
      resabs = abs(cheb24(25))
      k = 23
      do 150 j = 1,12
        resc24 = resc24+cheb24(k)*chebmo(m,k)
        ress24 = ress24+cheb24(k+1)*chebmo(m,k+1)
        resabs = abs(cheb24(k))+abs(cheb24(k+1))
        k = k-2
  150 continue
      estc = abs(resc24-resc12)
      ests = abs(ress24-ress12)
      resabs = resabs*abs(hlgth)
      if(integr.eq.2) go to 160
      result = conc*resc24-cons*ress24
      abserr = abs(conc*estc)+abs(cons*ests)
      go to 170
  160 result = conc*ress24+cons*resc24
      abserr = abs(conc*ests)+abs(cons*estc)
  170 return
      end

c**********************************************************************

      subroutine qelg(n,epstab,result,abserr,res3la,nres)
c***begin prologue  qelg
c***refer to  qagie,qagoe,qagpe,qagse
c***routines called  r1mach
c***revision date  830518   (yymmdd)
c***keywords  epsilon algorithm, convergence acceleration,
c             extrapolation
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math & progr. div. - k.u.leuven
c***purpose  the routine determines the limit of a given sequence of
c            approximations, by means of the epsilon algorithm of
c            p. wynn. an estimate of the absolute error is also given.
c            the condensed epsilon table is computed. only those
c            elements needed for the computation of the next diagonal
c            are preserved.
c***description
c
c           epsilon algorithm
c           standard fortran subroutine
c           real version
c
c           parameters
c              n      - integer
c                       epstab(n) contains the new element in the
c                       first column of the epsilon table.
c
c              epstab - real
c                       vector of dimension 52 containing the elements
c                       of the two lower diagonals of the triangular
c                       epsilon table. the elements are numbered
c                       starting at the right-hand corner of the
c                       triangle.
c
c              result - real
c                       resulting approximation to the integral
c
c              abserr - real
c                       estimate of the absolute error computed from
c                       result and the 3 previous results
c
c              res3la - real
c                       vector of dimension 3 containing the last 3
c                       results
c
c              nres   - integer
c                       number of calls to the routine
c                       (should be zero at first call)
c
c***end prologue  qelg
c
      real abserr,delta1,delta2,delta3,r1mach,
     *  epmach,epsinf,epstab,error,err1,err2,err3,e0,e1,e1abs,e2,e3,
     *  oflow,res,result,res3la,ss,tol1,tol2,tol3
      integer i,ib,ib2,ie,indx,k1,k2,k3,limexp,n,newelm,nres,num
      dimension epstab(52),res3la(3)
c
c           list of major variables
c           -----------------------
c
c           e0     - the 4 elements on which the
c           e1       computation of a new element in
c           e2       the epsilon table is based
c           e3                 e0
c                        e3    e1    new
c                              e2
c           newelm - number of elements to be computed in the new
c                    diagonal
c           error  - error = abs(e1-e0)+abs(e2-e1)+abs(new-e2)
c           result - the element in the new diagonal with least value
c                    of error
c
c           machine dependent constants
c           ---------------------------
c
c           epmach is the largest relative spacing.
c           oflow is the largest positive magnitude.
c           limexp is the maximum number of elements the epsilon
c           table can contain. if this number is reached, the upper
c           diagonal of the epsilon table is deleted.
c
c***first executable statement  qelg
      epmach = r1mach(4)
      oflow = r1mach(2)
      nres = nres+1
      abserr = oflow
      result = epstab(n)
      if(n.lt.3) go to 100
      limexp = 50
      epstab(n+2) = epstab(n)
      newelm = (n-1)/2
      epstab(n) = oflow
      num = n
      k1 = n
      do 40 i = 1,newelm
        k2 = k1-1
        k3 = k1-2
        res = epstab(k1+2)
        e0 = epstab(k3)
        e1 = epstab(k2)
        e2 = res
        e1abs = abs(e1)
        delta2 = e2-e1
        err2 = abs(delta2)
        tol2 = amax1(abs(e2),e1abs)*epmach
        delta3 = e1-e0
        err3 = abs(delta3)
        tol3 = amax1(e1abs,abs(e0))*epmach
        if(err2.gt.tol2.or.err3.gt.tol3) go to 10
c
c           if e0, e1 and e2 are equal to within machine
c           accuracy, convergence is assumed.
c           result = e2
c           abserr = abs(e1-e0)+abs(e2-e1)
c
        result = res
        abserr = err2+err3
c ***jump out of do-loop
        go to 100
   10   e3 = epstab(k1)
        epstab(k1) = e1
        delta1 = e1-e3
        err1 = abs(delta1)
        tol1 = amax1(e1abs,abs(e3))*epmach
c
c           if two elements are very close to each other, omit
c           a part of the table by adjusting the value of n
c
        if(err1.le.tol1.or.err2.le.tol2.or.err3.le.tol3) go to 20
        ss = 0.1e+01/delta1+0.1e+01/delta2-0.1e+01/delta3
        epsinf = abs(ss*e1)
c
c           test to detect irregular behaviour in the table, and
c           eventually omit a part of the table adjusting the value
c           of n.
c
        if(epsinf.gt.0.1e-03) go to 30
   20   n = i+i-1
c ***jump out of do-loop
        go to 50
c
c           compute a new element and eventually adjust
c           the value of result.
c
   30   res = e1+0.1e+01/ss
        epstab(k1) = res
        k1 = k1-2
        error = err2+abs(res-e2)+err3
        if(error.gt.abserr) go to 40
        abserr = error
        result = res
   40 continue
c
c           shift the table.
c
   50 if(n.eq.limexp) n = 2*(limexp/2)-1
      ib = 1
      if((num/2)*2.eq.num) ib = 2
      ie = newelm+1
      do 60 i=1,ie
        ib2 = ib+2
        epstab(ib) = epstab(ib2)
        ib = ib2
   60 continue
      if(num.eq.n) go to 80
      indx = num-n+1
      do 70 i = 1,n
        epstab(i)= epstab(indx)
        indx = indx+1
   70 continue
   80 if(nres.ge.4) go to 90
      res3la(nres) = result
      abserr = oflow
      go to 100
c
c           compute error estimate
c
   90 abserr = abs(result-res3la(3))+abs(result-res3la(2))
     *  +abs(result-res3la(1))
      res3la(1) = res3la(2)
      res3la(2) = res3la(3)
      res3la(3) = result
  100 abserr = amax1(abserr,0.5e+01*epmach*abs(result))
      return
      end

c**********************************************************************

      real function qwgtf(x,omega,p2,p3,p4,integr)
c***begin prologue  qwgtf
c***refer to   qk15w
c***routines called  (none)
c***revision date 810101   (yymmdd)
c***keywords  cos or sin in weight function
c***author  piessens,robert, appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. * progr. div. - k.u.leuven
c***end prologue  qwgtf
c
      real omega,omx,p2,p3,p4,x
      integer integr
c***first executable statement
      omx = omega*x
      go to(10,20),integr
   10 qwgtf = cos(omx)
      go to 30
   20 qwgtf = sin(omx)
   30 return
      end

c**********************************************************************

      REAL FUNCTION R1MACH(I)
      INTEGER I
C
C  SINGLE-PRECISION MACHINE CONSTANTS
C  R1MACH(1) = B**(EMIN-1), THE SMALLEST POSITIVE MAGNITUDE.
C  R1MACH(2) = B**EMAX*(1 - B**(-T)), THE LARGEST MAGNITUDE.
C  R1MACH(3) = B**(-T), THE SMALLEST RELATIVE SPACING.
C  R1MACH(4) = B**(1-T), THE LARGEST RELATIVE SPACING.
C  R1MACH(5) = LOG10(B)
C
      INTEGER SMALL(2)
      INTEGER LARGE(2)
      INTEGER RIGHT(2)
      INTEGER DIVER(2)
      INTEGER LOG10(2)
C     needs to be (2) for AUTODOUBLE, HARRIS SLASH 6, ...
      INTEGER SC
      SAVE SMALL, LARGE, RIGHT, DIVER, LOG10, SC
      REAL RMACH(5)
      EQUIVALENCE (RMACH(1),SMALL(1))
      EQUIVALENCE (RMACH(2),LARGE(1))
      EQUIVALENCE (RMACH(3),RIGHT(1))
      EQUIVALENCE (RMACH(4),DIVER(1))
      EQUIVALENCE (RMACH(5),LOG10(1))
      INTEGER J, K, L, T3E(3)
      DATA T3E(1) / 9777664 /
      DATA T3E(2) / 5323660 /
      DATA T3E(3) / 46980 /
C  THIS VERSION ADAPTS AUTOMATICALLY TO MOST CURRENT MACHINES,
C  INCLUDING AUTO-DOUBLE COMPILERS.
C  TO COMPILE ON OLDER MACHINES, ADD A C IN COLUMN 1
C  ON THE NEXT LINE
      DATA SC/0/
C  AND REMOVE THE C FROM COLUMN 1 IN ONE OF THE SECTIONS BELOW.
C  CONSTANTS FOR EVEN OLDER MACHINES CAN BE OBTAINED BY
C          mail netlib@research.bell-labs.com
C          send old1mach from blas
C  PLEASE SEND CORRECTIONS TO dmg OR ehg@bell-labs.com.
C
C     MACHINE CONSTANTS FOR THE HONEYWELL DPS 8/70 SERIES.
C      DATA RMACH(1) / O402400000000 /
C      DATA RMACH(2) / O376777777777 /
C      DATA RMACH(3) / O714400000000 /
C      DATA RMACH(4) / O716400000000 /
C      DATA RMACH(5) / O776464202324 /, SC/987/
C
C     MACHINE CONSTANTS FOR PDP-11 FORTRANS SUPPORTING
C     32-BIT INTEGERS (EXPRESSED IN INTEGER AND OCTAL).
C      DATA SMALL(1) /    8388608 /
C      DATA LARGE(1) / 2147483647 /
C      DATA RIGHT(1) /  880803840 /
C      DATA DIVER(1) /  889192448 /
C      DATA LOG10(1) / 1067065499 /, SC/987/
C      DATA RMACH(1) / O00040000000 /
C      DATA RMACH(2) / O17777777777 /
C      DATA RMACH(3) / O06440000000 /
C      DATA RMACH(4) / O06500000000 /
C      DATA RMACH(5) / O07746420233 /, SC/987/
C
C     MACHINE CONSTANTS FOR THE UNIVAC 1100 SERIES.
C      DATA RMACH(1) / O000400000000 /
C      DATA RMACH(2) / O377777777777 /
C      DATA RMACH(3) / O146400000000 /
C      DATA RMACH(4) / O147400000000 /
C      DATA RMACH(5) / O177464202324 /, SC/987/
C
      IF (SC .NE. 987) THEN
*        *** CHECK FOR AUTODOUBLE ***
         SMALL(2) = 0
         RMACH(1) = 1E13
         IF (SMALL(2) .NE. 0) THEN
*           *** AUTODOUBLED ***
            IF (      SMALL(1) .EQ. 1117925532
     *          .AND. SMALL(2) .EQ. -448790528) THEN
*              *** IEEE BIG ENDIAN ***
               SMALL(1) = 1048576
               SMALL(2) = 0
               LARGE(1) = 2146435071
               LARGE(2) = -1
               RIGHT(1) = 1017118720
               RIGHT(2) = 0
               DIVER(1) = 1018167296
               DIVER(2) = 0
               LOG10(1) = 1070810131
               LOG10(2) = 1352628735
            ELSE IF ( SMALL(2) .EQ. 1117925532
     *          .AND. SMALL(1) .EQ. -448790528) THEN
*              *** IEEE LITTLE ENDIAN ***
               SMALL(2) = 1048576
               SMALL(1) = 0
               LARGE(2) = 2146435071
               LARGE(1) = -1
               RIGHT(2) = 1017118720
               RIGHT(1) = 0
               DIVER(2) = 1018167296
               DIVER(1) = 0
               LOG10(2) = 1070810131
               LOG10(1) = 1352628735
            ELSE IF ( SMALL(1) .EQ. -2065213935
     *          .AND. SMALL(2) .EQ. 10752) THEN
*              *** VAX WITH D_FLOATING ***
               SMALL(1) = 128
               SMALL(2) = 0
               LARGE(1) = -32769
               LARGE(2) = -1
               RIGHT(1) = 9344
               RIGHT(2) = 0
               DIVER(1) = 9472
               DIVER(2) = 0
               LOG10(1) = 546979738
               LOG10(2) = -805796613
            ELSE IF ( SMALL(1) .EQ. 1267827943
     *          .AND. SMALL(2) .EQ. 704643072) THEN
*              *** IBM MAINFRAME ***
               SMALL(1) = 1048576
               SMALL(2) = 0
               LARGE(1) = 2147483647
               LARGE(2) = -1
               RIGHT(1) = 856686592
               RIGHT(2) = 0
               DIVER(1) = 873463808
               DIVER(2) = 0
               LOG10(1) = 1091781651
               LOG10(2) = 1352628735
            ELSE
               WRITE(*,9010)
               STOP 777
               END IF
         ELSE
            RMACH(1) = 1234567.
            IF (SMALL(1) .EQ. 1234613304) THEN
*              *** IEEE ***
               SMALL(1) = 8388608
               LARGE(1) = 2139095039
               RIGHT(1) = 864026624
               DIVER(1) = 872415232
               LOG10(1) = 1050288283
            ELSE IF (SMALL(1) .EQ. -1271379306) THEN
*              *** VAX ***
               SMALL(1) = 128
               LARGE(1) = -32769
               RIGHT(1) = 13440
               DIVER(1) = 13568
               LOG10(1) = 547045274
            ELSE IF (SMALL(1) .EQ. 1175639687) THEN
*              *** IBM MAINFRAME ***
               SMALL(1) = 1048576
               LARGE(1) = 2147483647
               RIGHT(1) = 990904320
               DIVER(1) = 1007681536
               LOG10(1) = 1091781651
            ELSE IF (SMALL(1) .EQ. 1251390520) THEN
*              *** CONVEX C-1 ***
               SMALL(1) = 8388608
               LARGE(1) = 2147483647
               RIGHT(1) = 880803840
               DIVER(1) = 889192448
               LOG10(1) = 1067065499
            ELSE
               DO 10 L = 1, 3
                  J = SMALL(1) / 10000000
                  K = SMALL(1) - 10000000*J
                  IF (K .NE. T3E(L)) GO TO 20
                  SMALL(1) = J
 10               CONTINUE
*              *** CRAY T3E ***
               CALL Terminate('Cray not supported')
c               CALL I1MCRA(SMALL, K, 16, 0, 0)
c               CALL I1MCRA(LARGE, K, 32751, 16777215, 16777215)
c               CALL I1MCRA(RIGHT, K, 15520, 0, 0)
c               CALL I1MCRA(DIVER, K, 15536, 0, 0)
c               CALL I1MCRA(LOG10, K, 16339, 4461392, 10451455)
               GO TO 30
20             CALL Terminate('Cray not supported') 
c 20            CALL I1MCRA(J, K, 16405, 9876536, 0)
               IF (SMALL(1) .NE. J) THEN
                  WRITE(*,9020)
                  STOP 777
                  END IF
*              *** CRAY 1, XMP, 2, AND 3 ***
               CALL Terminate('Cray not supported')
c               CALL I1MCRA(SMALL(1), K, 8195, 8388608, 1)
c               CALL I1MCRA(LARGE(1), K, 24574, 16777215, 16777214)
c               CALL I1MCRA(RIGHT(1), K, 16338, 8388608, 0)
c               CALL I1MCRA(DIVER(1), K, 16339, 8388608, 0)
c               CALL I1MCRA(LOG10(1), K, 16383, 10100890, 8715216)
               END IF
            END IF
 30      SC = 987
         END IF
*     SANITY CHECK
      IF (RMACH(4) .GE. 1.0) STOP 776
      IF (I .LT. 1 .OR. I .GT. 5) THEN
         WRITE(*,*) 'R1MACH(I): I =',I,' is out of bounds.'
         STOP
         END IF
      R1MACH = RMACH(I)
      RETURN
 9010 FORMAT(/' Adjust autodoubled R1MACH by getting data'/
     *' appropriate for your machine from D1MACH.')
 9020 FORMAT(/' Adjust R1MACH by uncommenting data statements'/
     *' appropriate for your machine.')
* /* C source for R1MACH -- remove the * in column 1 */
*#include <stdio.h>
*#include <float.h>
*#include <math.h>
*float r1mach_(long *i)
*{
*	switch(*i){
*	  case 1: return FLT_MIN;
*	  case 2: return FLT_MAX;
*	  case 3: return FLT_EPSILON/FLT_RADIX;
*	  case 4: return FLT_EPSILON;
*	  case 5: return log10((double)FLT_RADIX);
*	  }
*	fprintf(stderr, "invalid argument: r1mach(%ld)\n", *i);
*	exit(1); return 0; /* else complaint of missing return value */
*}
      END

cc      SUBROUTINE I1MCRA(A, A1, B, C, D)
**** SPECIAL COMPUTATION FOR CRAY MACHINES ****
cc      INTEGER A, A1, B, C, D
cc      A1 = 16777216*B + C
cc      A = 16777216*A1 + D
cc      END

c**********************************************************************

      subroutine sgtsl(n,c,d,e,b,info)
      integer n,info
      real c(1),d(1),e(1),b(1)
c
c     sgtsl given a general tridiagonal matrix and a right hand
c     side will find the solution.
c
c     on entry
c
c        n       integer
c                is the order of the tridiagonal matrix.
c
c        c       real(n)
c                is the subdiagonal of the tridiagonal matrix.
c                c(2) through c(n) should contain the subdiagonal.
c                on output c is destroyed.
c
c        d       real(n)
c                is the diagonal of the tridiagonal matrix.
c                on output d is destroyed.
c
c        e       real(n)
c                is the superdiagonal of the tridiagonal matrix.
c                e(1) through e(n-1) should contain the superdiagonal.
c                on output e is destroyed.
c
c        b       real(n)
c                is the right hand side vector.
c
c     on return
c
c        b       is the solution vector.
c
c        info    integer
c                = 0 normal value.
c                = k if the k-th element of the diagonal becomes
c                    exactly zero.  the subroutine returns when
c                    this is detected.
c
c     linpack. this version dated 08/14/78 .
c     jack dongarra, argonne national laboratory.
c
c     no externals
c     fortran abs
c
c     internal variables
c
      integer k,kb,kp1,nm1,nm2
      real t
c     begin block permitting ...exits to 100
c
         info = 0
         c(1) = d(1)
         nm1 = n - 1
         if (nm1 .lt. 1) go to 40
            d(1) = e(1)
            e(1) = 0.0e0
            e(n) = 0.0e0
c
            do 30 k = 1, nm1
               kp1 = k + 1
c
c              find the largest of the two rows
c
               if (abs(c(kp1)) .lt. abs(c(k))) go to 10
c
c                 interchange row
c
                  t = c(kp1)
                  c(kp1) = c(k)
                  c(k) = t
                  t = d(kp1)
                  d(kp1) = d(k)
                  d(k) = t
                  t = e(kp1)
                  e(kp1) = e(k)
                  e(k) = t
                  t = b(kp1)
                  b(kp1) = b(k)
                  b(k) = t
   10          continue
c
c              zero elements
c
               if (c(k) .ne. 0.0e0) go to 20
                  info = k
c     ............exit
                  go to 100
   20          continue
               t = -c(kp1)/c(k)
               c(kp1) = d(kp1) + t*d(k)
               d(kp1) = e(kp1) + t*e(k)
               e(kp1) = 0.0e0
               b(kp1) = b(kp1) + t*b(k)
   30       continue
   40    continue
         if (c(n) .ne. 0.0e0) go to 50
            info = n
         go to 90
   50    continue
c
c           back solve
c
            nm2 = n - 2
            b(n) = b(n)/c(n)
            if (n .eq. 1) go to 80
               b(nm1) = (b(nm1) - d(nm1)*b(n))/c(nm1)
               if (nm2 .lt. 1) go to 70
               do 60 kb = 1, nm2
                  k = nm2 - kb + 1
                  b(k) = (b(k) - d(k)*b(k+1) - e(k)*b(k+2))/c(k)
   60          continue
   70          continue
   80       continue
   90    continue
  100 continue
c
      return
      end

c**********************************************************************

      subroutine qk15(f,a,b,result,abserr,resabs,resasc)
c***begin prologue  qk15
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a1a2
c***keywords  15-point gauss-kronrod rules
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div - k.u.leuven
c***purpose  to compute i = integral of f over (a,b), with error
c                           estimate
c                       j = integral of abs(f) over (a,b)
c***description
c
c           integration rules
c           standard fortran subroutine
c           real version
c
c           parameters
c            on entry
c              f      - real
c                       function subprogram defining the integrand
c                       function f(x). the actual name for f needs to be
c                       declared e x t e r n a l in the calling program.
c
c              a      - real
c                       lower limit of integration
c
c              b      - real
c                       upper limit of integration
c
c            on return
c              result - real
c                       approximation to the integral i
c                       result is computed by applying the 15-point
c                       kronrod rule (resk) obtained by optimal addition
c                       of abscissae to the7-point gauss rule(resg).
c
c              abserr - real
c                       estimate of the modulus of the absolute error,
c                       which should not exceed abs(i-result)
c
c              resabs - real
c                       approximation to the integral j
c
c              resasc - real
c                       approximation to the integral of abs(f-i/(b-a))
c                       over (a,b)
c
c***references  (none)
c***routines called  r1mach
c***end prologue  qk15
c
      real a,absc,abserr,b,centr,dhlgth,epmach,f,fc,fsum,fval1,fval2,
     *  fv1,fv2,hlgth,resabs,resasc,resg,resk,reskh,result,r1mach,uflow,
     *  wg,wgk,xgk
      integer j,jtw,jtwm1
      external f
c
      dimension fv1(7),fv2(7),wg(4),wgk(8),xgk(8)
c
c           the abscissae and weights are given for the interval (-1,1).
c           because of symmetry only the positive abscissae and their
c           corresponding weights are given.
c
c           xgk    - abscissae of the 15-point kronrod rule
c                    xgk(2), xgk(4), ...  abscissae of the 7-point
c                    gauss rule
c                    xgk(1), xgk(3), ...  abscissae which are optimally
c                    added to the 7-point gauss rule
c
c           wgk    - weights of the 15-point kronrod rule
c
c           wg     - weights of the 7-point gauss rule
c
      data xgk(1),xgk(2),xgk(3),xgk(4),xgk(5),xgk(6),xgk(7),xgk(8)/
     *     0.9914553711208126e+00,   0.9491079123427585e+00,
     *     0.8648644233597691e+00,   0.7415311855993944e+00,
     *     0.5860872354676911e+00,   0.4058451513773972e+00,
     *     0.2077849550078985e+00,   0.0e+00              /
      data wgk(1),wgk(2),wgk(3),wgk(4),wgk(5),wgk(6),wgk(7),wgk(8)/
     *     0.2293532201052922e-01,   0.6309209262997855e-01,
     *     0.1047900103222502e+00,   0.1406532597155259e+00,
     *     0.1690047266392679e+00,   0.1903505780647854e+00,
     *     0.2044329400752989e+00,   0.2094821410847278e+00/
      data wg(1),wg(2),wg(3),wg(4)/
     *     0.1294849661688697e+00,   0.2797053914892767e+00,
     *     0.3818300505051189e+00,   0.4179591836734694e+00/
c
c
c           list of major variables
c           -----------------------
c
c           centr  - mid point of the interval
c           hlgth  - half-length of the interval
c           absc   - abscissa
c           fval*  - function value
c           resg   - result of the 7-point gauss formula
c           resk   - result of the 15-point kronrod formula
c           reskh  - approximation to the mean value of f over (a,b),
c                    i.e. to i/(b-a)
c
c           machine dependent constants
c           ---------------------------
c
c           epmach is the largest relative spacing.
c           uflow is the smallest positive magnitude.
c
c***first executable statement  qk15
      epmach = r1mach(4)
      uflow = r1mach(1)
c
      centr = 0.5e+00*(a+b)
      hlgth = 0.5e+00*(b-a)
      dhlgth = abs(hlgth)
c
c           compute the 15-point kronrod approximation to
c           the integral, and estimate the absolute error.
c
      fc = f(centr)
      resg = fc*wg(4)
      resk = fc*wgk(8)
      resabs = abs(resk)
      do 10 j=1,3
        jtw = j*2
        absc = hlgth*xgk(jtw)
        fval1 = f(centr-absc)
        fval2 = f(centr+absc)
        fv1(jtw) = fval1
        fv2(jtw) = fval2
        fsum = fval1+fval2
        resg = resg+wg(j)*fsum
        resk = resk+wgk(jtw)*fsum
        resabs = resabs+wgk(jtw)*(abs(fval1)+abs(fval2))
   10 continue
      do 15 j = 1,4
        jtwm1 = j*2-1
        absc = hlgth*xgk(jtwm1)
        fval1 = f(centr-absc)
        fval2 = f(centr+absc)
        fv1(jtwm1) = fval1
        fv2(jtwm1) = fval2
        fsum = fval1+fval2
        resk = resk+wgk(jtwm1)*fsum
        resabs = resabs+wgk(jtwm1)*(abs(fval1)+abs(fval2))
   15 continue
      reskh = resk*0.5e+00
      resasc = wgk(8)*abs(fc-reskh)
      do 20 j=1,7
        resasc = resasc+wgk(j)*(abs(fv1(j)-reskh)+abs(fv2(j)-reskh))
   20 continue
      result = resk*hlgth
      resabs = resabs*dhlgth
      resasc = resasc*dhlgth
      abserr = abs((resk-resg)*hlgth)
      if(resasc.ne.0.0e+00.and.abserr.ne.0.0e+00)
     *  abserr = resasc*amin1(0.1e+01,
     *  (0.2e+03*abserr/resasc)**1.5e+00)
      if(resabs.gt.uflow/(0.5e+02*epmach)) abserr = amax1
     *  ((epmach*0.5e+02)*resabs,abserr)
      return
      end

c**********************************************************************

      subroutine qk21(f,a,b,result,abserr,resabs,resasc)
c***begin prologue  qk21
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a1a2
c***keywords  21-point gauss-kronrod rules
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  to compute i = integral of f over (a,b), with error
c                           estimate
c                       j = integral of abs(f) over (a,b)
c***description
c
c           integration rules
c           standard fortran subroutine
c           real version
c
c           parameters
c            on entry
c              f      - real
c                       function subprogram defining the integrand
c                       function f(x). the actual name for f needs to be
c                       declared e x t e r n a l in the driver program.
c
c              a      - real
c                       lower limit of integration
c
c              b      - real
c                       upper limit of integration
c
c            on return
c              result - real
c                       approximation to the integral i
c                       result is computed by applying the 21-point
c                       kronrod rule (resk) obtained by optimal addition
c                       of abscissae to the 10-point gauss rule (resg).
c
c              abserr - real
c                       estimate of the modulus of the absolute error,
c                       which should not exceed abs(i-result)
c
c              resabs - real
c                       approximation to the integral j
c
c              resasc - real
c                       approximation to the integral of abs(f-i/(b-a))
c                       over (a,b)
c
c***references  (none)
c***routines called  r1mach
c***end prologue  qk21
c
      real a,absc,abserr,b,centr,dhlgth,epmach,f,fc,fsum,fval1,fval2,
     *  fv1,fv2,hlgth,resabs,resg,resk,reskh,result,r1mach,uflow,wg,wgk,
     *  xgk
      integer j,jtw,jtwm1
      external f
c
      dimension fv1(10),fv2(10),wg(5),wgk(11),xgk(11)
c
c           the abscissae and weights are given for the interval (-1,1).
c           because of symmetry only the positive abscissae and their
c           corresponding weights are given.
c
c           xgk    - abscissae of the 21-point kronrod rule
c                    xgk(2), xgk(4), ...  abscissae of the 10-point
c                    gauss rule
c                    xgk(1), xgk(3), ...  abscissae which are optimally
c                    added to the 10-point gauss rule
c
c           wgk    - weights of the 21-point kronrod rule
c
c           wg     - weights of the 10-point gauss rule
c
      data xgk(1),xgk(2),xgk(3),xgk(4),xgk(5),xgk(6),xgk(7),
     *  xgk(8),xgk(9),xgk(10),xgk(11)/
     *         0.9956571630258081e+00,     0.9739065285171717e+00,
     *     0.9301574913557082e+00,     0.8650633666889845e+00,
     *     0.7808177265864169e+00,     0.6794095682990244e+00,
     *     0.5627571346686047e+00,     0.4333953941292472e+00,
     *     0.2943928627014602e+00,     0.1488743389816312e+00,
     *     0.0000000000000000e+00/
c
      data wgk(1),wgk(2),wgk(3),wgk(4),wgk(5),wgk(6),wgk(7),
     *  wgk(8),wgk(9),wgk(10),wgk(11)/
     *     0.1169463886737187e-01,     0.3255816230796473e-01,
     *     0.5475589657435200e-01,     0.7503967481091995e-01,
     *     0.9312545458369761e-01,     0.1093871588022976e+00,
     *     0.1234919762620659e+00,     0.1347092173114733e+00,
     *     0.1427759385770601e+00,     0.1477391049013385e+00,
     *     0.1494455540029169e+00/
c
      data wg(1),wg(2),wg(3),wg(4),wg(5)/
     *     0.6667134430868814e-01,     0.1494513491505806e+00,
     *     0.2190863625159820e+00,     0.2692667193099964e+00,
     *     0.2955242247147529e+00/
c
c
c           list of major variables
c           -----------------------
c
c           centr  - mid point of the interval
c           hlgth  - half-length of the interval
c           absc   - abscissa
c           fval*  - function value
c           resg   - result of the 10-point gauss formula
c           resk   - result of the 21-point kronrod formula
c           reskh  - approximation to the mean value of f over (a,b),
c                    i.e. to i/(b-a)
c
c
c           machine dependent constants
c           ---------------------------
c
c           epmach is the largest relative spacing.
c           uflow is the smallest positive magnitude.
c
c***first executable statement  qk21
      epmach = r1mach(4)
      uflow = r1mach(1)
c
      centr = 0.5e+00*(a+b)
      hlgth = 0.5e+00*(b-a)
      dhlgth = abs(hlgth)
c
c           compute the 21-point kronrod approximation to
c           the integral, and estimate the absolute error.
c
      resg = 0.0e+00
      fc = f(centr)
      resk = wgk(11)*fc
      resabs = abs(resk)
      do 10 j=1,5
        jtw = 2*j
        absc = hlgth*xgk(jtw)
        fval1 = f(centr-absc)
        fval2 = f(centr+absc)
        fv1(jtw) = fval1
        fv2(jtw) = fval2
        fsum = fval1+fval2
        resg = resg+wg(j)*fsum
        resk = resk+wgk(jtw)*fsum
        resabs = resabs+wgk(jtw)*(abs(fval1)+abs(fval2))
   10 continue
      do 15 j = 1,5
        jtwm1 = 2*j-1
        absc = hlgth*xgk(jtwm1)
        fval1 = f(centr-absc)
        fval2 = f(centr+absc)
        fv1(jtwm1) = fval1
        fv2(jtwm1) = fval2
        fsum = fval1+fval2
        resk = resk+wgk(jtwm1)*fsum
        resabs = resabs+wgk(jtwm1)*(abs(fval1)+abs(fval2))
   15 continue
      reskh = resk*0.5e+00
      resasc = wgk(11)*abs(fc-reskh)
      do 20 j=1,10
        resasc = resasc+wgk(j)*(abs(fv1(j)-reskh)+abs(fv2(j)-reskh))
   20 continue
      result = resk*hlgth
      resabs = resabs*dhlgth
      resasc = resasc*dhlgth
      abserr = abs((resk-resg)*hlgth)
      if(resasc.ne.0.0e+00.and.abserr.ne.0.0e+00)
     *  abserr = resasc*amin1(0.1e+01,
     *  (0.2e+03*abserr/resasc)**1.5e+00)
      if(resabs.gt.uflow/(0.5e+02*epmach)) abserr = amax1
     *  ((epmach*0.5e+02)*resabs,abserr)
      return
      end

c**********************************************************************

      subroutine qk31(f,a,b,result,abserr,resabs,resasc)
c***begin prologue  qk31
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a1a2
c***keywords  31-point gauss-kronrod rules
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  to compute i = integral of f over (a,b) with error
c                           estimate
c                       j = integral of abs(f) over (a,b)
c***description
c
c           integration rules
c           standard fortran subroutine
c           real version
c
c           parameters
c            on entry
c              f      - real
c                       function subprogram defining the integrand
c                       function f(x). the actual name for f needs to be
c                       declared e x t e r n a l in the calling program.
c
c              a      - real
c                       lower limit of integration
c
c              b      - real
c                       upper limit of integration
c
c            on return
c              result - real
c                       approximation to the integral i
c                       result is computed by applying the 31-point
c                       gauss-kronrod rule (resk), obtained by optimal
c                       addition of abscissae to the 15-point gauss
c                       rule (resg).
c
c              abserr - real
c                       estimate of the modulus of the modulus,
c                       which should not exceed abs(i-result)
c
c              resabs - real
c                       approximation to the integral j
c
c              resasc - real
c                       approximation to the integral of abs(f-i/(b-a))
c                       over (a,b)
c
c***references  (none)
c***routines called  r1mach
c***end prologue  qk31
      real a,absc,abserr,b,centr,dhlgth,epmach,f,fc,fsum,fval1,fval2,
     *  fv1,fv2,hlgth,resabs,resasc,resg,resk,reskh,result,r1mach,uflow,
     *  wg,wgk,xgk
      integer j,jtw,jtwm1
      external f
c
      dimension fv1(15),fv2(15),xgk(16),wgk(16),wg(8)
c
c           the abscissae and weights are given for the interval (-1,1).
c           because of symmetry only the positive abscissae and their
c           corresponding weights are given.
c
c           xgk    - abscissae of the 31-point kronrod rule
c                    xgk(2), xgk(4), ...  abscissae of the 15-point
c                    gauss rule
c                    xgk(1), xgk(3), ...  abscissae which are optimally
c                    added to the 15-point gauss rule
c
c           wgk    - weights of the 31-point kronrod rule
c
c           wg     - weights of the 15-point gauss rule
c
      data xgk(1),xgk(2),xgk(3),xgk(4),xgk(5),xgk(6),xgk(7),xgk(8),
     *  xgk(9),xgk(10),xgk(11),xgk(12),xgk(13),xgk(14),xgk(15),
     *  xgk(16)/
     *     0.9980022986933971e+00,   0.9879925180204854e+00,
     *     0.9677390756791391e+00,   0.9372733924007059e+00,
     *     0.8972645323440819e+00,   0.8482065834104272e+00,
     *     0.7904185014424659e+00,   0.7244177313601700e+00,
     *     0.6509967412974170e+00,   0.5709721726085388e+00,
     *     0.4850818636402397e+00,   0.3941513470775634e+00,
     *     0.2991800071531688e+00,   0.2011940939974345e+00,
     *     0.1011420669187175e+00,   0.0e+00               /
      data wgk(1),wgk(2),wgk(3),wgk(4),wgk(5),wgk(6),wgk(7),wgk(8),
     *  wgk(9),wgk(10),wgk(11),wgk(12),wgk(13),wgk(14),wgk(15),
     *  wgk(16)/
     *     0.5377479872923349e-02,   0.1500794732931612e-01,
     *     0.2546084732671532e-01,   0.3534636079137585e-01,
     *     0.4458975132476488e-01,   0.5348152469092809e-01,
     *     0.6200956780067064e-01,   0.6985412131872826e-01,
     *     0.7684968075772038e-01,   0.8308050282313302e-01,
     *     0.8856444305621177e-01,   0.9312659817082532e-01,
     *     0.9664272698362368e-01,   0.9917359872179196e-01,
     *     0.1007698455238756e+00,   0.1013300070147915e+00/
      data wg(1),wg(2),wg(3),wg(4),wg(5),wg(6),wg(7),wg(8)/
     *     0.3075324199611727e-01,   0.7036604748810812e-01,
     *     0.1071592204671719e+00,   0.1395706779261543e+00,
     *     0.1662692058169939e+00,   0.1861610000155622e+00,
     *     0.1984314853271116e+00,   0.2025782419255613e+00/
c
c
c           list of major variables
c           -----------------------
c           centr  - mid point of the interval
c           hlgth  - half-length of the interval
c           absc   - abscissa
c           fval*  - function value
c           resg   - result of the 15-point gauss formula
c           resk   - result of the 31-point kronrod formula
c           reskh  - approximation to the mean value of f over (a,b),
c                    i.e. to i/(b-a)
c
c           machine dependent constants
c           ---------------------------
c           epmach is the largest relative spacing.
c           uflow is the smallest positive magnitude.
c
c***first executable statement  qk31
      epmach = r1mach(4)
      uflow = r1mach(1)
c
      centr = 0.5e+00*(a+b)
      hlgth = 0.5e+00*(b-a)
      dhlgth = abs(hlgth)
c
c           compute the 31-point kronrod approximation to
c           the integral, and estimate the absolute error.
c
      fc = f(centr)
      resg = wg(8)*fc
      resk = wgk(16)*fc
      resabs = abs(resk)
      do 10 j=1,7
        jtw = j*2
        absc = hlgth*xgk(jtw)
        fval1 = f(centr-absc)
        fval2 = f(centr+absc)
        fv1(jtw) = fval1
        fv2(jtw) = fval2
        fsum = fval1+fval2
        resg = resg+wg(j)*fsum
        resk = resk+wgk(jtw)*fsum
        resabs = resabs+wgk(jtw)*(abs(fval1)+abs(fval2))
   10 continue
      do 15 j = 1,8
        jtwm1 = j*2-1
        absc = hlgth*xgk(jtwm1)
        fval1 = f(centr-absc)
        fval2 = f(centr+absc)
        fv1(jtwm1) = fval1
        fv2(jtwm1) = fval2
        fsum = fval1+fval2
        resk = resk+wgk(jtwm1)*fsum
        resabs = resabs+wgk(jtwm1)*(abs(fval1)+abs(fval2))
   15 continue
      reskh = resk*0.5e+00
      resasc = wgk(16)*abs(fc-reskh)
      do 20 j=1,15
        resasc = resasc+wgk(j)*(abs(fv1(j)-reskh)+abs(fv2(j)-reskh))
   20 continue
      result = resk*hlgth
      resabs = resabs*dhlgth
      resasc = resasc*dhlgth
      abserr = abs((resk-resg)*hlgth)
      if(resasc.ne.0.0e+00.and.abserr.ne.0.0e+00)
     *  abserr = resasc*amin1(0.1e+01,
     *  (0.2e+03*abserr/resasc)**1.5e+00)
      if(resabs.gt.uflow/(0.5e+02*epmach)) abserr = amax1
     *  ((epmach*0.5e+02)*resabs,abserr)
      return
      end

c**********************************************************************

      subroutine qk41(f,a,b,result,abserr,resabs,resasc)
c***begin prologue  qk41
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a1a2
c***keywords  41-point gauss-kronrod rules
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  to compute i = integral of f over (a,b), with error
c                           estimate
c                       j = integral of abs(f) over (a,b)
c***description
c
c           integration rules
c           standard fortran subroutine
c           real version
c
c           parameters
c            on entry
c              f      - real
c                       function subprogram defining the integrand
c                       function f(x). the actual name for f needs to be
c                       declared e x t e r n a l in the calling program.
c
c              a      - real
c                       lower limit of integration
c
c              b      - real
c                       upper limit of integration
c
c            on return
c              result - real
c                       approximation to the integral i
c                       result is computed by applying the 41-point
c                       gauss-kronrod rule (resk) obtained by optimal
c                       addition of abscissae to the 20-point gauss
c                       rule (resg).
c
c              abserr - real
c                       estimate of the modulus of the absolute error,
c                       which should not exceed abs(i-result)
c
c              resabs - real
c                       approximation to the integral j
c
c              resasc - real
c                       approximation to the integal of abs(f-i/(b-a))
c                       over (a,b)
c
c***references  (none)
c***routines called  r1mach
c***end prologue  qk41
c
      real a,absc,abserr,b,centr,dhlgth,epmach,f,fc,fsum,fval1,fval2,
     *  fv1,fv2,hlgth,resabs,
     *  resasc,resg,resk,reskh,result,r1mach,uflow,
     *  wg,wgk,xgk
      integer j,jtw,jtwm1
      external f
c
      dimension fv1(20),fv2(20),xgk(21),wgk(21),wg(10)
c
c           the abscissae and weights are given for the interval (-1,1).
c           because of symmetry only the positive abscissae and their
c           corresponding weights are given.
c
c           xgk    - abscissae of the 41-point gauss-kronrod rule
c                    xgk(2), xgk(4), ...  abscissae of the 20-point
c                    gauss rule
c                    xgk(1), xgk(3), ...  abscissae which are optimally
c                    added to the 20-point gauss rule
c
c           wgk    - weights of the 41-point gauss-kronrod rule
c
c           wg     - weights of the 20-point gauss rule
c
      data xgk(1),xgk(2),xgk(3),xgk(4),xgk(5),xgk(6),xgk(7),xgk(8),
     *  xgk(9),xgk(10),xgk(11),xgk(12),xgk(13),xgk(14),xgk(15),
     *  xgk(16),xgk(17),xgk(18),xgk(19),xgk(20),xgk(21)/
     *     0.9988590315882777e+00,   0.9931285991850949e+00,
     *     0.9815078774502503e+00,   0.9639719272779138e+00,
     *     0.9408226338317548e+00,   0.9122344282513259e+00,
     *     0.8782768112522820e+00,   0.8391169718222188e+00,
     *     0.7950414288375512e+00,   0.7463319064601508e+00,
     *     0.6932376563347514e+00,   0.6360536807265150e+00,
     *     0.5751404468197103e+00,   0.5108670019508271e+00,
     *     0.4435931752387251e+00,   0.3737060887154196e+00,
     *     0.3016278681149130e+00,   0.2277858511416451e+00,
     *     0.1526054652409227e+00,   0.7652652113349733e-01,
     *     0.0e+00               /
      data wgk(1),wgk(2),wgk(3),wgk(4),wgk(5),wgk(6),wgk(7),wgk(8),
     *  wgk(9),wgk(10),wgk(11),wgk(12),wgk(13),wgk(14),wgk(15),wgk(16),
     *  wgk(17),wgk(18),wgk(19),wgk(20),wgk(21)/
     *     0.3073583718520532e-02,   0.8600269855642942e-02,
     *     0.1462616925697125e-01,   0.2038837346126652e-01,
     *     0.2588213360495116e-01,   0.3128730677703280e-01,
     *     0.3660016975820080e-01,   0.4166887332797369e-01,
     *     0.4643482186749767e-01,   0.5094457392372869e-01,
     *     0.5519510534828599e-01,   0.5911140088063957e-01,
     *     0.6265323755478117e-01,   0.6583459713361842e-01,
     *     0.6864867292852162e-01,   0.7105442355344407e-01,
     *     0.7303069033278667e-01,   0.7458287540049919e-01,
     *     0.7570449768455667e-01,   0.7637786767208074e-01,
     *     0.7660071191799966e-01/
      data wg(1),wg(2),wg(3),wg(4),wg(5),wg(6),wg(7),wg(8),wg(9),wg(10)/
     *     0.1761400713915212e-01,    0.4060142980038694e-01,
     *     0.6267204833410906e-01,    0.8327674157670475e-01,
     *     0.1019301198172404e+00,    0.1181945319615184e+00,
     *     0.1316886384491766e+00,    0.1420961093183821e+00,
     *     0.1491729864726037e+00,    0.1527533871307259e+00/
c
c
c           list of major variables
c           -----------------------
c
c           centr  - mid point of the interval
c           hlgth  - half-length of the interval
c           absc   - abscissa
c           fval*  - function value
c           resg   - result of the 20-point gauss formula
c           resk   - result of the 41-point kronrod formula
c           reskh  - approximation to mean value of f over (a,b), i.e.
c                    to i/(b-a)
c
c           machine dependent constants
c           ---------------------------
c
c           epmach is the largest relative spacing.
c           uflow is the smallest positive magnitude.
c
c***first executable statement  qk41
      epmach = r1mach(4)
      uflow = r1mach(1)
c
      centr = 0.5e+00*(a+b)
      hlgth = 0.5e+00*(b-a)
      dhlgth = abs(hlgth)
c
c           compute the 41-point gauss-kronrod approximation to
c           the integral, and estimate the absolute error.
c
      resg = 0.0e+00
      fc = f(centr)
      resk = wgk(21)*fc
      resabs = abs(resk)
      do 10 j=1,10
        jtw = j*2
        absc = hlgth*xgk(jtw)
        fval1 = f(centr-absc)
        fval2 = f(centr+absc)
        fv1(jtw) = fval1
        fv2(jtw) = fval2
        fsum = fval1+fval2
        resg = resg+wg(j)*fsum
        resk = resk+wgk(jtw)*fsum
        resabs = resabs+wgk(jtw)*(abs(fval1)+abs(fval2))
   10 continue
      do 15 j = 1,10
        jtwm1 = j*2-1
        absc = hlgth*xgk(jtwm1)
        fval1 = f(centr-absc)
        fval2 = f(centr+absc)
        fv1(jtwm1) = fval1
        fv2(jtwm1) = fval2
        fsum = fval1+fval2
        resk = resk+wgk(jtwm1)*fsum
        resabs = resabs+wgk(jtwm1)*(abs(fval1)+abs(fval2))
   15 continue
      reskh = resk*0.5e+00
      resasc = wgk(21)*abs(fc-reskh)
      do 20 j=1,20
        resasc = resasc+wgk(j)*(abs(fv1(j)-reskh)+abs(fv2(j)-reskh))
   20 continue
      result = resk*hlgth
      resabs = resabs*dhlgth
      resasc = resasc*dhlgth
      abserr = abs((resk-resg)*hlgth)
      if(resasc.ne.0.0e+00.and.abserr.ne.0.e+00)
     *  abserr = resasc*amin1(0.1e+01,
     *  (0.2e+03*abserr/resasc)**1.5e+00)
      if(resabs.gt.uflow/(0.5e+02*epmach)) abserr = amax1
     *  ((epmach*0.5e+02)*resabs,abserr)
      return
      end

c**********************************************************************

      subroutine qk51(f,a,b,result,abserr,resabs,resasc)
c***begin prologue  qk51
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a1a2
c***keywords  51-point gauss-kronrod rules
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math & progr. div. - k.u.leuven
c***purpose  to compute i = integral of f over (a,b) with error
c                           estimate
c                       j = integral of abs(f) over (a,b)
c***description
c
c           integration rules
c           standard fortran subroutine
c           real version
c
c           parameters
c            on entry
c              f      - real
c                       function subroutine defining the integrand
c                       function f(x). the actual name for f needs to be
c                       declared e x t e r n a l in the calling program.
c
c              a      - real
c                       lower limit of integration
c
c              b      - real
c                       upper limit of integration
c
c            on return
c              result - real
c                       approximation to the integral i
c                       result is computed by applying the 51-point
c                       kronrod rule (resk) obtained by optimal addition
c                       of abscissae to the 25-point gauss rule (resg).
c
c              abserr - real
c                       estimate of the modulus of the absolute error,
c                       which should not exceed abs(i-result)
c
c              resabs - real
c                       approximation to the integral j
c
c              resasc - real
c                       approximation to the integral of abs(f-i/(b-a))
c                       over (a,b)
c
c***references  (none)
c***routines called  r1mach
c***end prologue  qk51
c
      real a,absc,abserr,b,centr,dhlgth,epmach,f,fc,fsum,fval1,fval2,
     *  fv1,fv2,hlgth,resabs,resasc,resg,resk,reskh,result,r1mach,uflow,
     *  wg,wgk,xgk
      integer j,jtw,jtwm1
      external f
c
      dimension fv1(25),fv2(25),xgk(26),wgk(26),wg(13)
c
c           the abscissae and weights are given for the interval (-1,1).
c           because of symmetry only the positive abscissae and their
c           corresponding weights are given.
c
c           xgk    - abscissae of the 51-point kronrod rule
c                    xgk(2), xgk(4), ...  abscissae of the 25-point
c                    gauss rule
c                    xgk(1), xgk(3), ...  abscissae which are optimally
c                    added to the 25-point gauss rule
c
c           wgk    - weights of the 51-point kronrod rule
c
c           wg     - weights of the 25-point gauss rule
c
      data xgk(1),xgk(2),xgk(3),xgk(4),xgk(5),xgk(6),xgk(7),xgk(8),
     *  xgk(9),xgk(10),xgk(11),xgk(12),xgk(13),xgk(14)/
     *     0.9992621049926098e+00,   0.9955569697904981e+00,
     *     0.9880357945340772e+00,   0.9766639214595175e+00,
     *     0.9616149864258425e+00,   0.9429745712289743e+00,
     *     0.9207471152817016e+00,   0.8949919978782754e+00,
     *     0.8658470652932756e+00,   0.8334426287608340e+00,
     *     0.7978737979985001e+00,   0.7592592630373576e+00,
     *     0.7177664068130844e+00,   0.6735663684734684e+00/
       data xgk(15),xgk(16),xgk(17),xgk(18),xgk(19),xgk(20),xgk(21),
     *  xgk(22),xgk(23),xgk(24),xgk(25),xgk(26)/
     *     0.6268100990103174e+00,   0.5776629302412230e+00,
     *     0.5263252843347192e+00,   0.4730027314457150e+00,
     *     0.4178853821930377e+00,   0.3611723058093878e+00,
     *     0.3030895389311078e+00,   0.2438668837209884e+00,
     *     0.1837189394210489e+00,   0.1228646926107104e+00,
     *     0.6154448300568508e-01,   0.0e+00               /
      data wgk(1),wgk(2),wgk(3),wgk(4),wgk(5),wgk(6),wgk(7),wgk(8),
     *  wgk(9),wgk(10),wgk(11),wgk(12),wgk(13),wgk(14)/
     *     0.1987383892330316e-02,   0.5561932135356714e-02,
     *     0.9473973386174152e-02,   0.1323622919557167e-01,
     *     0.1684781770912830e-01,   0.2043537114588284e-01,
     *     0.2400994560695322e-01,   0.2747531758785174e-01,
     *     0.3079230016738749e-01,   0.3400213027432934e-01,
     *     0.3711627148341554e-01,   0.4008382550403238e-01,
     *     0.4287284502017005e-01,   0.4550291304992179e-01/
       data wgk(15),wgk(16),wgk(17),wgk(18),wgk(19),wgk(20),wgk(21)
     *  ,wgk(22),wgk(23),wgk(24),wgk(25),wgk(26)/
     *     0.4798253713883671e-01,   0.5027767908071567e-01,
     *     0.5236288580640748e-01,   0.5425112988854549e-01,
     *     0.5595081122041232e-01,   0.5743711636156783e-01,
     *     0.5868968002239421e-01,   0.5972034032417406e-01,
     *     0.6053945537604586e-01,   0.6112850971705305e-01,
     *     0.6147118987142532e-01,   0.6158081806783294e-01/
      data wg(1),wg(2),wg(3),wg(4),wg(5),wg(6),wg(7),wg(8),wg(9),
     *  wg(10),wg(11),wg(12),wg(13)/
     *     0.1139379850102629e-01,    0.2635498661503214e-01,
     *     0.4093915670130631e-01,    0.5490469597583519e-01,
     *     0.6803833381235692e-01,    0.8014070033500102e-01,
     *     0.9102826198296365e-01,    0.1005359490670506e+00,
     *     0.1085196244742637e+00,    0.1148582591457116e+00,
     *     0.1194557635357848e+00,    0.1222424429903100e+00,
     *     0.1231760537267155e+00/
c
c
c           list of major variables
c           -----------------------
c
c           centr  - mid point of the interval
c           hlgth  - half-length of the interval
c           absc   - abscissa
c           fval*  - function value
c           resg   - result of the 25-point gauss formula
c           resk   - result of the 51-point kronrod formula
c           reskh  - approximation to the mean value of f over (a,b),
c                    i.e. to i/(b-a)
c
c           machine dependent constants
c           ---------------------------
c
c           epmach is the largest relative spacing.
c           uflow is the smallest positive magnitude.
c
c***first executable statement  qk51
      epmach = r1mach(4)
      uflow = r1mach(1)
c
      centr = 0.5e+00*(a+b)
      hlgth = 0.5e+00*(b-a)
      dhlgth = abs(hlgth)
c
c           compute the 51-point kronrod approximation to
c           the integral, and estimate the absolute error.
c
      fc = f(centr)
      resg = wg(13)*fc
      resk = wgk(26)*fc
      resabs = abs(resk)
      do 10 j=1,12
        jtw = j*2
        absc = hlgth*xgk(jtw)
        fval1 = f(centr-absc)
        fval2 = f(centr+absc)
        fv1(jtw) = fval1
        fv2(jtw) = fval2
        fsum = fval1+fval2
        resg = resg+wg(j)*fsum
        resk = resk+wgk(jtw)*fsum
        resabs = resabs+wgk(jtw)*(abs(fval1)+abs(fval2))
   10 continue
      do 15 j = 1,13
        jtwm1 = j*2-1
        absc = hlgth*xgk(jtwm1)
        fval1 = f(centr-absc)
        fval2 = f(centr+absc)
        fv1(jtwm1) = fval1
        fv2(jtwm1) = fval2
        fsum = fval1+fval2
        resk = resk+wgk(jtwm1)*fsum
        resabs = resabs+wgk(jtwm1)*(abs(fval1)+abs(fval2))
   15 continue
      reskh = resk*0.5e+00
      resasc = wgk(26)*abs(fc-reskh)
      do 20 j=1,25
        resasc = resasc+wgk(j)*(abs(fv1(j)-reskh)+abs(fv2(j)-reskh))
   20 continue
      result = resk*hlgth
      resabs = resabs*dhlgth
      resasc = resasc*dhlgth
      abserr = abs((resk-resg)*hlgth)
      if(resasc.ne.0.0e+00.and.abserr.ne.0.0e+00)
     *  abserr = resasc*amin1(0.1e+01,
     *  (0.2e+03*abserr/resasc)**1.5e+00)
      if(resabs.gt.uflow/(0.5e+02*epmach)) abserr = amax1
     *  ((epmach*0.5e+02)*resabs,abserr)
      return
      end

c**********************************************************************

      subroutine qk61(f,a,b,result,abserr,resabs,resasc)
c***begin prologue  qk61
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a1a2
c***keywords  61-point gauss-kronrod rules
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  to compute i = integral of f over (a,b) with error
c                           estimate
c                       j = integral of dabs(f) over (a,b)
c***description
c
c        integration rule
c        standard fortran subroutine
c        real version
c
c
c        parameters
c         on entry
c           f      - real
c                    function subprogram defining the integrand
c                    function f(x). the actual name for f needs to be
c                    declared e x t e r n a l in the calling program.
c
c           a      - real
c                    lower limit of integration
c
c           b      - real
c                    upper limit of integration
c
c         on return
c           result - real
c                    approximation to the integral i
c                    result is computed by applying the 61-point
c                    kronrod rule (resk) obtained by optimal addition of
c                    abscissae to the 30-point gauss rule (resg).
c
c           abserr - real
c                    estimate of the modulus of the absolute error,
c                    which should equal or exceed dabs(i-result)
c
c           resabs - real
c                    approximation to the integral j
c
c           resasc - real
c                    approximation to the integral of dabs(f-i/(b-a))
c
c
c***references  (none)
c***routines called  r1mach
c***end prologue  qk61
c
      real a,absc,abserr,b,centr,dhlgth,epmach,f,fc,fsum,fval1,fval2,
     *  fv1,fv2,hlgth,resabs,resasc,resg,resk,reskh,result,r1mach,uflow,
     *  wg,wgk,xgk
      integer j,jtw,jtwm1
      external f
c
      dimension fv1(30),fv2(30),xgk(31),wgk(31),wg(15)
c
c           the abscissae and weights are given for the
c           interval (-1,1). because of symmetry only the positive
c           abscissae and their corresponding weights are given.
c
c           xgk   - abscissae of the 61-point kronrod rule
c                   xgk(2), xgk(4)  ... abscissae of the 30-point
c                   gauss rule
c                   xgk(1), xgk(3)  ... optimally added abscissae
c                   to the 30-point gauss rule
c
c           wgk   - weights of the 61-point kronrod rule
c
c           wg    - weigths of the 30-point gauss rule
c
      data xgk(1),xgk(2),xgk(3),xgk(4),xgk(5),xgk(6),xgk(7),xgk(8),
     *   xgk(9),xgk(10)/
     *     0.9994844100504906e+00,     0.9968934840746495e+00,
     *     0.9916309968704046e+00,     0.9836681232797472e+00,
     *     0.9731163225011263e+00,     0.9600218649683075e+00,
     *     0.9443744447485600e+00,     0.9262000474292743e+00,
     *     0.9055733076999078e+00,     0.8825605357920527e+00/
      data xgk(11),xgk(12),xgk(13),xgk(14),xgk(15),xgk(16),
     *  xgk(17),xgk(18),xgk(19),xgk(20)/
     *     0.8572052335460611e+00,     0.8295657623827684e+00,
     *     0.7997278358218391e+00,     0.7677774321048262e+00,
     *     0.7337900624532268e+00,     0.6978504947933158e+00,
     *     0.6600610641266270e+00,     0.6205261829892429e+00,
     *     0.5793452358263617e+00,     0.5366241481420199e+00/
      data xgk(21),xgk(22),xgk(23),xgk(24),
     *  xgk(25),xgk(26),xgk(27),xgk(28),xgk(29),xgk(30),xgk(31)/
     *     0.4924804678617786e+00,     0.4470337695380892e+00,
     *     0.4004012548303944e+00,     0.3527047255308781e+00,
     *     0.3040732022736251e+00,     0.2546369261678898e+00,
     *     0.2045251166823099e+00,     0.1538699136085835e+00,
     *     0.1028069379667370e+00,     0.5147184255531770e-01,
     *     0.0e+00                   /
      data wgk(1),wgk(2),wgk(3),wgk(4),wgk(5),wgk(6),wgk(7),wgk(8),
     *  wgk(9),wgk(10)/
     *     0.1389013698677008e-02,     0.3890461127099884e-02,
     *     0.6630703915931292e-02,     0.9273279659517763e-02,
     *     0.1182301525349634e-01,     0.1436972950704580e-01,
     *     0.1692088918905327e-01,     0.1941414119394238e-01,
     *     0.2182803582160919e-01,     0.2419116207808060e-01/
      data wgk(11),wgk(12),wgk(13),wgk(14),wgk(15),wgk(16),
     *  wgk(17),wgk(18),wgk(19),wgk(20)/
     *     0.2650995488233310e-01,     0.2875404876504129e-01,
     *     0.3090725756238776e-01,     0.3298144705748373e-01,
     *     0.3497933802806002e-01,     0.3688236465182123e-01,
     *     0.3867894562472759e-01,     0.4037453895153596e-01,
     *     0.4196981021516425e-01,     0.4345253970135607e-01/
      data wgk(21),wgk(22),wgk(23),wgk(24),
     *  wgk(25),wgk(26),wgk(27),wgk(28),wgk(29),wgk(30),wgk(31)/
     *     0.4481480013316266e-01,     0.4605923827100699e-01,
     *     0.4718554656929915e-01,     0.4818586175708713e-01,
     *     0.4905543455502978e-01,     0.4979568342707421e-01,
     *     0.5040592140278235e-01,     0.5088179589874961e-01,
     *     0.5122154784925877e-01,     0.5142612853745903e-01,
     *     0.5149472942945157e-01/
      data wg(1),wg(2),wg(3),wg(4),wg(5),wg(6),wg(7),wg(8)/
     *     0.7968192496166606e-02,     0.1846646831109096e-01,
     *     0.2878470788332337e-01,     0.3879919256962705e-01,
     *     0.4840267283059405e-01,     0.5749315621761907e-01,
     *     0.6597422988218050e-01,     0.7375597473770521e-01/
      data wg(9),wg(10),wg(11),wg(12),wg(13),wg(14),wg(15)/
     *     0.8075589522942022e-01,     0.8689978720108298e-01,
     *     0.9212252223778613e-01,     0.9636873717464426e-01,
     *     0.9959342058679527e-01,     0.1017623897484055e+00,
     *     0.1028526528935588e+00/
c
c           list of major variables
c           -----------------------
c
c           centr  - mid point of the interval
c           hlgth  - half-length of the interval
c           absc   - abscissa
c           fval*  - function value
c           resg   - result of the 30-point gauss rule
c           resk   - result of the 61-point kronrod rule
c           reskh  - approximation to the mean value of f
c                    over (a,b), i.e. to i/(b-a)
c
c           machine dependent constants
c           ---------------------------
c
c           epmach is the largest relative spacing.
c           uflow is the smallest positive magnitude.
c
c***first executable statement  qk61
      epmach = r1mach(4)
      uflow = r1mach(1)
c
      centr = 0.5e+00*(b+a)
      hlgth = 0.5e+00*(b-a)
      dhlgth = abs(hlgth)
c
c           compute the 61-point kronrod approximation to the
c           integral, and estimate the absolute error.
c
      resg = 0.0e+00
      fc = f(centr)
      resk = wgk(31)*fc
      resabs = abs(resk)
      do 10 j=1,15
        jtw = j*2
        absc = hlgth*xgk(jtw)
        fval1 = f(centr-absc)
        fval2 = f(centr+absc)
        fv1(jtw) = fval1
        fv2(jtw) = fval2
        fsum = fval1+fval2
        resg = resg+wg(j)*fsum
        resk = resk+wgk(jtw)*fsum
        resabs = resabs+wgk(jtw)*(abs(fval1)+abs(fval2))
   10 continue
      do 15 j=1,15
        jtwm1 = j*2-1
        absc = hlgth*xgk(jtwm1)
        fval1 = f(centr-absc)
        fval2 = f(centr+absc)
        fv1(jtwm1) = fval1
        fv2(jtwm1) = fval2
        fsum = fval1+fval2
        resk = resk+wgk(jtwm1)*fsum
        resabs = resabs+wgk(jtwm1)*(abs(fval1)+abs(fval2))
  15    continue
      reskh = resk*0.5e+00
      resasc = wgk(31)*abs(fc-reskh)
      do 20 j=1,30
        resasc = resasc+wgk(j)*(abs(fv1(j)-reskh)+abs(fv2(j)-reskh))
   20 continue
      result = resk*hlgth
      resabs = resabs*dhlgth
      resasc = resasc*dhlgth
      abserr = abs((resk-resg)*hlgth)
      if(resasc.ne.0.0e+00.and.abserr.ne.0.0e+00)
     *  abserr = resasc*amin1(0.1e+01,
     *  (0.2e+03*abserr/resasc)**1.5e+00)
      if(resabs.gt.uflow/(0.5e+02*epmach)) abserr = amax1
     *  ((epmach*0.5e+02)*resabs,abserr)
      return
      end

c**********************************************************************
