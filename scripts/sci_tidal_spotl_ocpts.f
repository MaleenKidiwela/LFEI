c  sci_tidal_spotl_ocpts.f - helper for scripts/sci_tidal_ocean_spotl.py (LFE project).
c
c  Returns SPOTL's own ocean-load value (complex tidal amplitude times
c  sea-water density, kg/m**2, Greenwich phase convention of the model
c  files) at arbitrary points, using SPOTL's routine ocmodl (and hence its
c  1/64-deg land-sea database, bilinear interpolation with plane fill-in
c  near the coast, and the WOA sea-water density).  Optionally a SPOTL
c  polygon file restricts the region, exactly as in nloadf.
c
c  usage:  ocpts modelfile F|C [polyfile [+|-]]  < latlon.txt > amps.txt
c     input lines: lat lon (deg N, deg E);  output lines: re im
c  Must run in tools/spotl/working (land-sea files are hard-wired there).
c  Build (from tools/spotl/src):
c   gfortran -O2 -std=legacy -fno-automatic ../../../scripts/sci_tidal_spotl_ocpts.f
c        ocst.a mapst.a seadens.f -o ../bin/ocpts
c
      program ocpts
      logical*1 ispoly,use
      character*1 fingrd,alpoly
      character*80 mdfile,polyf,polynm,polnam,dumm
      character*50 mdnam
      character*4 dsym
      complex amp
      common/modpar/mdfile,dsym,icte(6),mdnam
      common/polinf/xpoly(500,10),ypoly(500,10),np,npoly(10),
     1              polyf,polynm(10),use(10),ispoly
      common/polmor/polnam,alpoly
      common/coninf/dist,close,clat,clong,numnf,clnf,farnf
      common/ldens/rhol
      ispoly = .false.
      alpoly = 'N'
      call getarg(1,mdfile)
      call getarg(2,dumm)
      fingrd = dumm(1:1)
      if(iargc().ge.3) then
         call getarg(3,polyf)
         ispoly = .true.
         alpoly = 'Y'
         if(iargc().ge.4) then
            call getarg(4,dumm)
            alpoly = dumm(1:1)
         endif
      endif
      dist = 0.
      numnf = 0
 1    read(5,*,end=9) rlat,rlon
      call ocmodl(rlat,rlon,fingrd,amp)
      write(6,100) real(amp),aimag(amp)
 100  format(2e15.7)
      go to 1
 9    stop
      end
