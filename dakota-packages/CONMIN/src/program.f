      SUBROUTINE RUN_EXAMPLE(XIN, VLBIN, VUBIN, GOUT)
      DOUBLE PRECISION XIN(6), VLBIN(6), VUBIN(6), GOUT(11)
      
      DIMENSION S(6),G1(11),G2(11),B(11,11),C(11),MS1(22) 
      DIMENSION VLB(6),VUB(6),SCAL(6),DF(6),A(6,11),
     .          ISC(11),IC(11)

      COMMON /CNMN1/ DELFUN,DABFUN,FDCH,FDCHM,CT,CTMIN,CTL,CTLMIN,
     .               ALPHAX,ABOBJ1,THETA,OBJ,NDV,NCON,NSIDE,IPRINT, 
     .               NFDG,NSCAL,LINOBJ,ITMAX,ITRM,ICNDIR,IGOTO,NAC, 
     .               INFO,INFOG,ITER
      COMMON /VARABLE/ AOBJ,X(6),G(11)
      COMMON /ANDATA/ LOOPCNT 
      NAMELIST /CONPAR/ INFOG,INFO,NFDG,IPRINT,NDV,ITMAX,NCON,NSIDE,
     .                  ICNDIR,NSCAL,FDCH,FDCHM,CT,CTMIN,CTLMIN,THETA,
     .                  PHI,DELFUN,DABFUN,LINOBJ,ITRM,X,VLB,VUB,
     .                  N1,N2,N3,N4,N5,ALPHAX,ABOBJ1,CTL,ISC,SCAL 

C     Copy inputs into COMMON arrays
      DO I = 1, 6
         X(I) = XIN(I)
         VLB(I) = VLBIN(I)
         VUB(I) = VUBIN(I)
      END DO

C     Initialization (same as before)
      INFOG=0 
      INFO=0
      NFDG=0
      IPRINT=2
      NDV=4 
      ITMAX=40
      NCON=3
      NSIDE=0 
      ICNDIR=0
      NSCAL=0 
      FDCH=0.0
      FDCHM=0.0 
      CT=0.0
      CTMIN=0.0 
      CTL=0.0 
      CTLMIN=0.0
      THETA=0.0 
      PHI=0.0 
      DELFUN=0.0
      DABFUN=0.0
      LINOBJ=0.0
      ITRM=0
      N1=6
      N2=11 
      N3=11 
      N4=11 
      N5=22 
      ALPHAX=0.0
      ABOBJ1=0.0
      CTL=0.0 

      DO 6 J=1,NCON 
        ISC(J)=0
    6 CONTINUE

      NLIM=ITMAX*(NDV+5)
      IGOTO = 0 

      DO 1000 I = 1, NLIM
         LOOPCNT=I 
         CALL CONMIN(X,VLB,VUB,G,SCAL,DF,A,S,G1,G2,B,C,ISC,IC,MS1,
     .               N1,N2,N3,N4,N5) 
         IF(IGOTO.EQ.0) LOOPCNT=-999 
         CALL ANALYS 
         OBJ=AOBJ
         IF (IGOTO.EQ.0) GO TO 1100
 1000 CONTINUE

 1100 CONTINUE

C     Copy result G to output array
      DO I = 1, 11
         GOUT(I) = G(I)
      END DO

      RETURN
      END


      SUBROUTINE ANALYS 
      COMMON /VARABLE/ AOBJ,X(6),G(11)
      COMMON/GRAD/ ISC(11),IC(11),DF(6),A(6,11) 
      COMMON /CNMN1/ DELFUN,DABFUN,FDCH,FDCHM,CT,CTMIN,CTL,CTLMIN,
     .               ALPHAX,ABOBJ1,THETA,OBJ,NDV,NCON,NSIDE,IPRINT, 
     .               NFDG,NSCAL,LINOBJ,ITMAX,ITRM,ICNDIR,IGOTO,NAC, 
     .               INFO,INFOG,ITER
C 
C   ROUTINE TO CALCULATE OBJECTIVE FUNCTION AND 
C   CONSTRAINT  VALUES FOR OPTIMIZATION OF CONSTRAINED ROSEN-SUZUKI 
C   FUNCTION. 
C 
C 
      IF(INFO.GE.2) GO TO 10
C 
C  OBJECTIVE FUNCTION 
C 
      AOBJ = X(1)**2 - 5.*X(1) + X(2)**2 - 5.*X(2) + 2.*X(3)**2 
     .       - 21.*X(3) + X(4)**2 + 7.0*X(4) + 50.
C 
C 
C   CONSTRAINT VALUES 
C 
      G(1) = X(1)**2 + X(1) + X(2)**2 - X(2) + X(3)**2 + X(3) 
     .       + X(4)**2 - X(4) - 8.0
C 
      G(2) = X(1)**2 - X(1) + 2. * X(2)**2 + X(3)**2 + 2.*X(4)**2 
     .       - X(4) - 10.0 
C 
      G(3) = 2.*X(1)**2 + 2.*X(1) + X(2)**2 - X(2) + X(3)**2 - X(4) -5.0 
C 
      GO TO 999 
   10 CONTINUE
C 
C 
C    GRADIENT INFORMATION 
C 
      DF(1)=2.0*X(1) - 5.0
      DF(2)=2.0*X(2) - 5.0
      DF(3)=4.0*X(3) - 21.
      DF(4)=2.0*X(4) + 7. 
C 
C  GRADIENTS OF ACTIVE AND VIOLATED CONSTRAINTS 
C 
      NAC=0 
      IF(G(1).LT.CT) GO TO 20 
      NAC=1 
      IC(1)=1 
      A(1,1)=2.*X(1)+1. 
      A(2,1)=2.*X(2)-1. 
      A(3,1)=2.*X(3)+1. 
      A(4,1)=2.*X(4)-1. 
C 
   20 IF(G(2).LT.CT) GO TO 30 
      NAC=NAC+1 
      IC(NAC)=2 
      A(1,NAC)=2.*X(1)-1.0
      A(2,NAC)=4.*X(2)
      A(3,NAC)=2.*X(3)
      A(4,NAC)=4.*X(4)-1.0
C 
   30 IF(G(3).LT.CT) GO TO 999
      NAC=NAC+1 
      IC(NAC)=3 
      A(1,NAC)=4.*X(1)+2. 
      A(2,NAC)=2.*X(2)-1. 
      A(3,NAC)=2.*X(3)
      A(4,NAC)=-1.
  999 RETURN
      END 

CCCCC