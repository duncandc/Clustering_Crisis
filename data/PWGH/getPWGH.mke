# Compile UNIX (SUN) version of software for getPWGH.f

COMPILER = gfortran

FC = gfortran

OBJECTS  = getPWGH.o cosmo_sub.o quadpack.o

getPWGH: ${OBJECTS}
	 ${COMPILER} -o getPWGH ${OBJECTS} -lm

getPWGH.o: getPWGH.f
	   ${COMPILER} -c getPWGH.f

${OBJECTS}: paramfile.h
 
