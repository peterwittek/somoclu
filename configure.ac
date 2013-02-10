#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.61])
AC_INIT([Somoclu], [1.0])
m4_include([m4/ax_mpi_options.m4])
m4_include([m4/ax_mpi_tests.m4])
AC_CONFIG_SRCDIR([src/somoclu.cpp])
AC_CONFIG_HEADERS([config.h])

# Checks for programs.
CXXFLAGS="-O3"
AC_PROG_CXX
AC_PROG_INSTALL

AC_OPENMP
CXXFLAGS="${CXXFLAGS} ${OPENMP_CFLAGS}"

#find out what version we are running
ARCH=`uname -m`
if [[ $ARCH == "x86_64" ]];
then
  SUFFIX="64"
else
  SUFFIX=""
fi

# Setup CUDA paths
# ------------------------------------------------------------------------------
CUDA_PATH=/usr/local/cuda
AC_ARG_WITH([cuda],
   [  --with-cuda=PATH    prefix where cuda is installed [default=/usr/local/cuda]])

if test x"$with_cuda" != x"no" ; then

  if test -n "$with_cuda"
  then
     CUDA_PATH=$with_cuda
  fi

  AC_CHECK_PROG(NVCC_CHECK,nvcc,yes,no,$CUDA_PATH/bin)
  if test x"$NVCC_CHECK" = x"yes" ; then
     cuda_enabled=yes
     GENCODE_SM10="-gencode arch=compute_10,code=sm_10"
     GENCODE_SM20="-gencode arch=compute_20,code=sm_20"
     if test `nvcc --version|grep release|awk '{print $5}'|cut -d. -f1` -ge 5 ; then
      GENCODE_SM30="-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35"
     fi
     CUDA_CFLAGS="-I$CUDA_PATH/include -use_fast_math ${GENCODE_SM10} ${GENCODE_SM20} ${GENCODE_SM30} -Xcompiler \"${CXXFLAGS}\""
     CUDA_LIBS="-lcudart -lcublas"
     CUDA_LDFLAGS="-L$CUDA_PATH/lib$SUFFIX"
     NVCC="$CUDA_PATH/bin/nvcc"
     AC_SUBST(CUDA_CFLAGS)
     AC_SUBST(CUDA_LIBS)
     AC_SUBST(CUDA_LDFLAGS)
     AC_SUBST(NVCC)
     AC_DEFINE([CUDA], 1, [cuda enabled])
  elif test -n "$with_cuda" ; then
     echo "---------------------------------------"
     echo "Unable to find CUDA in $with_cuda."
     echo "Building a version without CUDA.    "
     echo "---------------------------------------"
     cuda_enabled=no
  else 
     cuda_enabled=no
  fi
else
  cuda_enabled=no  
fi


#Setup MPI Paths
# ------------------------------------------------------------------------------
AX_MPI_OPTIONS
AX_MPI_TESTS
CXX=$MPI_CXX

AC_CONFIG_FILES([Makefile
                 src/Makefile])
AC_OUTPUT

echo \
"-------------------------------------------------

 ${PACKAGE_NAME} Version ${PACKAGE_VERSION}

 Prefix: '${prefix}'.
 Compiler: '${CXX} ${CXXFLAGS} ${MPI_LIBDIR} ${MPI_LIBS}'

 Package features:
   CUDA enabled: ${cuda_enabled}

 Now type 'make @<:@<target>@:>@'
   where the optional <target> is:
     all                - build all binaries
     install            - install everything

--------------------------------------------------"
