#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.61])
AC_INIT([Somoclu], [1.0])
AC_CONFIG_MACRO_DIR([m4])
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
     CUDA_CFLAGS="-I$CUDA_PATH/include -Xcompiler \"${CXXFLAGS}\""
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
with_mpi=yes
AX_PROG_CXX_MPI([test x"$with_mpi" != xno],[
  CXX=mpicxx
  ],[
  AC_MSG_ERROR([No MPI compiler found.])
])
# Checks for libraries.
# Checks for typedefs, structures, and compiler characteristics.
# Checks for library functions.

AC_CONFIG_FILES([Makefile
                 src/Makefile])
AC_OUTPUT

echo \
"-------------------------------------------------

 ${PACKAGE_NAME} Version ${PACKAGE_VERSION}

 Prefix: '${prefix}'.
 Compiler: '${CXX} ${CXXFLAGS}'

 Package features:
   CUDA enabled: ${cuda_enabled}

 Now type 'make @<:@<target>@:>@'
   where the optional <target> is:
     all                - build all binaries
     install            - install everything

--------------------------------------------------"
