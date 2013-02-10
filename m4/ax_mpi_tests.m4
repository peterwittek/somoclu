dnl MPI tests
dnl
dnl   AX_MPI_TESTS(subpackage)
dnl
dnl     Use "subpackage" option to omit tests and just set
dnl     makefile conditionals and variables.  Also avoids
dnl     repeating MPI paths in CPPFLAGS and LDFLAGS.

AC_DEFUN([AX_MPI_TESTS],[
  dnl
  dnl Test to see if MPI compilers work properly
  dnl 

dnl
dnl Preprocess and compile check
dnl

if test -n "${MPI_DIR}" && test -z "${MPI_INC}"; then
  MPI_INC="${MPI_DIR}/include"
fi

if test -z "${MPI_INC}"; then
  AC_LANG([C++])
  AC_MSG_CHECKING(whether we can preprocess mpi.h)
  AC_PREPROC_IFELSE(
  [AC_LANG_SOURCE([[#include "mpi.h"]])],
  [
    AC_MSG_RESULT(yes)
  ],[
    AC_MSG_RESULT(no)
    echo "---"
    echo "Cannot find header file mpi.h."
    echo "Either compile without mpi, or view the mpi options with \"configure --help\"."
    echo "---"
    AC_MSG_ERROR(cannot find mpi.h)
  ])

  AC_MSG_CHECKING(whether we can compile mpi.h)
  AC_COMPILE_IFELSE(
  [AC_LANG_SOURCE([[#include "mpi.h"]],[[int c; char** v; MPI_Init(&c,&v);]])],
  [
    AC_MSG_RESULT(yes)
    AC_DEFINE(HAVE_MPI,,[define that mpi is being used])
  ],[
    AC_MSG_RESULT(no)
    echo "---"
    echo "mpi.h has compile errors"
    echo "View the mpi options with \"configure --help\", and provide a valid MPI."
    echo "---"
    AC_MSG_ERROR(invalid mpi.h)
  ])
fi

AC_SUBST([MPI_INC])
  
if test -n "${MPI_DIR}" && test -z "${MPI_LIBDIR}"; then
  MPI_LIBDIR="${MPI_DIR}/lib"
fi
AC_SUBST([MPI_LIBDIR])

if test -z "${MPI_LIBS}" && test -n "${MPI_LIBDIR}"; then
  MPI_LIBS="-lmpi"
fi
AC_SUBST([MPI_LIBS])

AC_LANG([C++])

AC_MSG_CHECKING(whether special compile flag for MPICH is required)
AC_RUN_IFELSE(
[AC_LANG_PROGRAM(
       [[#define MPICH_IGNORE_CXX_SEEK]
        [#include <mpi.h>]], 
       [[#ifdef MPICH_NAME
           return 0; 
         #endif
         return 1;]])],
[AC_MSG_RESULT(yes)
 CXXFLAGS="${CXXFLAGS} -DMPICH_IGNORE_CXX_SEEK"
 echo "-----"
 echo "Adding -DMPICH_IGNORE_CXX_SEEK to MPICH compilations"
 echo "-----"],
[AC_MSG_RESULT(no)],
[AC_MSG_RESULT(cross compiling)
 CXXFLAGS="${CXXFLAGS} -DMPICH_IGNORE_CXX_SEEK"
 echo "-----"
 echo "Adding -DMPICH_IGNORE_CXX_SEEK because we can't determine whether or not it is required"
 echo "-----"]
)

])
