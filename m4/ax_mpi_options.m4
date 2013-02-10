AC_DEFUN([AX_MPI_OPTIONS],[
  dnl
  dnl Command-line options for setting up MPI
  dnl 

MPI_CXX=none

###

AC_ARG_WITH(mpi-compilers,
AS_HELP_STRING([--with-mpi-compilers=DIR or --with-mpi-compilers=yes],
[use MPI compiler (mpicxx) found in directory DIR, or in your PATH if =yes]),
[
  if test "X${withval}" = "Xyes"; then
    # Look for MPI C++ compile script/program in user's path, set
    # MPI_CXX to name of program found, else set it to "none".  Could
    # add more paths to check as 4th argument if you wanted to.
    # Calls AC_SUBST for MPI_CXX, etc.

    if test "X${MPI_CXX}" = "Xnone" ; then 
      echo "Looking for an MPI C++ compiler in your path"
      AC_CHECK_PROGS([MY_CXX], [mpic++ mpicxx mpiCC mpiicpc], [none],)
      MPI_CXX=${MY_CXX}
    fi

  else
    foundCompiler=no
    if test "X${MPI_CXX}" = "Xnone" ; then
      AC_MSG_CHECKING(MPI C++ compiler in ${withval})
      if test -f ${withval}/mpic++ ; then
        MPI_CXX=${withval}/mpic++
      fi
      if test "X${MPI_CXX}" = "Xnone" && test -f ${withval}/mpicxx ; then
        MPI_CXX=${withval}/mpicxx
      fi
      if test "X${MPI_CXX}" = "Xnone" && test -f ${withval}/mpiCC ; then
        MPI_CXX=${withval}/mpiCC
      fi
      if test "X${MPI_CXX}" = "Xnone" && test -f ${withval}/mpiicpc ; then
        MPI_CXX=${withval}/mpiicpc
      fi
      AC_MSG_RESULT([${MPI_CXX}])
    fi
  fi

  if test "X${MPI_CXX}" = "Xnone" ; then
    AC_MSG_ERROR([MPI C++ compiler script/program not found.])
  fi

]
)

if test "X${MPI_CXX}" = "Xnone" ; then 
  echo "Looking for an MPI C++ compiler in your path"
  AC_CHECK_PROGS([MY_CXX], [mpic++ mpicxx mpiCC mpiicpc], [none],)
  MPI_CXX=${MY_CXX}
fi

if test "X${MPI_CXX}" = "Xnone" ; then
    AC_MSG_ERROR([No C++ compiler: try --with-mpi-compilers= or --with-mpi-cxx=])
fi


AC_ARG_WITH(mpi,
AS_HELP_STRING([--with-mpi=MPIROOT],[use MPI root directory.]),
[
  if test "X${withval}" = "Xno"; then :; else
     MPI_DIR=${withval}
     AC_MSG_CHECKING(MPI directory)
     AC_MSG_RESULT([${MPI_DIR}])

  fi
]
)

AC_ARG_WITH(mpi-libs,
AS_HELP_STRING([--with-mpi-libs="LIBS"],[MPI libraries @<:@default "-lmpi"@:>@]),
[
  if test "X${withval}" = "Xno"; then :; else
    MPI_LIBS=${withval}
    AC_MSG_CHECKING(user-defined MPI libraries)
    AC_MSG_RESULT([${MPI_LIBS}])

  fi
]
)

AC_ARG_WITH(mpi-incdir,
AS_HELP_STRING([--with-mpi-incdir=DIR],[MPI include directory @<:@default MPIROOT/include@:>@]),
[
  if test "X${withval}" = "Xno"; then :; else
    MPI_INC=${withval}
    AC_MSG_CHECKING(user-defined MPI includes)
    AC_MSG_RESULT([${MPI_INC}])
  fi
]
)

AC_ARG_WITH(mpi-libdir,
AS_HELP_STRING([--with-mpi-libdir=DIR],[MPI library directory @<:@default MPIROOT/lib@:>@]),
[
  if test "X${withval}" = "Xno"; then :; else
    MPI_LIBDIR=${withval}
    AC_MSG_CHECKING(user-defined MPI library directory)
    AC_MSG_RESULT([${MPI_LIBDIR}])

  fi
]
)

])
