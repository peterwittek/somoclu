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
  AC_CHECK_PROGS([MY_CXX], [mpic++ mpicxx mpiCC mpiicpc], [none],)
  MPI_CXX=${MY_CXX}
fi



# if the user gives configure the option --with-mpi=directory then set MPI_DIR
AC_ARG_WITH(mpi,
AS_HELP_STRING([--with-mpi=MPIROOT],[use MPI root directory.]),
[
	if test x"$with_mpi" != x"yes"; then
	  MPI_DIR=${with_mpi}
      AC_MSG_CHECKING(MPI directory)
      AC_MSG_RESULT([${MPI_DIR}])
    fi
]
)

#otherwise try to find it
if test -z "${MPI_DIR}";	then
	# set MPI_DIR to the correct directory
	# otherwise send error
	
	AC_MSG_CHECKING([for MPI directory])
	
	
	pathlibs="$(echo $LD_LIBRARY_PATH|sed -e 's/:/ /g')"
	counter=1
	end=no
  CANDIDATES="/usr"
	until [test x"$end" = x"yes"]
	do
		pathlib="$(echo $pathlibs | awk -v awk_var=$counter '{print $awk_var}' )" 
		if test -n "$pathlib"; then
		  match="$(echo ${pathlib:(-4)})"
		  if test x"$match" = x"/lib"; then
			index="$(echo ${#pathlib})"
			index=$(($index-4))
			search_path="$(echo ${pathlib:0:$index})"
			CANDIDATES="$CANDIDATES $search_path"
		  fi
		  match="$(echo ${pathlib:(-6)})"
		  if test x"$match" = x"/lib64"; then
			index="$(echo ${#pathlib})"
			index=$(($index-6))
			search_path="$(echo ${pathlib:0:$index})"
			CANDIDATES="$CANDIDATES $search_path"
		  fi
		else
		  end=yes
		fi
		counter=$(($counter+1))
	done
	
	#find what path has the openmpi directory
	counter=1
	DIR=no
	until [test -z "$DIR"]
	do
		DIR="$(echo $CANDIDATES | awk -v awk_var=$counter '{print $awk_var}' )"
		if test -n "$DIR"; then
		  DIR_COPY=$DIR
		  until [test -z "$DIR_COPY" || test x"$FIND" = x"yes"]
		  do
			NUM="$(expr match "$DIR_COPY" 'openmpi')"
			if test x"$NUM" = x"7"; then
			  FIND=yes
			  MPI_DIR=$DIR
			  AC_MSG_RESULT([${MPI_DIR}])
			  DIR=
			fi
			DIR_COPY="$(echo ${DIR_COPY:1})"
		  done
		fi
		counter=$(($counter+1))
	done
	
	if test -z "${MPI_DIR}";	then	
			AC_MSG_RESULT([${MPI_DIR}])
	fi
fi


AC_ARG_WITH(mpi-incdir,
AS_HELP_STRING([--with-mpi-incdir=DIR],[MPI include directory @<:@default MPIROOT/include@:>@]),
[
  if test "X${withval}" != "Xno" && test "X${withval}" != "Xyes"; then
    MPI_INC="-I${withval}"
    AC_MSG_CHECKING(user-defined MPI includes)
    AC_MSG_RESULT([${MPI_INC}])
  fi
]
)

# set MPI_INC to the default directory if not already specified by the user
if test -z "${MPI_INC}"; then
  MPI_INC="-I${MPI_DIR}/include"
fi

AC_ARG_WITH(mpi-libdir,
AS_HELP_STRING([--with-mpi-libdir=DIR],[MPI library directory @<:@default MPIROOT/lib@:>@]),
[
  if test "X${withval}" != "Xno" && test "X${withval}" != "Xyes"; then
    MPI_LIBDIR="-L${withval}"
    AC_MSG_CHECKING(user-defined MPI library directory)
    AC_MSG_RESULT([${MPI_LIBDIR}])

  fi
]
)

if test -z "${MPI_LIBDIR}"; then
  MPI_LIBDIR="-L${MPI_DIR}/lib"
fi

AC_ARG_WITH(mpi-libs,
AS_HELP_STRING([--with-mpi-libs="LIBS"],[MPI libraries @<:@default "-lmpi"@:>@]),
[
  if test "X${withval}" != "Xno" && test "X${withval}" != "Xyes"; then
    MPI_LIBS=${withval}
    AC_MSG_CHECKING(user-defined MPI libraries)
    AC_MSG_RESULT([${MPI_LIBS}])
  fi
]
)

if test -n "${MPI_LIBDIR}"; then
  MPI_LIBS="-lmpi"
fi

])
