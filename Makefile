export CUDA_DIR=/usr/local/cuda
export MPI_DIR=/usr
export INCLUDES=-I$(MPI_DIR)/include -I$(CUDA_DIR)/include
export LIBS=-L $(CUDA_DIR)/lib64 -lcublas -lcudart

# Package-related substitution variables
export package     = somoclu
export version     = 1.0
export tarname     = $(package)
export distdir     = $(tarname)-$(version)

# Prefix-related substitution variables
export prefix      = /usr/local
export exec_prefix = $(prefix)
export bindir      = $(prefix)/bin

# Tool-related substitution variables
export CXX          = mpicxx
export CXXFLAGS     = -g -O2 -Wall

export NVCC=$(CUDA_DIR)/bin/nvcc
export NVCCFLAGS=-arch sm_20 -g --compiler-options -rdynamic

all clean install uninstall somoclu:
	$(MAKE) -C src $@

dist: $(distdir).tar.gz

$(distdir).tar.gz: FORCE $(distdir)
	tar chof - $(distdir) | gzip -9 -c >$(distdir).tar.gz
	rm -rf $(distdir)

$(distdir):
	mkdir -p $(distdir)/src
	cp Makefile $(distdir)
	cp src/Makefile $(distdir)/src
	cp src/*.c* src/*.h $(distdir)/src

distcheck: $(distdir).tar.gz
	gzip -cd $+ | tar xvf -
	$(MAKE) -C $(distdir) all check
	$(MAKE) -C $(distdir) DESTDIR=$${PWD}/$(distdir)/_inst install uninstall
	$(MAKE) -C $(distdir) clean
	rm -rf $(distdir)
	@echo "*** Package $(distdir).tar.gz is ready for distribution."

FORCE:
	-rm -rf $(distdir) &>/dev/null
	-rm $(distdir).tar.gz &>/dev/null

.PHONY: FORCE all clean check dist distcheck install uninstall
