OBJS=sparseCpuKernels.o somoclu.o io.o denseCpuKernels.o denseGpuKernels.cu.co training.o

all: somoclu

somoclu: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
		$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

%.cu.co: %.cu
		$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<

clean:
	-rm somoclu $(OBJS) &>/dev/null

install:
	install -d $(DESTDIR)$(bindir)
	install -m 0755 somoclu $(DESTDIR)$(bindir)

uninstall:
	-rm $(DESTDIR)$(bindir)/somoclu &>/dev/null

.PHONY: all clean install uninstall
