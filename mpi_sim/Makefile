#Compiler and Linker
CC          := mpicxx.openmpi

#The Directories, Source, Includes, Objects, Binary and Resources
SRCDIR      := .
INCDIR      := include
BUILDDIR    := obj
TARGETDIR   := bin
SRCEXT      := cpp
DEPEXT      := d
OBJEXT      := o
$(shell echo $(SRCDIR) > temp.txt)

DEFS        :=
STD         :=c++11
NENGO_CPP_LIBS=-lcblas -lhdf5_open -lboost -ldl -lm
NENGO_MPILIBS=-lcblas -lhdf5_open -lboost -ldl -lm
MPI_SIM_SO_LIBS=-lcblas -lhdf5_open -lboost -ldl -lm

#Flags, Libraries and Includes
CFLAGS      := -std=c++11 -fPIC
LIB         := -ldl -lm -lcblas -lhdf5_openmpi
INC         := -I$(INCDIR) -I/usr/local/include -I/usr/include/hdf5/openmpi
INCDEP      := -I$(INCDIR)

#OBJECTS=signal.o operator.o simulator.o spec.o spaun.o probe.o chunk.o sim_log.o debug.o utils.o
#MPI_OBJECTS=$(OBJECTS) mpi_simulator.o mpi_operator.o psim_log.o

#---------------------------------------------------------------------------------
#DO NOT EDIT BELOW THIS LINE
#---------------------------------------------------------------------------------
SOURCES     := $(shell find $(SRCDIR) -type f -name "*.$(SRCEXT)")
OBJECTS     := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.$(OBJEXT)))

#Defauilt Make
all: directories nengo_cpp nengo_mpi mpi_sim.so

#Remake
remake: cleaner all

#Make the Directories
directories:
	mkdir -p $(TARGETDIR)
	mkdir -p $(BUILDDIR)

#Clean only Objecst
clean:
	@$(RM) -rf $(BUILDDIR)

#Full Clean, Objects and Binaries
cleaner: clean
	@$(RM) -rf $(TARGETDIR)

#Pull in dependency info for *existing* .o files
-include $(OBJECTS:.$(OBJEXT)=.$(DEPEXT))

nengo_cpp: $(OBJECTS)
	$(CC) -o $(TARGETDIR)/nengo_cpp $^ $(DEFS) -std=$(STD) $(NENGO_CPP_LIBS)

nengo_mpi: $(OBJECTS)
	$(CC) -o $(TARGETDIR)/nengo_mpi $^ $(DEFS) -std=$(STD) $(NENGO_MPI_LIBS)

mpi_sim.so: $(OBJECTS)
	$(CC) -o $(TARGETDIR)/mpi_sim.so $^ -shared $(DEFS) -std=$(STD) $(MPI_SIM_SO_LIBS)

#Compile
$(BUILDDIR)/%.$(OBJEXT): $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<
	@$(CC) $(CFLAGS) $(INCDEP) -MM $(SRCDIR)/$*.$(SRCEXT) > $(BUILDDIR)/$*.$(DEPEXT)
	@cp -f $(BUILDDIR)/$*.$(DEPEXT) $(BUILDDIR)/$*.$(DEPEXT).tmp
	@sed -e 's|.*:|$(BUILDDIR)/$*.$(OBJEXT):|' < $(BUILDDIR)/$*.$(DEPEXT).tmp > $(BUILDDIR)/$*.$(DEPEXT)
	@sed -e 's/.*://' -e 's/\\$$//' < $(BUILDDIR)/$*.$(DEPEXT).tmp | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >> $(BUILDDIR)/$*.$(DEPEXT)
	@rm -f $(BUILDDIR)/$*.$(DEPEXT).tmp

#Non-File Targets
.PHONY: all remake clean cleaner resources
