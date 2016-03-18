#pragma once

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <exception>

#include <hdf5.h>
#include <mpi.h>

#include "sim_log.hpp"
#include "spec.hpp"

#include "typedef.hpp"
#include "debug.hpp"


// A parallel version of SimulationLog. Represents an HDF5 file to which we
// write data collected throughout the simulation. All processors have access
// to the same file, and all processors can write to it independently.
class ParallelSimulationLog: public SimulationLog{
public:
    ParallelSimulationLog(){};

    ParallelSimulationLog(
        unsigned n_processors, unsigned processor,
        vector<ProbeSpec> probe_info, dtype dt, MPI_Comm comm);

    // Called by master
    void prep_for_simulation(string fn, unsigned n_steps);

    // Called by workers
    void prep_for_simulation();

    // Use the `probe_info` (which is read from the HDF5 file that specifies the
    // network), to collectively construct a shared parallel HDF5 which all
    // processors can write simulation results to.
    void setup_hdf5(unsigned n_steps);

    virtual void write_file(string filename_suffix, unsigned rank, unsigned max_buffer_size, string data);

protected:
    unsigned n_processors;
    unsigned processor;
    MPI_Comm comm;

    unsigned mpi_rank;
    unsigned mpi_size;
};
