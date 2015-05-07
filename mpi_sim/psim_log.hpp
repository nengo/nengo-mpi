#ifndef NENGO_MPI_PARALLEL_SIMULATION_LOG_HPP
#define NENGO_MPI_PARALLEL_SIMULATION_LOG_HPP

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <exception>

#include <hdf5.h>
#include <mpi.h>

#include "sim_log.hpp"
#include "debug.hpp"

// A parallel version of SimulationLog. Represents an HDF5 file to which we
// write data collected throughout the simulation. All processors have access
// to the same file, and all processors can write to it independently.
class ParallelSimulationLog: public SimulationLog{
public:
    ParallelSimulationLog(){};

    ParallelSimulationLog(
        int n_processors, int processor, vector<string> probe_strings,
        dtype dt, MPI_Comm comm);

    // Called by master
    void prep_for_simulation(string filename, int num_steps);

    // Called by workers
    void prep_for_simulation();

    // Use the `probe_info` (which is read from the HDF5 file that specifies the
    // network), to collectively construct a shared parallel HDF5 which all
    // processors can write simulation results to.
    void setup_hdf5(string filename, int num_steps);

protected:
    int n_processors;
    int processor;
    MPI_Comm comm;

    int mpi_rank;
    int mpi_size;
};

#endif
