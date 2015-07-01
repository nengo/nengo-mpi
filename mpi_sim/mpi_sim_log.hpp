#ifndef NENGO_MPI_PARALLEL_SIMULATION_LOG_HPP
#define NENGO_MPI_PARALLEL_SIMULATION_LOG_HPP

#include <mpi.h>

#include "sim_log.hpp"

// A parallel version of SimulationLog. Represents an HDF5 file to which we
// write data collected throughout the simulation. All processors have access
// to the same file, and all processors can write to it independently.
class MpiSimulationLog: public SimulationLog{
public:
    MpiSimulationLog(){};

    MpiSimulationLog(
        int n_processors, int rank, vector<ProbeSpec> probe_info,
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
    int rank;
    MPI_Comm comm;

    int mpi_rank;
    int mpi_size;
};

#endif
