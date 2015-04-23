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

class ParallelSimulationLog: public SimulationLog{
public:
    ParallelSimulationLog(){};

    // Called by master
    ParallelSimulationLog(int n_processors, vector<string> probe_info, dtype dt, MPI_Comm comm);
    void prep_for_simulation(string filename, int num_steps);

    // Called by workers
    ParallelSimulationLog(int n_processors, int processor, dtype dt, MPI_Comm comm);
    void prep_for_simulation();

    void setup_hdf5(string filename, int num_steps);

protected:
    int n_processors;
    int processor;
    MPI_Comm comm;

    int mpi_rank;
    int mpi_size;
};


vector<string> bcast_recv_probe_info(MPI_Comm comm);
void bcast_send_probe_info(vector<string> probe_info, MPI_Comm comm);

#endif
