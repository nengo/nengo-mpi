#ifndef NENGO_MPI_SIMULATION_LOG_HPP
#define NENGO_MPI_SIMULATION_LOG_HPP

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <exception>

#include <hdf5.h>
#include <mpi.h>

#include "operator.hpp"
#include "debug.hpp"

using namespace std;

struct HDF5Dataset{
    HDF5Dataset(){};

    HDF5Dataset(string n, hid_t d, hid_t dspace)
        :parallel(false), dset_id(d), dataspace_id(dspace),
         plist_id(H5P_DEFAULT), name(n), row_offset(0){};

    HDF5Dataset(string n, hid_t d, hid_t dspace, hid_t p)
        :parallel(true), dset_id(d), dataspace_id(dspace),
         plist_id(p), name(n), row_offset(0){};

    bool parallel;

    hid_t dset_id;
    hid_t dataspace_id;
    hid_t plist_id;

    string name;
    int row_offset;
};

struct ProbeInfo{
    int component;
    key_type probe_key;
    string signal_string;
    int period;
    string name;

    int shape1;
    int shape2;
};

// Represents an HDF5 file that can be written-to in parallel.
// If filename given to prep_for_simulation is empty string, no logging is done
class SimulationLog{
public:
    SimulationLog(){};

    // Used when only one component being used
    SimulationLog(vector<string> probe_strings, dtype dt);

    // Called by master
    SimulationLog(int num_components, vector<string> probe_info, dtype dt, MPI_Comm comm);
    void prep_for_simulation(string filename, int num_steps);

    // Called by workers
    SimulationLog(int num_components, int component, dtype dt, MPI_Comm comm);
    void prep_for_simulation();

    bool is_ready(){return ready_for_simulation;};

    void store_probe_info(vector<string> probe_strings);
    void setup_hdf5(string filename, int num_steps);
    void write(key_type probe_key, vector<unique_ptr<BaseSignal>> data);
    void close();

private:
    bool ready_for_simulation;

    int num_components;
    int component;
    dtype dt;
    MPI_Comm comm;

    int mpi_rank;
    int mpi_size;

    //string filename;
    hid_t file_id;

    vector<ProbeInfo> probe_info;

    map<key_type, HDF5Dataset> dset_map;
    vector<HDF5Dataset> datasets;
};

vector<string> bcast_recv_probe_info(MPI_Comm comm);
void bcast_send_probe_info(vector<string> probe_info, MPI_Comm comm);

#endif
