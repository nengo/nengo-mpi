#ifndef NENGO_MPI_SIMULATION_LOG_HPP
#define NENGO_MPI_SIMULATION_LOG_HPP

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <exception>

#include <hdf5.h>

#include "operator.hpp"
#include "debug.hpp"

using namespace std;

struct HDF5Dataset{
    HDF5Dataset(){};

    HDF5Dataset(string n, int n_cols, hid_t d, hid_t dspace)
        :parallel(false), name(n), n_cols(n_cols), dset_id(d),
         dataspace_id(dspace), plist_id(H5P_DEFAULT), row_offset(0){};

    HDF5Dataset(string n, int n_cols, hid_t d, hid_t dspace, hid_t p)
        :parallel(true), name(n), n_cols(n_cols), dset_id(d),
         dataspace_id(dspace), plist_id(p), row_offset(0){};

    void close(){
        H5Dclose(dset_id);
        H5Sclose(dataspace_id);

        if(parallel){
            H5Pclose(plist_id);
        }
    }

    bool parallel;

    string name;

    int n_cols;

    hid_t dset_id;
    hid_t dataspace_id;
    hid_t plist_id;

    int row_offset;
};

const int MAX_PROBE_NAME_LENGTH = 512;

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

    SimulationLog(vector<string> probe_strings, dtype dt);
    SimulationLog(dtype dt);

    virtual void prep_for_simulation(string filename, int num_steps);
    virtual void prep_for_simulation();

    bool is_ready(){return ready_for_simulation;};

    void store_probe_info(vector<string> probe_strings);
    virtual void setup_hdf5(string filename, int num_steps);
    void write(key_type probe_key, shared_ptr<dtype> buffer, int n_rows);
    void close();

    bool is_closed(){return closed;};

protected:
    bool ready_for_simulation;

    dtype dt;

    hid_t file_id;

    vector<ProbeInfo> probe_info;

    map<key_type, HDF5Dataset> dset_map;
    vector<HDF5Dataset> datasets;
    bool closed;
};

#endif
