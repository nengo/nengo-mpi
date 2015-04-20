#include "psim_log.hpp"


// Master version
ParallelSimulationLog::ParallelSimulationLog(
    int n_processors, vector<string> probe_strings, dtype dt, MPI_Comm comm)
:SimulationLog(probe_strings, dt), n_processors(n_processors), component(0), comm(comm){

    if(n_processors > 1){
        bcast_send_probe_info(probe_strings, comm);
    }
}

// Worker version
ParallelSimulationLog::ParallelSimulationLog(
    int n_processors, int component, dtype dt, MPI_Comm comm)
:SimulationLog(dt), n_processors(n_processors), component(component), comm(comm){

    vector<string> probe_strings = bcast_recv_probe_info(comm);
    store_probe_info(probe_strings);
}

// Master version
void ParallelSimulationLog::prep_for_simulation(string filename, int num_steps){

    // Send filename size
    int size = filename.size();
    MPI_Bcast(&size, 1, MPI_INT, 0, comm);

    if(size == 0){
        ready_for_simulation = false;
        return;
    }

    // Send filename
    unique_ptr<char[]> buffer(new char[size+1]);
    strcpy(buffer.get(), filename.c_str());
    MPI_Bcast(buffer.get(), size+1, MPI_CHAR, 0, comm);

    // Send number of steps in simulation
    MPI_Bcast(&num_steps, 1, MPI_INT, 0, comm);

    setup_hdf5(filename, num_steps);

    ready_for_simulation = true;
}

// Worker version
void ParallelSimulationLog::prep_for_simulation(){

    // Get filename size
    int size;
    MPI_Bcast(&size, 1, MPI_INT, 0, comm);

    if(size == 0){
        ready_for_simulation = false;
        return;
    }

    // Get filename
    unique_ptr<char[]> buffer(new char[size+1]);
    MPI_Bcast(buffer.get(), size+1, MPI_CHAR, 0, comm);
    string filename(buffer.get());

    // Get number of steps in simulation
    int num_steps;
    MPI_Bcast(&num_steps, 1, MPI_INT, 0, comm);

    setup_hdf5(filename, num_steps);

    ready_for_simulation = true;
}

void ParallelSimulationLog::setup_hdf5(string filename, int num_steps){
    hid_t dset_id, dataspace_id, plist_id, att_id, att_dataspace_id;

    // Set up file access property list with parallel I/O access
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);

    // Create a new file collectively and release property list identifier.
    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);

    hid_t str_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type, MAX_PROBE_NAME_LENGTH);
    H5Tset_strpad(str_type, H5T_STR_NULLTERM);

    for(ProbeInfo pi : probe_info){
        hsize_t dset_dims[] = {num_steps, pi.shape1};

        // Create the dataspace for the dataset.
        dataspace_id = H5Screate_simple(2, dset_dims, NULL);

        string dspace_key = to_string(pi.probe_key);

        // Create the dataset with default properties
        dset_id = H5Dcreate(
            file_id, dspace_key.c_str(), H5T_NATIVE_DOUBLE, dataspace_id,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        // Set the ``name'' attribute of the dataset so we know which probe the data came from
        att_dataspace_id  = H5Screate(H5S_SCALAR);
        att_id = H5Acreate2(dset_id, "name", str_type, att_dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(att_id, str_type, pi.name.substr(0, MAX_PROBE_NAME_LENGTH).c_str());

        H5Sclose(att_dataspace_id);
        H5Aclose(att_id);

        // Create property list for independent dataset write.
        plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);

        HDF5Dataset d(pi.name, pi.shape1, dset_id, dataspace_id, plist_id);

        if(pi.component == component){
            dset_map[pi.probe_key] = d;
        }

        datasets.push_back(d);
    }

    closed = false;
}

vector<string> bcast_recv_probe_info(MPI_Comm comm){
    int src = 0;

    int num_probes, size;
    MPI_Bcast(&num_probes, 1, MPI_INT, src, comm);

    vector<string> probe_info;

    for(int i = 0; i < num_probes; i++){
        MPI_Bcast(&size, 1, MPI_INT, src, comm);

        unique_ptr<char[]> buffer(new char[size+1]);

        MPI_Bcast(buffer.get(), size+1, MPI_CHAR, src, comm);

        string s(buffer.get());

        probe_info.push_back(s);
    }

    return probe_info;
}

void bcast_send_probe_info(vector<string> probe_info, MPI_Comm comm){
    int src = 0;

    int num_probes = probe_info.size();
    MPI_Bcast(&num_probes, 1, MPI_INT, src, comm);

    for(auto& s : probe_info){
        int size = s.size();
        MPI_Bcast(&size, 1, MPI_INT, src, comm);

        unique_ptr<char[]> buffer(new char[size+1]);

        strcpy(buffer.get(), s.c_str());

        MPI_Bcast(buffer.get(), size+1, MPI_CHAR, src, comm);
    }
}
