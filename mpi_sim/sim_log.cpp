#include "sim_log.hpp"


SimulationLog::SimulationLog(vector<string> probe_strings, dtype dt)
:dt(dt), ready_for_simulation(false), closed(true){
    store_probe_info(probe_strings);
}

SimulationLog::SimulationLog(dtype dt)
:dt(dt), ready_for_simulation(false), closed(true){}

void SimulationLog::store_probe_info(vector<string> probe_strings){
    for(string s : probe_strings){
        // Extract probe data from the string
        vector<string> tokens;
        boost::split(tokens, s, boost::is_any_of("|"), boost::token_compress_on);

        ProbeInfo pi;

        pi.component = boost::lexical_cast<int>(tokens[1]);
        pi.probe_key = boost::lexical_cast<key_type>(tokens[2]);
        pi.signal_string = tokens[3];
        pi.period = boost::lexical_cast<int>(tokens[4]);

        // TODO: get the actual name of the probe
        pi.name = tokens[2];

        tokens.clear();
        boost::split(tokens, pi.signal_string, boost::is_any_of(":"), boost::token_compress_on);

        key_type key = boost::lexical_cast<key_type>(tokens[0]);

        vector<string> shape_tokens;
        boost::trim_if(tokens[1], boost::is_any_of("(,)"));
        boost::split(shape_tokens, tokens[1], boost::is_any_of(","));

        pi.shape1 = boost::lexical_cast<int>(shape_tokens[0]);
        pi.shape2 = shape_tokens.size() == 1 ? 1 : boost::lexical_cast<int>(shape_tokens[1]);

        probe_info.push_back(pi);
    }
}

void SimulationLog::prep_for_simulation(string filename, int num_steps){
    if(filename.size() == 0){
        ready_for_simulation = false;
        return;
    }

    setup_hdf5(filename, num_steps);

    ready_for_simulation = true;
}

void SimulationLog::prep_for_simulation(){
    throw logic_error(
        "Calling prep_for_simulation with no arguments on non-parallel simulation log.");
}

void SimulationLog::setup_hdf5(string filename, int num_steps){
    hid_t dset_id, dataspace_id, plist_id;

    // Create a new file collectively and release property list identifier.
    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    for(ProbeInfo pi : probe_info){
        hsize_t dset_dims[] = {num_steps, pi.shape1};

        // Create the dataspace for the dataset.
        dataspace_id = H5Screate_simple(2, dset_dims, NULL);

        // Create the dataset with default properties
        dset_id = H5Dcreate2(
            file_id, pi.name.c_str(), H5T_NATIVE_DOUBLE, dataspace_id,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        HDF5Dataset d(pi.name, pi.shape1, dset_id, dataspace_id);

        dset_map[pi.probe_key] = d;
        datasets.push_back(d);
    }
}

void SimulationLog::write(key_type probe_key, shared_ptr<dtype> buffer, int n_rows){
    herr_t status;

    HDF5Dataset& d = dset_map.at(probe_key);

    int n_cols = d.n_cols;

    hsize_t     count[] = {n_rows, n_cols};
    hsize_t     offset[] = {d.row_offset, 0};
    hsize_t     stride[] = {1, 1};
    hsize_t     block[] = {1, 1};

    hid_t memspace_id = H5Screate_simple(2, count, NULL);

    status = H5Sselect_hyperslab(
        d.dataspace_id, H5S_SELECT_SET, offset, stride, count, block);

    status = H5Dwrite(
        d.dset_id, H5T_NATIVE_DOUBLE, memspace_id, d.dataspace_id,
        d.plist_id, buffer.get());

    H5Sclose(memspace_id);

    d.row_offset += n_rows;
}

void SimulationLog::close(){
    if(!closed){
        for(auto& d : datasets){
            d.close();
        }

        H5Fclose(file_id);

        closed = true;
        ready_for_simulation = false;
    }
}

// Master version
ParallelSimulationLog::ParallelSimulationLog(
    int num_components, vector<string> probe_strings, dtype dt, MPI_Comm comm)
:SimulationLog(probe_strings, dt), num_components(num_components), component(0), comm(comm){

    if(num_components == 1){
        throw logic_error(
            "Trying to construct ParallelSimulationLog, but only one component.");
    }

    if(num_components > 1){
        bcast_send_probe_info(probe_strings, comm);
    }
}

// Worker version
ParallelSimulationLog::ParallelSimulationLog(
    int num_components, int component, dtype dt, MPI_Comm comm)
:SimulationLog(dt), num_components(num_components), component(component), comm(comm){

    if(num_components == 1){
        throw logic_error(
            "Trying to construct ParallelSimulationLog, but only one component.");
    }

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
    hid_t dset_id, dataspace_id, plist_id;

    // Set up file access property list with parallel I/O access
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);

    // Create a new file collectively and release property list identifier.
    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);

    for(ProbeInfo pi : probe_info){
        hsize_t dset_dims[] = {num_steps, pi.shape1};

        // Create the dataspace for the dataset.
        dataspace_id = H5Screate_simple(2, dset_dims, NULL);

        // Create the dataset with default properties
        dset_id = H5Dcreate(
            file_id, pi.name.c_str(), H5T_NATIVE_DOUBLE, dataspace_id,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

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
