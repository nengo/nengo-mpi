#include "mpi_sim_log.hpp"

MpiSimulationLog::MpiSimulationLog(
    int n_processors, int rank, vector<ProbeSpec> probe_info, dtype dt, MPI_Comm comm)
:SimulationLog(probe_info, dt), n_processors(n_processors), rank(rank), comm(comm){}

// Master version
void MpiSimulationLog::prep_for_simulation(string filename, int num_steps){

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
void MpiSimulationLog::prep_for_simulation(){

    // Receive filename size
    int size;
    MPI_Bcast(&size, 1, MPI_INT, 0, comm);

    if(size == 0){
        ready_for_simulation = false;
        return;
    }

    // Receive filename
    unique_ptr<char[]> buffer(new char[size+1]);
    MPI_Bcast(buffer.get(), size+1, MPI_CHAR, 0, comm);
    string filename(buffer.get());

    // Receive number of steps in simulation
    int num_steps;
    MPI_Bcast(&num_steps, 1, MPI_INT, 0, comm);

    setup_hdf5(filename, num_steps);

    ready_for_simulation = true;
}

void MpiSimulationLog::setup_hdf5(string filename, int num_steps){
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

    for(ProbeSpec ps : probe_info){
        hsize_t dset_dims[] = {num_steps, ps.signal_spec.shape1};

        // Create the dataspace for the dataset.
        dataspace_id = H5Screate_simple(2, dset_dims, NULL);

        string dspace_key = to_string(ps.probe_key);

        // Create the dataset with default properties
        dset_id = H5Dcreate(
            file_id, dspace_key.c_str(), H5T_NATIVE_DOUBLE, dataspace_id,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        // Set the ``name'' attribute of the dataset so we know which probe the data came from
        att_dataspace_id  = H5Screate(H5S_SCALAR);
        att_id = H5Acreate2(dset_id, "name", str_type, att_dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(att_id, str_type, ps.name.substr(0, MAX_PROBE_NAME_LENGTH).c_str());

        H5Sclose(att_dataspace_id);
        H5Aclose(att_id);

        // Create property list for independent dataset write.
        plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);

        HDF5Dataset d(ps.name, ps.signal_spec.shape1, dset_id, dataspace_id, plist_id);

        if(ps.component % n_processors == rank){
            dset_map[ps.probe_key] = d;
        }

        datasets.push_back(d);
    }

    closed = false;
}
