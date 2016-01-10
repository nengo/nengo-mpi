#include "psim_log.hpp"

ParallelSimulationLog::ParallelSimulationLog(
    unsigned n_processors, unsigned processor, vector<ProbeSpec> probe_info, dtype dt, MPI_Comm comm)
:SimulationLog(probe_info, dt), n_processors(n_processors), processor(processor), comm(comm){}

// Master version
void ParallelSimulationLog::prep_for_simulation(string fn, unsigned n_steps){
    filename = fn;

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
    MPI_Bcast(&n_steps, 1, MPI_INT, 0, comm);

    setup_hdf5(n_steps);

    ready_for_simulation = true;
}

// Worker version
void ParallelSimulationLog::prep_for_simulation(){

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
    filename = string(buffer.get());

    // Receive number of steps in simulation
    int n_steps;
    MPI_Bcast(&n_steps, 1, MPI_INT, 0, comm);

    setup_hdf5(n_steps);

    ready_for_simulation = true;
}

void ParallelSimulationLog::setup_hdf5(unsigned n_steps){
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
        hsize_t dset_dims[] = {n_steps, ps.signal_spec.shape1};

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

        if(ps.component % n_processors == processor){
            dset_map[ps.probe_key] = d;
        }

        datasets.push_back(d);
    }

    closed = false;
}

void ParallelSimulationLog::write_file(
        string filename_suffix, unsigned rank, unsigned max_buffer_size, string data){

    string fn = filename.substr(0, filename.find_last_of('.')) + filename_suffix;

    if(data.length() < max_buffer_size){
        data += string(max_buffer_size - data.length(), ' ');
    }

    if(data.length() > max_buffer_size){
        data = data.substr(0, max_buffer_size);
    }

    MPI_File fh;

    char c_filename[fn.length() + 1];
    strcpy(c_filename, fn.c_str());

    MPI_File_open(comm, c_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    MPI_Offset offset = max_buffer_size * rank * sizeof(char);
    MPI_File_set_view(fh, offset, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);

    char c_data[data.length() + 1];
    strcpy(c_data, data.c_str());

    MPI_File_write(fh, c_data, max_buffer_size, MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}
