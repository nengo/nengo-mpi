#include "sim_log.hpp"

SimulationLog::SimulationLog(vector<string> probe_strings, dtype dt)
:num_components(1), component(0), dt(dt), ready_for_simulation(false){
    cout << "using serial constructor" << endl;

    cout << "Master probe info:" << endl;
    for(auto s : probe_strings){
        cout << s << endl;
    }

    store_probe_info(probe_strings);
}

SimulationLog::SimulationLog(
    int num_components, vector<string> probe_strings, dtype dt, MPI_Comm comm)
:num_components(num_components), component(0),
 dt(dt), comm(comm), ready_for_simulation(false){

    cout << "using nonserial constructor" << num_components << endl;
    if(num_components > 1){
        bcast_send_probe_info(probe_strings, comm);
    }

    cout << "Master probe info:" << endl;
    for(auto s : probe_strings){
        cout << s << endl;
    }

    store_probe_info(probe_strings);
}

SimulationLog::SimulationLog(
    int num_components, int component, dtype dt, MPI_Comm comm)
:num_components(num_components), component(component), dt(dt),
 comm(comm), ready_for_simulation(false){

    vector<string> probe_strings = bcast_recv_probe_info(comm);

    cout << "Worker " << component << " probe info:" << endl;
    for(auto s : probe_strings){
        cout << s << endl;
    }

    store_probe_info(probe_strings);
}

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

        cout << "shape1 : " << pi.shape1 << ", shape2: " << pi.shape2 << endl;

        probe_info.push_back(pi);
    }
}

// Master version
void SimulationLog::prep_for_simulation(string filename, int num_steps){
    cout  << "IN prep_for " << num_components << endl;
    if(num_components > 1){
        int size = filename.size();
        MPI_Bcast(&size, 1, MPI_INT, 0, comm);

        unique_ptr<char[]> buffer(new char[size+1]);
        strcpy(buffer.get(), filename.c_str());
        MPI_Bcast(buffer.get(), size+1, MPI_CHAR, 0, comm);

        if(filename.compare("") == 0){
            ready_for_simulation = false;
            return;
        }

        MPI_Bcast(&num_steps, 1, MPI_INT, 0, comm);
    }

    if(filename.compare("") == 0){
        ready_for_simulation = false;
        return;
    }

    cout  << "IN prep_for " << filename << endl;
    setup_hdf5(filename, num_steps);

    ready_for_simulation = true;
}

// Worker version
void SimulationLog::prep_for_simulation(){

    if(num_components == 1){
        throw logic_error(
            "Calling Worker constructor for SimulationLog, but only one component");
    }

    int size;
    MPI_Bcast(&size, 1, MPI_INT, 0, comm);

    unique_ptr<char[]> buffer(new char[size+1]);
    MPI_Bcast(buffer.get(), size+1, MPI_CHAR, 0, comm);
    string filename(buffer.get());

    if(filename.compare("") == 0){
        ready_for_simulation = false;
        return;
    }

    int num_steps;
    MPI_Bcast(&num_steps, 1, MPI_INT, 0, comm);

    setup_hdf5(filename, num_steps);

    ready_for_simulation = true;
}

void SimulationLog::setup_hdf5(string filename, int num_steps){
    hid_t dset_id, dataspace_id, plist_id;

    cout << "in hdf5 " << filename << endl;

    if(num_components > 1){
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
            dset_id = H5Dcreate2(
                file_id, pi.name.c_str(), H5T_NATIVE_DOUBLE, dataspace_id,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

            // Create property list for independent dataset write.
            plist_id = H5Pcreate(H5P_DATASET_XFER);
            H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);

            HDF5Dataset d(pi.name, dset_id, dataspace_id, plist_id);

            if(pi.component == component){
                dset_map[pi.probe_key] = d;
            }

            datasets.push_back(d);
        }
    }else{
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

            HDF5Dataset d(pi.name, dset_id, dataspace_id);

            if(pi.component == component){
                dset_map[pi.probe_key] = d;
            }

            datasets.push_back(d);
        }
    }
    cout  << "IN setup_hdf5" << endl;
}

void SimulationLog::write(key_type probe_key, vector<unique_ptr<BaseSignal>> data){
    herr_t status;

    int num_rows = data.size();
    int num_cols = data[0]->size1();

    // TODO: calculate size
    int buffer_size = num_rows * num_cols;

    cout << "in write: " << num_rows << ", " << num_cols << endl;

    unique_ptr<dtype[]> buffer(new dtype[buffer_size]);

    int i = 0;

    // For now, assume everything we're probing is a column vector
    // We lay it out in the buffer as a row.
    for(auto& sig: data){
        for(int j = 0; j < num_cols; j++){
            buffer[i] = (*sig)(j, 0);
            i++;
        }
    }

    HDF5Dataset d = dset_map.at(probe_key);
    cout << "in write offset: " << d.row_offset << endl;

    hsize_t     count[] = {num_rows, num_cols};
    hsize_t     offset[] = {d.row_offset, 0};
    hsize_t     stride[] = {1, 1};
    hsize_t     block[] = {1, 1};

    hid_t memspace_id = H5Screate_simple(2, count, NULL);

    status = H5Sselect_hyperslab(
        d.dataspace_id, H5S_SELECT_SET, offset, stride, count, block);

    status = H5Dwrite(
        d.dset_id, H5T_NATIVE_DOUBLE, memspace_id, d.dataspace_id,
        d.plist_id, buffer.get());

    d.row_offset += num_rows;
}

void SimulationLog::close(){
    for(auto d : datasets){
        H5Dclose(d.dset_id);
        H5Sclose(d.dataspace_id);

        if(d.parallel){
            H5Pclose(d.plist_id);
        }
    }

    H5Fclose(file_id);
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
