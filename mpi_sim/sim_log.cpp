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
        pi.name = tokens[5];

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
    hid_t dset_id, dataspace_id, plist_id, att_id, att_dataspace_id;

    // Create a new file collectively and release property list identifier.
    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    hid_t str_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type, MAX_PROBE_NAME_LENGTH);
    H5Tset_strpad(str_type, H5T_STR_NULLTERM);

    for(ProbeInfo pi : probe_info){
        hsize_t dset_dims[] = {num_steps, pi.shape1};

        // Create the dataspace for the dataset.
        dataspace_id = H5Screate_simple(2, dset_dims, NULL);

        string dspace_key = to_string(pi.probe_key);

        // Create the dataset with default properties
        dset_id = H5Dcreate2(
            file_id, dspace_key.c_str(), H5T_NATIVE_DOUBLE, dataspace_id,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        // Set the ``name'' attribute of the dataset so we know which probe the data came from
        att_dataspace_id  = H5Screate(H5S_SCALAR);
        att_id = H5Acreate2(dset_id, "name", str_type, att_dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(att_id, str_type, pi.name.substr(0, MAX_PROBE_NAME_LENGTH).c_str());

        H5Sclose(att_dataspace_id);
        H5Aclose(att_id);

        HDF5Dataset d(pi.name, pi.shape1, dset_id, dataspace_id);

        dset_map[pi.probe_key] = d;
        datasets.push_back(d);
    }

    H5Tclose(str_type);
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
