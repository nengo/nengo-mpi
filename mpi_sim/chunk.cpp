#include "chunk.hpp"

// in bytes, for each process.
#define MAX_RUNTIME_OUTPUT_SIZE 5000

MpiSimulatorChunk::MpiSimulatorChunk(bool collect_timings)
:time(0.0), dt(0.001), n_steps(0), rank(0), n_processors(1),
mpi_merged(false), collect_timings(collect_timings){
}

MpiSimulatorChunk::MpiSimulatorChunk(int rank, int n_processors, bool mpi_merged, bool collect_timings)
:time(0.0), dt(0.001), n_steps(0), rank(rank), n_processors(n_processors),
mpi_merged(mpi_merged), collect_timings(collect_timings){
    stringstream ss;
    ss << "Chunk " << rank;
    label = ss.str();
}

void MpiSimulatorChunk::from_file(string filename, hid_t file_plist, hid_t read_plist){
    herr_t err;
    hsize_t shape[2];
    hid_t dspace, attr;
    int ndim;
    char* str_ptr;

    hid_t str_type = H5Tcopy(H5T_C_S1);
    H5Tset_strpad(str_type, H5T_STR_NULLPAD);

    hid_t f = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, file_plist);

    // Get n_components
    int n_components;
    attr = H5Aopen(f, "n_components", H5P_DEFAULT);
    H5Aread(attr, H5T_NATIVE_INT, &n_components);
    H5Aclose(attr);

    if(rank == 0){
        cout << "Loading nengo network from file." << endl;
        cout << "Network has " << n_components << " components." << endl;
    }

    // Get dt
    attr = H5Aopen(f, "dt", H5P_DEFAULT);
    H5Aread(attr, H5T_NATIVE_DOUBLE, &dt);
    H5Aclose(attr);

    int component = rank;
    while(component < n_components){

        stringstream ss;
        ss << component;

        // Open the group assigned to my component
        hid_t component_group = H5Gopen(f, ss.str().c_str(), H5P_DEFAULT);

        // signal keys
        hid_t signal_keys = H5Dopen(component_group, "signal_keys", H5P_DEFAULT);

        dspace = H5Dget_space(signal_keys);
        ndim = H5Sget_simple_extent_dims(dspace, shape, NULL);
        H5Sclose(dspace);

        assert(ndim == 1);

        hsize_t n_signals = shape[0];

        auto signal_keys_buffer = unique_ptr<key_type[]>(new key_type[n_signals]);
        err = H5Dread(
            signal_keys, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL,
            read_plist, signal_keys_buffer.get());

        H5Dclose(signal_keys);

        // signal shapes
        hid_t signal_shapes = H5Dopen(component_group, "signal_shapes", H5P_DEFAULT);

        dspace = H5Dget_space(signal_shapes);
        ndim = H5Sget_simple_extent_dims(dspace, shape, NULL);
        H5Sclose(dspace);

        assert(shape[0] == n_signals);
        assert(n_signals == 0 || shape[1] == 2);
        assert(n_signals == 0 || ndim == 2);

        auto signal_shapes_buffer = unique_ptr<short[]>(new short [2 * n_signals]);
        err = H5Dread(
            signal_shapes, H5T_NATIVE_SHORT, H5S_ALL, H5S_ALL,
            read_plist, signal_shapes_buffer.get());

        H5Dclose(signal_shapes);

        // signal labels
        hid_t labels = H5Dopen(component_group, "signal_labels", H5P_DEFAULT);

        dspace = H5Dget_space(labels);
        ndim = H5Sget_simple_extent_dims(dspace, shape, NULL);
        H5Sclose(dspace);

        assert(ndim == 1);

        auto label_buffer = unique_ptr<char>(new char[shape[0]]);
        err = H5Dread(labels, str_type, H5S_ALL, H5S_ALL, read_plist, label_buffer.get());
        H5Dclose(labels);

        // signals
        hid_t signals = H5Dopen(component_group, "signals", H5P_DEFAULT);

        dspace = H5Dget_space(signals);
        ndim = H5Sget_simple_extent_dims(dspace, shape, NULL);
        H5Sclose(dspace);

        assert(ndim == 1);

        auto signal_buffer = unique_ptr<dtype[]>(new dtype[shape[0]]);

        err = H5Dread(
            signals, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
            read_plist, signal_buffer.get());

        H5Dclose(signals);

        long long signal_offset = 0;
        str_ptr = label_buffer.get();

        // Read signals for component one at a time
        // Name of the dataset containing a signal is equal to the signal key
        for(int i = 0; i < n_signals; i++){
            shape[0] = signal_shapes_buffer[2*i];
            shape[1] = signal_shapes_buffer[2*i + 1];

            // Get the signal data
            auto data = unique_ptr<BaseSignal>(new BaseSignal(shape[0], shape[1]));

            for(int j = 0; j < shape[0]; j++){
                for(int k = 0; k < shape[1]; k++){
                    (*data)(j, k) = signal_buffer[signal_offset + j * shape[1] + k];
                }
            }

            signal_offset += shape[0] * shape[1];

            string label = string(str_ptr);
            while(*str_ptr != '\0'){
                str_ptr++;
            }

            if(i < n_signals-1){
                str_ptr++;
            }

            add_base_signal(signal_keys_buffer[i], label, move(data));
        }

        // Read operators for component

        // Open the dataset
        hid_t operators = H5Dopen(component_group, "operators", H5P_DEFAULT);

        // Get number of ops
        int n_operators;
        attr = H5Aopen(operators, "n_strings", H5P_DEFAULT);
        H5Aread(attr, H5T_NATIVE_INT, &n_operators);
        H5Aclose(attr);

        // Get its dimensions
        dspace = H5Dget_space(operators);
        ndim = H5Sget_simple_extent_dims(dspace, shape, NULL);
        H5Sclose(dspace);

        // Read the data set
        auto op_buffer = unique_ptr<char>(new char[shape[0]]);
        err = H5Dread(operators, str_type, H5S_ALL, H5S_ALL, read_plist, op_buffer.get());
        H5Dclose(operators);

        // Add the ops
        str_ptr = op_buffer.get();

        for(int op_idx=0; op_idx < n_operators; op_idx++){
            string op_str = string(str_ptr);

            add_op(OpSpec(op_str));

            while(*str_ptr != '\0'){
                str_ptr++;
            }

            if(op_idx < n_operators-1){
                str_ptr++;
            }
        }

        // Read probes for component

        // Open the dataset
        hid_t probes = H5Dopen(component_group, "probes", H5P_DEFAULT);

        // Get number of probes
        int n_probes;
        attr = H5Aopen(probes, "n_strings", H5P_DEFAULT);
        H5Aread(attr, H5T_NATIVE_INT, &n_probes);
        H5Aclose(attr);

        // Get its dimensions
        dspace = H5Dget_space(probes);
        ndim = H5Sget_simple_extent_dims(dspace, shape, NULL);
        H5Sclose(dspace);

        // Read the data set
        auto probe_buffer = unique_ptr<char>(new char[shape[0]]);
        err = H5Dread(probes, str_type, H5S_ALL, H5S_ALL, read_plist, probe_buffer.get());
        H5Dclose(probes);

        // Add the probes
        str_ptr = probe_buffer.get();

        for(int probe_idx=0; probe_idx < n_probes; probe_idx++){
            string probe_str = string(str_ptr);
            add_probe(ProbeSpec(probe_str));

            while(*str_ptr != '\0'){
                str_ptr++;
            }

            if(probe_idx < n_probes-1){
                str_ptr++;
            }
        }

        H5Gclose(component_group);

        component += n_processors;
    }

    // Read probe info

    // Open the dataset
    hid_t probe_info_dset = H5Dopen(f, "probe_info", H5P_DEFAULT);

    // Get total number of probes in simulation
    int n_total_probes;
    attr = H5Aopen(probe_info_dset, "n_strings", H5P_DEFAULT);
    H5Aread(attr, H5T_NATIVE_INT, &n_total_probes);
    H5Aclose(attr);

    // Get its dimensions
    dspace = H5Dget_space(probe_info_dset);
    ndim = H5Sget_simple_extent_dims(dspace, shape, NULL);
    H5Sclose(dspace);

    // Read the data set
    auto probe_buffer = unique_ptr<char>(new char[shape[0]]);
    err = H5Dread(probe_info_dset, str_type, H5S_ALL, H5S_ALL, read_plist, probe_buffer.get());
    H5Dclose(probe_info_dset);

    str_ptr = probe_buffer.get();

    for(int probe_idx=0; probe_idx < n_total_probes; probe_idx++){
        string probe_str = string(str_ptr);
        probe_info.push_back(probe_str);

        while(*str_ptr != '\0'){
            str_ptr++;
        }

        if(probe_idx < n_total_probes-1){
            str_ptr++;
        }
    }

    H5Fclose(f);
}

void MpiSimulatorChunk::finalize_build(){
    finalize_build(MPI_COMM_NULL);
}

void MpiSimulatorChunk::finalize_build(MPI_Comm comm){
    if(n_processors != 1){
        sim_log = unique_ptr<SimulationLog>(
            new ParallelSimulationLog(n_processors, rank, probe_info, dt, comm));
    }else{
        sim_log = unique_ptr<SimulationLog>(new SimulationLog(probe_info, dt));
    }

    if(mpi_merged){
        for(auto& kv : merged_sends){

            int dst = kv.first;
            vector<pair<int, SignalView>> content = kv.second;
            vector<pair<int, SignalView*>> content_prime;
            for(auto& p : content){
                content_prime.push_back({p.first, &(p.second)});
            }

            stable_sort(
                content_prime.begin(), content_prime.end(), compare_first_lt<int, SignalView*>);
            vector<SignalView> signals_only;

            for(auto& p : content_prime){
                signals_only.push_back(*(p.second));
            }

            int tag = send_tags[dst];

            // Create the merged op, put it in the op list
            auto merged_send = unique_ptr<MPIOperator>(
                new MergedMPISend(dst, tag, signals_only));

            operator_list.push_back((Operator*) merged_send.get());
            mpi_sends.push_back(move(merged_send));
        }

        for(auto& kv : merged_recvs){

            int src = kv.first;
            vector<pair<int, SignalView>> content = kv.second;
            vector<pair<int, SignalView*>> content_prime;
            for(auto& p : content){
                content_prime.push_back({p.first, &(p.second)});
            }
            stable_sort(
                content_prime.begin(), content_prime.end(), compare_first_lt<int, SignalView*>);

            vector<SignalView> signals_only;
            for(auto& p : content_prime){
                signals_only.push_back(*(p.second));
            }

            int tag = recv_tags[src];

            // Create the merged op, put it in the op list
            auto merged_recv = unique_ptr<MPIOperator>(
                new MergedMPIRecv(src, tag, signals_only));

            operator_list.push_back((Operator*) merged_recv.get());
            mpi_recvs.push_back(move(merged_recv));
        }
    }

    for(auto& send: mpi_sends){
        send->set_communicator(comm);
    }

    for(auto& recv: mpi_recvs){
        recv->set_communicator(comm);
    }

    // Very important; ensures ops are executed in correct order
    operator_list.sort(compare_op_ptr);
}

void MpiSimulatorChunk::run_n_steps(int steps, bool progress){

    stringstream ss;
    ss << "chunk_" << rank << "_dbg";
    dbgfile(ss.str());

    if(rank == 0){
        sim_log->prep_for_simulation(log_filename, steps);
    }else{
        sim_log->prep_for_simulation();
    }

    int flush_every;
    if(sim_log->is_ready()){
        flush_every = FLUSH_PROBES_EVERY;
    }else{
        flush_every = 0;
    }

    for(auto& kv: probe_map){
        (kv.second)->init_for_simulation(steps, flush_every);
    }

    ez::ezETAProgressBar eta(steps);

    if(progress){
        eta.start();
    }

    map<string, double> per_class_average_timings;
    double per_op_timings[operator_list.size()];
    fill_n(per_op_timings, operator_list.size(), 0.0);

    vector<double> step_times;

    for(unsigned step = 0; step < steps; ++step){
        clock_t begin = clock();

        if(step % FLUSH_PROBES_EVERY == 0 && step != 0){
            dbg("Rank " << rank << " beginning step: " << step << ", flushing probes." << endl);
            flush_probes();
        }

        if(!progress && rank == 0 && step % 100 == 0){
            cout << "Master beginning step: " << step << endl;
        }

        dbg("Rank " << rank << " beginning step: " << step << endl);

        // Update time before calling operators, as refimpl does
        n_steps++;
        time = n_steps * dt;

        if(collect_timings){
            int op_index = 0;
            for(auto& op: operator_list){
                clock_t op_begin = clock();

                //Call the operator
                (*op)();

                clock_t op_end = clock();

                if(collect_timings){
                    per_op_timings[op_index] += double(op_end - op_begin) / CLOCKS_PER_SEC;
                }

                op_index++;
            }
        }else{
            for(auto& op: operator_list){
                //Call the operator
                (*op)();
            }

        }

        for(auto& kv: probe_map){
            (kv.second)->gather(n_steps);
        }

        if(progress){
            ++eta;
        }

        clock_t end = clock();
        step_times.push_back(double(end - begin) / CLOCKS_PER_SEC);
    }

    flush_probes();

    for(auto& send: mpi_sends){
        send->complete();
    }

    for(auto& recv : mpi_recvs){
        recv->complete();
    }

    clsdbgfile();

    if(collect_timings){
        // Handle timing data
        double sum = std::accumulate(step_times.begin(), step_times.end(), 0.0);
        double mean = sum / step_times.size();

        vector<double> diff(step_times.size());
        transform(step_times.begin(), step_times.end(), diff.begin(),
                       bind2nd(minus<double>(), mean));
        double sq_sum = inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        double stdev = sqrt(sq_sum / step_times.size());

        map<string, double> class_average;
        map<string, double> class_count;
        map<string, double> class_slowest;
        map<string, Operator*> class_slowest_op;
        map<string, double> class_cumulative;

        int op_index = 0;
        for(auto& op: operator_list){
            string class_name = op->classname();

            class_count[class_name] += 1;

            if(class_slowest[class_name] < per_op_timings[op_index]){
                class_slowest[class_name] = per_op_timings[op_index];
                class_slowest_op[class_name] = op;
            }

            class_cumulative[class_name] += per_op_timings[op_index];

            op_index++;
        }

        stringstream runtimes_ss;
        string delim = ",";

        runtimes_ss << endl << "Rank " << rank << " runtimes." << endl;
        runtimes_ss << "Mean seconds-per-step: " << mean << ", stdev: " << stdev << endl;

        for(auto& p : class_cumulative){
            string class_name = p.first;
            double value = p.second / class_count[class_name] / double(n_steps);
            runtimes_ss << class_name << "_average" << delim << value << endl;

            value = p.second / double(n_steps);
            runtimes_ss << class_name << "_cumulative" << delim << value << endl;

            value = class_slowest[class_name] / double(n_steps);
            runtimes_ss << class_name << "_slowest" << delim << value << endl;
            runtimes_ss << class_name << "_slowest_op" << delim << endl << *class_slowest_op[class_name] << endl;
        }

        vector<pair<double, Operator*>> op_runtimes;

        op_index = 0;
        for(auto op: operator_list){
            op_runtimes.push_back({per_op_timings[op_index], op});
            op_index++;
        }

        int n_show = 10;
        partial_sort(
            op_runtimes.begin(), op_runtimes.begin()+n_show,
            op_runtimes.end(), compare_first_gt<double, Operator*>);

        runtimes_ss << n_show << " slowest operators: " << endl;
        for(int i = 0; i < n_show; i++){
            runtimes_ss << "OPERATOR " << i << endl;
            runtimes_ss << "Seconds per step: " << op_runtimes[i].first / double(n_steps) << endl;
            runtimes_ss << *(op_runtimes[i].second) << endl;
        }

        sim_log->write_file("_runtimes", rank, MAX_RUNTIME_OUTPUT_SIZE, runtimes_ss.str());
    }
}

void MpiSimulatorChunk::add_base_signal(
        key_type key, string l, unique_ptr<BaseSignal> data){

    auto key_location = signal_map.find(key);

    if(key_location != signal_map.end()){
        shared_ptr<BaseSignal> existing_data = key_location->second;

        bool identical = true;
        identical &= existing_data->size1() == data->size1();
        identical &= existing_data->size2() == data->size2();

        if(identical){
            for(int i = 0; i < data->size1(); i++){
                for(int j = 0; j < data->size2(); j++){
                    identical &= (*data)(i, j) == (*existing_data)(i, j);
                }
            }
        }

        if(!identical){
            stringstream msg;
            msg << "Adding signal with duplicate key to chunk with rank " << rank
                << ", but the data is not identical." << endl;

            throw logic_error(msg.str());
        }
    }else{
        signal_labels[key] = l;
        signal_map[key] = shared_ptr<BaseSignal>(move(data));
    }
}

SignalView MpiSimulatorChunk::get_signal_view(
        key_type key, int shape1, int shape2, int stride1, int stride2, int offset){

    if(signal_map.find(key) == signal_map.end()){
        stringstream msg;
        msg << "Error accessing MpiSimulatorChunk :: signal with key " << key;
        throw out_of_range(msg.str());
    }

    // TODO: This only handles cases that could just as easily be handled with ranges
    // rather than slices. So either make it more general, or make it use ranges.
    int start1 = offset / stride1;
    int start2 = offset % stride1;

    return SignalView(
        *signal_map.at(key), ublas::slice(start1, 1, shape1),
        ublas::slice(start2, 1, shape2));
}

SignalView MpiSimulatorChunk::get_signal_view(SignalSpec ss){
    return get_signal_view(
        ss.key, ss.shape1, ss.shape2, ss.stride1, ss.stride2, ss.offset);
}

SignalView MpiSimulatorChunk::get_signal_view(string ss){
    return get_signal_view(SignalSpec(ss));
}

SignalView MpiSimulatorChunk::get_signal_view(key_type key){
    shared_ptr<BaseSignal> base_signal = signal_map.at(key);
    return SignalView(
        *base_signal, ublas::slice(0, 1, base_signal->size1()),
        ublas::slice(0, 1, base_signal->size2()));
}

void MpiSimulatorChunk::add_op(unique_ptr<Operator> op){
    operator_list.push_back(op.get());
    operator_store.push_back(move(op));
}

void MpiSimulatorChunk::add_op(OpSpec op_spec){
    string type_string = op_spec.type_string;
    vector<string>& args = op_spec.arguments;
    float index = op_spec.index;

    try{
        if(type_string.compare("Reset") == 0){
            SignalView dst = get_signal_view(args[0]);
            dtype value = boost::lexical_cast<dtype>(args[1]);

            add_op(unique_ptr<Operator>(new Reset(dst, value)));

        }else if(type_string.compare("Copy") == 0){

            SignalView dst = get_signal_view(args[0]);
            SignalView src = get_signal_view(args[1]);

            add_op(unique_ptr<Operator>(new Copy(dst, src)));

        }else if(type_string.compare("SlicedCopy") == 0){

            SignalView dst = get_signal_view(args[0]);
            SignalView src = get_signal_view(args[1]);

            bool inc = bool(boost::lexical_cast<int>(args[2]));

            int start_A = boost::lexical_cast<int>(args[3]);
            int stop_A = boost::lexical_cast<int>(args[4]);
            int step_A = boost::lexical_cast<int>(args[5]);

            int start_B = boost::lexical_cast<int>(args[6]);
            int stop_B = boost::lexical_cast<int>(args[7]);
            int step_B = boost::lexical_cast<int>(args[8]);

            vector<int> seq_A = python_list_to_index_vector(args[9]);
            vector<int> seq_B = python_list_to_index_vector(args[10]);

            add_op(unique_ptr<Operator>(
                new SlicedCopy(
                    dst, src, inc, start_A, stop_A, step_A,
                    start_B, stop_B, step_B, seq_A, seq_B)));

        }else if(type_string.compare("DotInc") == 0){
            SignalView A = get_signal_view(args[0]);
            SignalView X = get_signal_view(args[1]);
            SignalView Y = get_signal_view(args[2]);

            add_op(unique_ptr<Operator>(new DotInc(A, X, Y)));

        }else if(type_string.compare("ElementwiseInc") == 0){
            SignalView A = get_signal_view(args[0]);
            SignalView X = get_signal_view(args[1]);
            SignalView Y = get_signal_view(args[2]);

            add_op(unique_ptr<Operator>(new ElementwiseInc(A, X, Y)));

        }else if(type_string.compare("LIF") == 0){
            int n_neurons = boost::lexical_cast<int>(args[0]);
            dtype tau_rc = boost::lexical_cast<dtype>(args[1]);
            dtype tau_ref = boost::lexical_cast<dtype>(args[2]);
            dtype min_voltage = boost::lexical_cast<dtype>(args[3]);
            dtype dt = boost::lexical_cast<dtype>(args[4]);

            SignalView J = get_signal_view(args[5]);
            SignalView output = get_signal_view(args[6]);
            SignalView voltage = get_signal_view(args[7]);
            SignalView ref_time = get_signal_view(args[8]);

            add_op(unique_ptr<Operator>(
                new LIF(
                    n_neurons, tau_rc, tau_ref, min_voltage,
                    dt, J, output, voltage, ref_time)));

        }else if(type_string.compare("LIFRate") == 0){
            int n_neurons = boost::lexical_cast<int>(args[0]);
            dtype tau_rc = boost::lexical_cast<dtype>(args[1]);
            dtype tau_ref = boost::lexical_cast<dtype>(args[2]);

            SignalView J = get_signal_view(args[3]);
            SignalView output = get_signal_view(args[4]);

            add_op(unique_ptr<Operator>(
                new LIFRate(n_neurons, tau_rc, tau_ref, J, output)));

        }else if(type_string.compare("AdaptiveLIF") == 0){
            int n_neurons = boost::lexical_cast<int>(args[0]);

            dtype tau_n = boost::lexical_cast<dtype>(args[1]);
            dtype inc_n = boost::lexical_cast<dtype>(args[2]);

            dtype tau_rc = boost::lexical_cast<dtype>(args[3]);
            dtype tau_ref = boost::lexical_cast<dtype>(args[4]);
            dtype min_voltage = boost::lexical_cast<dtype>(args[5]);
            dtype dt = boost::lexical_cast<dtype>(args[6]);

            SignalView J = get_signal_view(args[7]);
            SignalView output = get_signal_view(args[8]);
            SignalView voltage = get_signal_view(args[9]);
            SignalView ref_time = get_signal_view(args[10]);
            SignalView adaptation = get_signal_view(args[11]);

            add_op(unique_ptr<Operator>(
                new AdaptiveLIF(
                    n_neurons, tau_n, inc_n, tau_rc, tau_ref,
                    min_voltage, dt, J, output, voltage, ref_time,
                    adaptation)));

        }else if(type_string.compare("AdaptiveLIFRate") == 0){
            int n_neurons = boost::lexical_cast<int>(args[0]);

            dtype tau_n = boost::lexical_cast<dtype>(args[1]);
            dtype inc_n = boost::lexical_cast<dtype>(args[2]);

            dtype tau_rc = boost::lexical_cast<dtype>(args[3]);
            dtype tau_ref = boost::lexical_cast<dtype>(args[4]);

            dtype dt = boost::lexical_cast<dtype>(args[5]);

            SignalView J = get_signal_view(args[6]);
            SignalView output = get_signal_view(args[7]);
            SignalView adaptation = get_signal_view(args[8]);

            add_op(unique_ptr<Operator>(
                new AdaptiveLIFRate(
                    n_neurons, tau_n, inc_n, tau_rc, tau_ref,
                    dt, J, output, adaptation)));

        }else if(type_string.compare("RectifiedLinear") == 0){
            int n_neurons = boost::lexical_cast<int>(args[0]);

            SignalView J = get_signal_view(args[1]);
            SignalView output = get_signal_view(args[2]);

            add_op(unique_ptr<Operator>(new RectifiedLinear(n_neurons, J, output)));

        }else if(type_string.compare("Sigmoid") == 0){
            int n_neurons = boost::lexical_cast<int>(args[0]);
            dtype tau_ref = boost::lexical_cast<dtype>(args[1]);

            SignalView J = get_signal_view(args[2]);
            SignalView output = get_signal_view(args[3]);

            add_op(unique_ptr<Operator>(new Sigmoid(n_neurons, tau_ref, J, output)));

        }else if(type_string.compare("Izhikevich") == 0){
            int n_neurons = boost::lexical_cast<int>(args[0]);

            dtype tau_recovery = boost::lexical_cast<dtype>(args[1]);
            dtype coupling = boost::lexical_cast<dtype>(args[2]);
            dtype reset_voltage = boost::lexical_cast<dtype>(args[3]);
            dtype reset_recovery = boost::lexical_cast<dtype>(args[4]);
            dtype dt = boost::lexical_cast<dtype>(args[5]);

            SignalView J = get_signal_view(args[6]);
            SignalView output = get_signal_view(args[7]);
            SignalView voltage = get_signal_view(args[8]);
            SignalView recovery = get_signal_view(args[9]);

            add_op(unique_ptr<Operator>(
                new Izhikevich(
                    n_neurons, tau_recovery, coupling, reset_voltage,
                    reset_recovery, dt, J, output, voltage, recovery)));

        }else if(type_string.compare("NoDenSynapse") == 0){

            SignalView input = get_signal_view(args[0]);
            SignalView output = get_signal_view(args[1]);
            dtype b = boost::lexical_cast<dtype>(args[2]);

            add_op(unique_ptr<Operator>(new NoDenSynapse(input, output, b)));

        }else if(type_string.compare("SimpleSynapse") == 0){

            SignalView input = get_signal_view(args[0]);
            SignalView output = get_signal_view(args[1]);
            dtype a = boost::lexical_cast<dtype>(args[2]);
            dtype b = boost::lexical_cast<dtype>(args[3]);

            add_op(unique_ptr<Operator>(new SimpleSynapse(input, output, a, b)));

        }else if(type_string.compare("Synapse") == 0){

            SignalView input = get_signal_view(args[0]);
            SignalView output = get_signal_view(args[1]);

            unique_ptr<BaseSignal> numerator = python_list_to_signal(args[2]);
            unique_ptr<BaseSignal> denominator = python_list_to_signal(args[3]);

            add_op(unique_ptr<Operator>(new Synapse(input, output, *numerator, *denominator)));

        }else if(type_string.compare("WhiteNoise") == 0){

            SignalView output = get_signal_view(args[0]);

            dtype mean = boost::lexical_cast<dtype>(args[1]);
            dtype std = boost::lexical_cast<dtype>(args[2]);

            bool do_scale = bool(boost::lexical_cast<int>(args[3]));
            bool inc = bool(boost::lexical_cast<int>(args[4]));

            dtype dt = boost::lexical_cast<dtype>(args[5]);

            add_op(unique_ptr<Operator>(
                new WhiteNoise(output, mean, std, do_scale, inc, dt)));

        }else if(type_string.compare("MpiSend") == 0){

            if(n_processors > 1){
                int dst = boost::lexical_cast<int>(args[0]);
                dst = dst % n_processors;
                if(dst != rank){

                    int tag = boost::lexical_cast<int>(args[1]);
                    key_type signal_key = boost::lexical_cast<key_type>(args[2]);
                    SignalView content = get_signal_view(signal_key);

                    add_mpi_send(index, dst, tag, content);
                }
            }

        }else if(type_string.compare("MpiRecv") == 0){

            if(n_processors > 1){
                int src = boost::lexical_cast<int>(args[0]);
                src = src % n_processors;

                if(src != rank){
                    int tag = boost::lexical_cast<int>(args[1]);
                    key_type signal_key = boost::lexical_cast<key_type>(args[2]);
                    SignalView content = get_signal_view(signal_key);

                    add_mpi_recv(index, src, tag, content);
                }
            }

        }else if(type_string.compare("SpaunStimulus") == 0){
            SignalView output = get_signal_view(args[0]);

            string stim_seq_str = args[1];
            boost::trim_if(stim_seq_str, boost::is_any_of("[]"));
            boost::replace_all(stim_seq_str, "\"", "");
            boost::replace_all(stim_seq_str, "\'", "");

            vector<string> stim_seq;
            boost::split(stim_seq, stim_seq_str, boost::is_any_of(","));

            float present_interval = boost::lexical_cast<float>(args[2]);
            float present_blanks = boost::lexical_cast<float>(args[3]);

            auto op = unique_ptr<Operator>(
                new SpaunStimulus(
                     output, get_time_pointer(), stim_seq,
                     present_interval, present_blanks));

            add_op(move(op));

        }else{
            stringstream msg;
            msg << "Received an operator type that nengo_mpi can't handle: " << type_string;
            throw runtime_error(msg.str());
        }

        operator_list.back()->set_index(index);

    }catch(const boost::bad_lexical_cast& e){
        stringstream msg;
        msg << "Caught bad lexical cast while extracting operator from OpSpec "
               "with error " << e.what() << endl;
        msg << "The operator type was: " << type_string << endl;
        msg << "The arguments were: " << endl;
        int i = 0;
        for(auto& s: args){
            msg << i << ": " << s << endl;
            i++;
        }

        throw runtime_error(msg.str());
    }
}

void MpiSimulatorChunk::add_mpi_send(float index, int dst, int tag, SignalView content){

    if(mpi_merged){
        if(merged_sends.find(dst) == merged_sends.end()){
            send_indices[dst] = index;
            send_tags[dst] = tag;
        }else{
            send_indices[dst] = max(send_indices[dst], index);
            send_tags[dst] = min(send_tags[dst], tag);
        }

        merged_sends[dst].push_back({tag, content});

    }else{
        auto mpi_send = unique_ptr<MPIOperator>(new MPISend(dst, tag, content));
        operator_list.push_back((Operator *) mpi_send.get());
        mpi_sends.push_back(move(mpi_send));
    }
}

void MpiSimulatorChunk::add_mpi_recv(float index, int src, int tag, SignalView content){

    if(mpi_merged){
        if(merged_recvs.find(src) == merged_recvs.end()){
            recv_indices[src] = index;
            recv_tags[src] = tag;
        }else{
            recv_indices[src] = min(recv_indices[src], index);
            recv_tags[src] = min(recv_tags[src], tag);
        }

        merged_recvs[src].push_back({tag, content});

    }else{
        auto mpi_recv = unique_ptr<MPIOperator>(new MPIRecv(src, tag, content));
        operator_list.push_back((Operator *) mpi_recv.get());
        mpi_recvs.push_back(move(mpi_recv));
    }
}

void MpiSimulatorChunk::add_probe(ProbeSpec ps){
    SignalView signal = get_signal_view(ps.signal_spec);
    probe_map[ps.probe_key] = shared_ptr<Probe>(new Probe(signal, ps.period));
}

void MpiSimulatorChunk::set_log_filename(string lf){
    log_filename = lf;
}

bool MpiSimulatorChunk::is_logging(){
    if(sim_log){
        return sim_log->is_ready();
    }else{
        throw logic_error(
            "Calling chunk.is_logging when the chunk does not have logger object.");
    }
}

void MpiSimulatorChunk::close_simulation_log(){
    sim_log->close();
}

void MpiSimulatorChunk::flush_probes(){
    if(sim_log->is_ready()){
        for(auto& kv : probe_map){
            int n_rows;
            shared_ptr<dtype> buffer = (kv.second)->flush_to_buffer(n_rows);

            try{
                sim_log->write(kv.first, buffer, n_rows);
            }catch(out_of_range& e){
                stringstream msg;
                msg << "Trying to write to simulation log on rank " << rank << " using "
                       "invalid probe key: " << kv.first << "." << endl;
                throw out_of_range(msg.str());
            }
        }
    }
}

string MpiSimulatorChunk::to_string() const{
    stringstream out;

    out << "<MpiSimulatorChunk" << endl;
    out << "    Label: " << label << endl;

    out << "** Signals: **" << endl;
    for(auto const& kv: signal_map){
        out << "Key: " << kv.first << endl;
        out << "Label: " << signal_labels.at(kv.first);
        out << "SignalView: " << *(kv.second) << endl;
    }
    out << endl;

    out << "** Probes: **" << endl;
    for(auto const& kv: probe_map){
        out << "Key: " << kv.first << endl;
        out << "Probe: " << *(kv.second) << endl;
    }
    out << endl;

    out << "** Operators: **" << endl;
    for(auto const& op : operator_list){
        out << *op << endl;
    }
    out << endl;

    out << "** MPISends: **" << endl;
    for(auto const& send : mpi_sends){
        out << "Value: " << *send << endl;
    }

    out << "** MPIRecvs: **" << endl;
    for(auto const& recv : mpi_recvs){
        out << "Value: " << *recv << endl;
    }

    return out.str();
}
