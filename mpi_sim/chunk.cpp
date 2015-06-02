#include "chunk.hpp"

MpiSimulatorChunk::MpiSimulatorChunk()
:time(0.0), dt(0.001), n_steps(0), rank(0), n_processors(1), mpi_merged(false){
}

MpiSimulatorChunk::MpiSimulatorChunk(int rank, int n_processors, bool mpi_merged)
:time(0.0), dt(0.001), n_steps(0), rank(rank), n_processors(n_processors), mpi_merged(mpi_merged){
    stringstream ss;
    ss << "Chunk " << rank;
    label = ss.str();
}

void MpiSimulatorChunk::from_file(string filename, hid_t file_plist, hid_t read_plist){
    from_file(filename, file_plist, read_plist, MPI_COMM_NULL);
}

void MpiSimulatorChunk::from_file(string filename, hid_t file_plist, hid_t read_plist, MPI_Comm comm){
    herr_t err;
    hid_t f = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, file_plist);

    // Get n_components
    int n_components;
    hid_t attr = H5Aopen(f, "n_components", H5P_DEFAULT);
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

    const int MAX_LENGTH = 256;
    char c_signal_key[MAX_LENGTH];
    char c_label[MAX_LENGTH];
    int length;

    hid_t str_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type, MAX_LENGTH);
    H5Tset_strpad(str_type, H5T_STR_NULLTERM);

    list<OpSpec> all_ops;

    int component = rank;
    while(component < n_components){

        stringstream ss;
        ss << component;

        // Open the group assigned to my component
        hid_t component_group = H5Gopen(f, ss.str().c_str(), H5P_DEFAULT);

        // Open the signals subgroup
        hid_t signal_group = H5Gopen(component_group, "signals", H5P_DEFAULT);

        hsize_t n_signals;
        err = H5Gget_num_objs(signal_group, &n_signals);

        // Read signals for component one at a time
        // Name of the dataset containing a signal is equal to the signal key
        for(int i = 0; i < n_signals; i++){

            // Get the signal key/dataset name
            length = H5Gget_objname_by_idx(
                signal_group, (hsize_t) i, c_signal_key, (size_t) MAX_LENGTH);
            key_type signal_key = boost::lexical_cast<key_type>(c_signal_key);

            // Open the dataset containing the signal
            hid_t signal = H5Dopen(signal_group, c_signal_key, H5P_DEFAULT);

            // Get array shape
            hid_t dspace = H5Dget_space(signal);
            hsize_t shape[2], max_shape[2];
            int ndim = H5Sget_simple_extent_dims(dspace, shape, max_shape);
            H5Sclose(dspace);

            if(ndim == 1){
                shape[1] = 1;
                max_shape[1] = 1;
            }

            if(ndim != 1 && ndim != 2){
                throw runtime_error(
                    "Got improper value of ndim while reading signal.");
            }

            // Get the signal label
            attr = H5Aopen(signal, "label", H5P_DEFAULT);
            hid_t atype = H5Aget_type(attr);
            H5T_class_t type_class = H5Tget_class(atype);
            if (type_class == H5T_STRING) {
                 hid_t atype_mem = H5Tget_native_type(atype, H5T_DIR_ASCEND);
                 H5Aread(attr, atype_mem, c_label);
                 H5Tclose(atype_mem);
            }else{
                throw runtime_error("Label has incorrect data type.");
            }

            string signal_label = c_label;

            H5Aclose(attr);
            H5Tclose(atype);

            // Get the signal data
            auto data = unique_ptr<BaseSignal>(new BaseSignal(shape[0], shape[1]));
            auto buffer = unique_ptr<dtype>(new dtype[shape[0] * shape[1]]);

            err = H5Dread(signal, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, read_plist, buffer.get());

            int idx = 0;
            for(int i = 0; i < shape[0]; i++){
                for(int j = 0; j < shape[1]; j++){
                    (*data)(i, j) = buffer.get()[idx];
                    idx++;
                }
            }

            H5Dclose(signal);

            // Add the signal to the chunk
            add_base_signal(signal_key, signal_label, move(data));
        }

        H5Gclose(signal_group);

        hsize_t shape[2], max_shape[2];

        // Read operators for component
        //
        // Open the dataset
        hid_t operators = H5Dopen(component_group, "operators", H5P_DEFAULT);

        // Get its dimensions
        hid_t dspace = H5Dget_space(operators);
        int ndim = H5Sget_simple_extent_dims(dspace, shape, NULL);
        H5Sclose(dspace);

        // Get the length of the strings (the width variable). The length
        // of all the strings in the dataset is equal to the length of the longest
        // string (have to do this since HDF5 does not support reading variable-length
        // data-types in parallel).
        hid_t type = H5Dget_type(operators);
        size_t width = H5Tget_size(type);

        // Read the data set
        auto op_buffer = unique_ptr<char>(new char[width * shape[0]]);

        hid_t memtype = H5Tcopy(H5T_C_S1);
        H5Tset_size(memtype, width);
        err = H5Dread(operators, memtype, H5S_ALL, H5S_ALL, read_plist, op_buffer.get());

        list<OpSpec> component_ops;
        for(int i = 0; i < shape[0]; i++){
            string op_string(op_buffer.get() + i * width);
            component_ops.push_back(OpSpec(op_string));
        }

        all_ops.merge(component_ops, compare_indices);

        H5Dclose(operators);

        // Read probes for component
        //
        // Open the dataset
        hid_t probes = H5Dopen(component_group, "probes", H5P_DEFAULT);

        // Get its dimensions
        dspace = H5Dget_space(probes);
        ndim = H5Sget_simple_extent_dims(dspace, shape, NULL);
        H5Sclose(dspace);

        type = H5Dget_type(probes);
        width = H5Tget_size(type);

        // Read the data set
        auto probe_buffer = unique_ptr<char>(new char[width * shape[0]]);

        memtype = H5Tcopy(H5T_C_S1);
        H5Tset_size(memtype, width);
        err = H5Dread(probes, memtype, H5S_ALL, H5S_ALL, read_plist, probe_buffer.get());

        for(int i = 0; i < shape[0]; i++){
            string probe_string(probe_buffer.get() + i * width);

            add_probe(ProbeSpec(probe_string));
        }

        H5Dclose(probes);

        H5Gclose(component_group);

        component += n_processors;
    }

    for(auto& op : all_ops){
        add_op(op);
    }

    // Open the dataset
    hid_t all_probes = H5Dopen(f, "all_probes", H5P_DEFAULT);

    hsize_t shape[2];

    // Get its dimensions
    hid_t dspace = H5Dget_space(all_probes);
    int ndim = H5Sget_simple_extent_dims(dspace, shape, NULL);
    H5Sclose(dspace);

    hid_t type = H5Dget_type(all_probes);
    size_t width = H5Tget_size(type);

    // Read the data set
    auto probe_buffer = unique_ptr<char>(new char[width * shape[0]]);

    hid_t memtype = H5Tcopy(H5T_C_S1);
    H5Tset_size(memtype, width);
    err = H5Dread(all_probes, memtype, H5S_ALL, H5S_ALL, read_plist, probe_buffer.get());

    for(int i = 0; i < shape[0]; i++){
        string probe_string(probe_buffer.get() + i * width);
        probe_info.push_back(ProbeSpec(probe_string));
    }

    H5Dclose(all_probes);
    H5Fclose(f);

    finalize_build(comm);
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

            stable_sort(content_prime.begin(), content_prime.end(), compare_first);
            vector<SignalView> signals_only;

            for(auto& p : content_prime){
                signals_only.push_back(*(p.second));
            }

            int tag = send_tags[dst];

            // Create the merged op, put it in the op list
            auto merged_send = unique_ptr<MPIOperator>(new MergedMPISend(dst, tag, signals_only));

            auto it = send_indices.at(dst);
            it++;
            operator_list.insert(it, (Operator*) merged_send.get());
            mpi_sends.push_back(move(merged_send));
        }

        for(auto& kv : merged_recvs){

            int src = kv.first;
            vector<pair<int, SignalView>> content = kv.second;
            vector<pair<int, SignalView*>> content_prime;
            for(auto& p : content){
                content_prime.push_back({p.first, &(p.second)});
            }
            stable_sort(content_prime.begin(), content_prime.end(), compare_first);

            vector<SignalView> signals_only;
            for(auto& p : content_prime){
                signals_only.push_back(*(p.second));
            }

            int tag = recv_tags[src];

            // Create the merged op, put it in the op list
            auto merged_recv = unique_ptr<MPIOperator>(new MergedMPIRecv(src, tag, signals_only));

            auto it = recv_indices.at(src);
            if(it != operator_list.begin()){
                it++;
            }
            operator_list.insert(it, (Operator*) merged_recv.get());
            mpi_recvs.push_back(move(merged_recv));
        }
    }

    for(auto& send: mpi_sends){
        send->set_communicator(comm);
    }

    for(auto& recv: mpi_recvs){
        recv->set_communicator(comm);
    }
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


    for(unsigned step = 0; step < steps; ++step){
        if(step % FLUSH_PROBES_EVERY == 0 && step != 0){
            dbg("Rank " << rank << " beginning step: " << step << ", flushing probes." << endl);
            flush_probes();
        }

        if(!progress && rank == 0 && step % 100 == 0){
            cout << "Master beginning step: " << step << endl;
        }

        if(!progress){
            dbg("Rank " << rank << " beginning step: " << step << endl);
        }

        // Update time before calling operators, as refimpl does
        n_steps++;
        time = n_steps * dt;

        for(auto& op: operator_list){
            //Call the operator
            (*op)();
        }

        for(auto& kv: probe_map){
            (kv.second)->gather(n_steps);
        }

        if(progress){
            ++eta;
        }
    }

    flush_probes();

    for(auto& send: mpi_sends){
        send->complete();
    }

    for(auto& recv : mpi_recvs){
        recv->complete();
    }

    clsdbgfile();
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

    try{
        if(type_string.compare("Reset") == 0){
            SignalView dst = get_signal_view(args[0]);
            dtype value = boost::lexical_cast<dtype>(args[1]);

            add_op(unique_ptr<Operator>(new Reset(dst, value)));

        }else if(type_string.compare("Copy") == 0){

            SignalView dst = get_signal_view(args[0]);
            SignalView src = get_signal_view(args[1]);

            add_op(unique_ptr<Operator>(new Copy(dst, src)));

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
            int num_neurons = boost::lexical_cast<int>(args[0]);
            dtype tau_ref = boost::lexical_cast<dtype>(args[1]);
            dtype tau_rc = boost::lexical_cast<dtype>(args[2]);
            dtype dt = boost::lexical_cast<dtype>(args[3]);

            SignalView J = get_signal_view(args[4]);
            SignalView output = get_signal_view(args[5]);

            add_op(unique_ptr<Operator>(new SimLIF(num_neurons, tau_ref, tau_rc, dt, J, output)));

        }else if(type_string.compare("LIFRate") == 0){
            int num_neurons = boost::lexical_cast<int>(args[0]);
            dtype tau_ref = boost::lexical_cast<dtype>(args[1]);
            dtype tau_rc = boost::lexical_cast<dtype>(args[2]);

            SignalView J = get_signal_view(args[3]);
            SignalView output = get_signal_view(args[4]);

            add_op(unique_ptr<Operator>(new SimLIFRate(num_neurons, tau_ref, tau_rc, J, output)));

        }else if(type_string.compare("RectifiedLinear") == 0){
            int num_neurons = boost::lexical_cast<int>(args[0]);

            SignalView J = get_signal_view(args[1]);
            SignalView output = get_signal_view(args[2]);

            add_op(unique_ptr<Operator>(new SimRectifiedLinear(num_neurons, J, output)));

        }else if(type_string.compare("Sigmoid") == 0){
            int num_neurons = boost::lexical_cast<int>(args[0]);
            dtype tau_ref = boost::lexical_cast<dtype>(args[1]);

            SignalView J = get_signal_view(args[2]);
            SignalView output = get_signal_view(args[3]);

            add_op(unique_ptr<Operator>(new SimSigmoid(num_neurons, tau_ref, J, output)));

        }else if(type_string.compare("LinearFilter") == 0){

            SignalView input = get_signal_view(args[0]);
            SignalView output = get_signal_view(args[1]);

            unique_ptr<BaseSignal> numerator = extract_float_list(args[2]);
            unique_ptr<BaseSignal> denominator = extract_float_list(args[3]);

            add_op(unique_ptr<Operator>(new Synapse(input, output, *numerator, *denominator)));

        }else if(type_string.compare("MpiSend") == 0){

            if(n_processors > 1){
                int dst = boost::lexical_cast<int>(args[0]);
                dst = dst % n_processors;
                if(dst != rank){

                    int tag = boost::lexical_cast<int>(args[1]);
                    key_type signal_key = boost::lexical_cast<key_type>(args[2]);
                    SignalView content = get_signal_view(signal_key);

                    add_mpi_send(dst, tag, content);
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

                    add_mpi_recv(src, tag, content);
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
    }catch(const boost::bad_lexical_cast& e){
        stringstream msg;
        msg << "Caught bad lexical cast while extracting operator from OpSpec "
               "with error " << e.what() << endl;
        msg << "The operator type was: " << type_string << endl;

        throw runtime_error(msg.str());
    }
}

void MpiSimulatorChunk::add_mpi_send(int dst, int tag, SignalView content){

    if(mpi_merged){
        if(operator_list.empty()){
            send_indices[dst] = operator_list.begin();
        }else{
            send_indices[dst] = operator_list.end();
            send_indices[dst]--;
        }

        if(merged_sends.find(dst) == merged_sends.end()){
            send_tags[dst] = tag;
        }else{
            send_tags[dst] = min(send_tags[dst], tag);
        }

        merged_sends[dst].push_back({tag, content});

    }else{
        auto mpi_send = unique_ptr<MPIOperator>(new MPISend(dst, tag, content));
        operator_list.push_back((Operator *) mpi_send.get());
        mpi_sends.push_back(move(mpi_send));
    }
}

void MpiSimulatorChunk::add_mpi_recv(int src, int tag, SignalView content){

    if(mpi_merged){
        if(merged_recvs.find(src) == merged_recvs.end()){
            if(operator_list.empty()){
                recv_indices[src] = operator_list.begin();
            }else{
                recv_indices[src] = operator_list.end();
                recv_indices[src]--;
            }

            recv_tags[src] = tag;
        }else{
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
