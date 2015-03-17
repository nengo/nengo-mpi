#include "chunk.hpp"

MpiSimulatorChunk::MpiSimulatorChunk()
    :time(0.0), dt(0.001), n_steps(0){
}

MpiSimulatorChunk::MpiSimulatorChunk(string label, dtype dt)
    :time(0.0), label(label), dt(dt), n_steps(0){
}

void MpiSimulatorChunk::run_n_steps(int steps, bool progress){

    cout << label << ": running " << steps << " steps." << endl;

    for(auto& kv: probe_map){
        (kv.second)->init_for_simulation(steps);
    }

    ez::ezETAProgressBar eta(steps);

    if(progress){
        eta.start();
    }

    for(unsigned step = 0; step < steps; ++step){

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

    for(auto& send: mpi_sends){
        send->complete();
    }

    for(auto& recv : mpi_recvs){
        recv->complete();
    }
}

void MpiSimulatorChunk::add_base_signal(
        key_type key, string l, unique_ptr<BaseSignal> data){

    signal_labels[key] = l;
    signal_map[key] = shared_ptr<BaseSignal>(move(data));
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

SignalView MpiSimulatorChunk::get_signal_view(string signal_string){
    vector<string> tokens;
    boost::split(tokens, signal_string, boost::is_any_of(":"));

    try{
        key_type key = boost::lexical_cast<key_type>(tokens[0]);

        vector<string> shape_tokens;
        boost::trim_if(tokens[1], boost::is_any_of("(,)"));
        boost::split(shape_tokens, tokens[1], boost::is_any_of(","));

        int shape1 = boost::lexical_cast<int>(shape_tokens[0]);
        int shape2 = shape_tokens.size() == 1 ? 1 : boost::lexical_cast<int>(shape_tokens[1]);

        vector<string> stride_tokens;
        boost::trim_if(tokens[2], boost::is_any_of("(,)"));
        boost::split(stride_tokens, tokens[2], boost::is_any_of(","));

        int stride1 = boost::lexical_cast<int>(stride_tokens[0]);
        int stride2 = stride_tokens.size() == 1 ? 1 : boost::lexical_cast<int>(stride_tokens[1]);

        int offset = boost::lexical_cast<int>(tokens[3]);

        return get_signal_view(key, shape1, shape2, stride1, stride2, offset);
    }catch(const boost::bad_lexical_cast& e){
        cout << "Caught bad lexical cast while extracting signal from string "
                "with error " << e.what() << endl;
        terminate();
    }
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

void MpiSimulatorChunk::add_op(string op_string){
    vector<string> tokens;
    boost::split(tokens, op_string, boost::is_any_of(";"));

    auto it = tokens.cbegin();
    string type_string = *(it++);

    try{
        if(type_string.compare("Reset") == 0){
                SignalView dst = get_signal_view(*(it++));
                dtype value = boost::lexical_cast<dtype>(*(it++));

                add_op(unique_ptr<Operator>(new Reset(dst, value)));

         }else if(type_string.compare("Copy") == 0){

                SignalView dst = get_signal_view(tokens[1]);
                SignalView src = get_signal_view(tokens[2]);

                add_op(unique_ptr<Operator>(new Copy(dst, src)));

         }else if(type_string.compare("DotInc") == 0){
                SignalView A = get_signal_view(tokens[1]);
                SignalView X = get_signal_view(tokens[2]);
                SignalView Y = get_signal_view(tokens[3]);

                add_op(unique_ptr<Operator>(new DotInc(A, X, Y)));

         }else if(type_string.compare("ElementwiseInc") == 0){
                SignalView A = get_signal_view(tokens[1]);
                SignalView X = get_signal_view(tokens[2]);
                SignalView Y = get_signal_view(tokens[3]);

                add_op(unique_ptr<Operator>(new ElementwiseInc(A, X, Y)));

         }else if(type_string.compare("LIF") == 0){
                int num_neurons = boost::lexical_cast<int>(tokens[1]);
                dtype tau_ref = boost::lexical_cast<dtype>(tokens[2]);
                dtype tau_rc = boost::lexical_cast<dtype>(tokens[3]);
                dtype dt = boost::lexical_cast<dtype>(tokens[4]);

                SignalView J = get_signal_view(tokens[5]);
                SignalView output = get_signal_view(tokens[6]);

                add_op(unique_ptr<Operator>(new SimLIF(num_neurons, tau_ref, tau_rc, dt, J, output)));

         }else if(type_string.compare("LIFRate") == 0){
                int num_neurons = boost::lexical_cast<int>(tokens[1]);
                dtype tau_ref = boost::lexical_cast<dtype>(tokens[2]);
                dtype tau_rc = boost::lexical_cast<dtype>(tokens[3]);

                SignalView J = get_signal_view(tokens[4]);
                SignalView output = get_signal_view(tokens[5]);

                add_op(unique_ptr<Operator>(new SimLIFRate(num_neurons, tau_ref, tau_rc, J, output)));

         }else if(type_string.compare("RectifiedLinear") == 0){
                int num_neurons = boost::lexical_cast<int>(tokens[1]);

                SignalView J = get_signal_view(tokens[2]);
                SignalView output = get_signal_view(tokens[3]);

                add_op(unique_ptr<Operator>(new SimRectifiedLinear(num_neurons, J, output)));

         }else if(type_string.compare("Sigmoid") == 0){
                int num_neurons = boost::lexical_cast<int>(tokens[1]);
                dtype tau_ref = boost::lexical_cast<dtype>(tokens[2]);

                SignalView J = get_signal_view(tokens[3]);
                SignalView output = get_signal_view(tokens[4]);

                add_op(unique_ptr<Operator>(new SimSigmoid(num_neurons, tau_ref, J, output)));

         }else if(type_string.compare("LinearFilter") == 0){

                SignalView input = get_signal_view(tokens[1]);
                SignalView output = get_signal_view(tokens[2]);

                unique_ptr<BaseSignal> numerator = extract_float_list(tokens[3]);
                unique_ptr<BaseSignal> denominator = extract_float_list(tokens[4]);

                add_op(unique_ptr<Operator>(new Synapse(input, output, *numerator, *denominator)));

         }else if(type_string.compare("MpiSend") == 0){
                int dst = boost::lexical_cast<int>(tokens[1]);
                int tag = boost::lexical_cast<int>(tokens[2]);
                key_type signal_key = boost::lexical_cast<key_type>(tokens[3]);
                SignalView content = get_signal_view(signal_key);

                add_mpi_send(unique_ptr<MPISend>(new MPISend(dst, tag, content)));

         }else if(type_string.compare("MpiRecv") == 0){
                int src = boost::lexical_cast<int>(tokens[1]);
                int tag = boost::lexical_cast<int>(tokens[2]);
                key_type signal_key = boost::lexical_cast<key_type>(tokens[3]);
                SignalView content = get_signal_view(signal_key);

                add_mpi_recv(unique_ptr<MPIRecv>(new MPIRecv(src, tag, content)));

         }else if(type_string.compare("SpaunStimulus") == 0){
                SignalView output = get_signal_view(tokens[1]);

                string stim_seq_str = tokens[2];
                boost::trim_if(stim_seq_str, boost::is_any_of("[]"));
                boost::replace_all(stim_seq_str, "\"", "");
                boost::replace_all(stim_seq_str, "\'", "");

                vector<string> stim_seq;
                boost::split(stim_seq, stim_seq_str, boost::is_any_of(","));

                add_op(unique_ptr<Operator>(new SpaunStimulus(output, get_time_pointer(), stim_seq)));

        }else{
            stringstream msg;
            msg << "Received an operator type that we can't handle: " << type_string;
            throw runtime_error(msg.str());
        }
    }catch(const boost::bad_lexical_cast& e){
        stringstream msg;
        msg << "Caught bad lexical cast while extracting operator from string "
               "with error " << e.what() << endl;
        throw runtime_error(msg.str());
    }
}

void MpiSimulatorChunk::add_mpi_send(unique_ptr<MPISend> mpi_send){
    operator_list.push_back((Operator *) mpi_send.get());
    mpi_sends.push_back(move(mpi_send));
}

void MpiSimulatorChunk::add_mpi_recv(unique_ptr<MPIRecv> mpi_recv){
    operator_list.push_back((Operator *) mpi_recv.get());
    mpi_recvs.push_back(move(mpi_recv));
}

void MpiSimulatorChunk::add_probe(key_type probe_key, string signal_string, dtype period){
    SignalView signal = get_signal_view(signal_string);
    probe_map[probe_key] = shared_ptr<Probe>(new Probe(signal, period));
}

void MpiSimulatorChunk::add_probe(key_type probe_key, shared_ptr<Probe> probe){
    probe_map[probe_key] = probe;
}

void MpiSimulatorChunk::set_communicator(MPI_Comm comm){
    for(auto& send: mpi_sends){
        send->set_communicator(comm);
    }

    for(auto& recv: mpi_recvs){
        recv->set_communicator(comm);
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
