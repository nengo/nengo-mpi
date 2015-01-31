#include "chunk.hpp"

MpiSimulatorChunk::MpiSimulatorChunk()
    :time(0.0), dt(0.001), n_steps(0){
}

MpiSimulatorChunk::MpiSimulatorChunk(string label, dtype dt)
    :time(0.0), label(label), dt(dt), n_steps(0){
}

void MpiSimulatorChunk::run_n_steps(int steps, bool progress){

    cout << label << ": running " << steps << " steps." << endl;

    map<key_type, Probe*>::iterator probe_it;
    for(probe_it = probe_map.begin(); probe_it != probe_map.end(); ++probe_it){
        probe_it->second->init_for_simulation(steps);
    }

    ez::ezETAProgressBar eta(steps);

    if(progress){
        eta.start();
    }

    for(unsigned step = 0; step < steps; ++step){

        // Update time before calling operators, as refimpl does
        n_steps++;
        time = n_steps * dt;

        list<Operator*>::const_iterator it;
        for(it = operator_list.begin(); it != operator_list.end(); ++it){
            //Call the operator
            (**it)();
        }

        map<key_type, Probe*>::iterator probe_it;
        for(probe_it = probe_map.begin(); probe_it != probe_map.end(); ++probe_it){
            //Call the operator
            probe_it->second->gather(n_steps);
        }

        if(progress){
            ++eta;
        }
    }

    vector<MPISend*>::const_iterator send_it = mpi_sends.begin();
    for(; send_it != mpi_sends.end(); ++send_it){
        (*send_it)->complete();
    }

    vector<MPIRecv*>::const_iterator recv_it = mpi_recvs.begin();
    for(; recv_it != mpi_recvs.end(); ++recv_it){
        (*recv_it)->complete();
    }
}

void MpiSimulatorChunk::add_base_signal(key_type key, string l, BaseMatrix* data){
    signal_map[key] = data;
    signal_labels[key] = l;
}

BaseMatrix* MpiSimulatorChunk::get_base_signal(key_type key){
    try{
        BaseMatrix* mat = signal_map.at(key);
        return mat;
    }catch(const out_of_range& e){
        cerr << "Error accessing MpiSimulatorChunk :: signal with key " << key << endl;
        throw e;
    }
}

Matrix MpiSimulatorChunk::get_signal(key_type key, int shape1, int shape2,
                                      int stride1, int stride2, int offset){
    BaseMatrix* mat = get_base_signal(key);

    // TODO: This only handles cases that could just as easily be handled with ranges
    // rather than slices. So either make it more general, or make it use ranges.
    int start1 = offset / stride1;
    int start2 = offset % stride1;

    return Matrix(*mat, ublas::slice(start1, 1, shape1), ublas::slice(start2, 1, shape2));
}

Matrix MpiSimulatorChunk::get_signal(string signal_string){
    vector<string> tokens;
    boost::split(tokens, signal_string, boost::is_any_of(":"));
    vector<string>::iterator it;

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

        return get_signal(key, shape1, shape2, stride1, stride2, offset);
    }catch(const boost::bad_lexical_cast& e){
        cout << "Caught bad lexical cast while extracting signal from string "
                "with error " << e.what() << endl;
        terminate();
    }
}

void MpiSimulatorChunk::add_op(Operator *op){
    operator_list.push_back(op);
}

void MpiSimulatorChunk::add_op(string op_string){
    vector<string> tokens;
    boost::split(tokens, op_string, boost::is_any_of(";"));

    vector<string>::const_iterator it = tokens.begin();
    string type_string = *(it++);

    try{
        if(type_string.compare("Reset") == 0){
                Matrix dst = get_signal(*(it++));
                dtype value = boost::lexical_cast<dtype>(*(it++));

                add_op(new Reset(dst, value));

         }else if(type_string.compare("Copy") == 0){

                Matrix dst = get_signal(tokens[1]);
                Matrix src = get_signal(tokens[2]);

                add_op(new Copy(dst, src));

         }else if(type_string.compare("DotInc") == 0){
                Matrix A = get_signal(tokens[1]);
                Matrix X = get_signal(tokens[2]);
                Matrix Y = get_signal(tokens[3]);

                add_op(new DotInc(A, X, Y));

         }else if(type_string.compare("ElementwiseInc") == 0){
                Matrix A = get_signal(tokens[1]);
                Matrix X = get_signal(tokens[2]);
                Matrix Y = get_signal(tokens[3]);

                add_op(new ElementwiseInc(A, X, Y));

         }else if(type_string.compare("LIF") == 0){
                int num_neurons = boost::lexical_cast<int>(tokens[1]);
                dtype tau_ref = boost::lexical_cast<dtype>(tokens[2]);
                dtype tau_rc = boost::lexical_cast<dtype>(tokens[3]);
                dtype dt = boost::lexical_cast<dtype>(tokens[4]);

                Matrix J = get_signal(tokens[5]);
                Matrix output = get_signal(tokens[6]);

                add_op(new SimLIF(num_neurons, tau_ref, tau_rc, dt, J, output));

         }else if(type_string.compare("LIFRate") == 0){
                int num_neurons = boost::lexical_cast<int>(tokens[1]);
                dtype tau_ref = boost::lexical_cast<dtype>(tokens[2]);
                dtype tau_rc = boost::lexical_cast<dtype>(tokens[3]);

                Matrix J = get_signal(tokens[4]);
                Matrix output = get_signal(tokens[5]);

                add_op(new SimLIFRate(num_neurons, tau_ref, tau_rc, J, output));

         }else if(type_string.compare("RectifiedLinear") == 0){
                int num_neurons = boost::lexical_cast<int>(tokens[1]);

                Matrix J = get_signal(tokens[2]);
                Matrix output = get_signal(tokens[3]);

                add_op(new SimRectifiedLinear(num_neurons, J, output));

         }else if(type_string.compare("Sigmoid") == 0){
                int num_neurons = boost::lexical_cast<int>(tokens[1]);
                dtype tau_ref = boost::lexical_cast<dtype>(tokens[2]);

                Matrix J = get_signal(tokens[3]);
                Matrix output = get_signal(tokens[4]);

                add_op(new SimSigmoid(num_neurons, tau_ref, J, output));

         }else if(type_string.compare("LinearFilter") == 0){

                Matrix input = get_signal(tokens[1]);
                Matrix output = get_signal(tokens[2]);

                BaseMatrix* numerator = extract_list(tokens[3]);
                BaseMatrix* denominator = extract_list(tokens[4]);

                add_op(new Synapse(input, output, numerator, denominator));

         }else if(type_string.compare("MpiSend") == 0){
                int dst = boost::lexical_cast<int>(tokens[1]);
                int tag = boost::lexical_cast<int>(tokens[2]);
                key_type signal_key = boost::lexical_cast<key_type>(tokens[3]);
                BaseMatrix* content = get_base_signal(signal_key);

                add_mpi_send(new MPISend(dst, tag, content));

         }else if(type_string.compare("MpiRecv") == 0){
                int src = boost::lexical_cast<int>(tokens[1]);
                int tag = boost::lexical_cast<int>(tokens[2]);
                key_type signal_key = boost::lexical_cast<key_type>(tokens[3]);
                BaseMatrix* content = get_base_signal(signal_key);

                add_mpi_recv(new MPIRecv(src, tag, content));

        }else{
            stringstream ss;
            ss << "Received an operator type that we can't handle: " << type_string;
            throw runtime_error(ss.str());
        }
    }catch(const boost::bad_lexical_cast& e){
        cout << "Caught bad lexical cast while extracting operator from string "
                "with error " << e.what() << endl;
        terminate();
    }
}

void MpiSimulatorChunk::add_mpi_send(MPISend* mpi_send){
    operator_list.push_back(mpi_send);

    mpi_sends.push_back(mpi_send);
}

void MpiSimulatorChunk::add_mpi_recv(MPIRecv* mpi_recv){
    operator_list.push_back(mpi_recv);

    mpi_recvs.push_back(mpi_recv);
}

void MpiSimulatorChunk::add_probe(key_type probe_key, string signal_string, dtype period){
    Matrix signal = get_signal(signal_string);
    Probe* probe = new Probe(signal, period);
    probe_map[probe_key] = probe;
}

void MpiSimulatorChunk::add_probe(key_type probe_key, Probe* probe){
    probe_map[probe_key] = probe;
}

BaseMatrix* MpiSimulatorChunk::extract_list(string s){
    // Remove surrounding square brackets
    boost::trim_if(s, boost::is_any_of("[]"));

    vector<string> tokens;
    boost::split(tokens, s, boost::is_any_of(","));

    int length = tokens.size();
    int i = 0;
    vector<string>::const_iterator it;

    BaseMatrix* ret = new BaseMatrix(length, 1);
    try{
        for(it = tokens.begin(); it != tokens.end(); ++it){
            string val = *it;
            boost::trim(val);
            (*ret)(i, 0) = boost::lexical_cast<dtype>(val);
            i++;
        }
    }catch(const boost::bad_lexical_cast& e){
        cout << "Caught bad lexical cast while extracting list "
                "with error " << e.what() << endl;
        terminate();
    }

    return ret;
}

string MpiSimulatorChunk::to_string() const{
    stringstream out;

    out << "<MpiSimulatorChunk" << endl;
    out << "    Label: " << label << endl;

    map<key_type, BaseMatrix*>::const_iterator signal_it = signal_map.begin();
    out << "** Signals: **" << endl;
    for(; signal_it != signal_map.end(); signal_it++){
        out << "Key: " << signal_it->first << endl;
        out << "Label: " << signal_labels.at(signal_it->first);
        out << "Matrix: " << *(signal_it->second) << endl;
    }
    out << endl;

    map<key_type, Probe*>::const_iterator probe_it = probe_map.begin();
    out << "** Probes: **" << endl;
    for(; probe_it != probe_map.end(); probe_it++){
        out << "Key: " << probe_it->first << endl;
        out << "Probe: " << *(probe_it->second) << endl;
    }
    out << endl;

    list<Operator*>::const_iterator it;
    out << "** Operators: **" << endl;
    for(it = operator_list.begin(); it != operator_list.end(); ++it){
        out << (**it) << endl;
    }
    out << endl;

    vector<MPISend*>::const_iterator send_it = mpi_sends.begin();
    out << "** MPISends: **" << endl;
    for(; send_it != mpi_sends.end(); ++send_it){
        out << "Value: " << **send_it << endl;
    }

    vector<MPIRecv*>::const_iterator recv_it = mpi_recvs.begin();
    out << "** MPIRecvs: **" << endl;
    for(; recv_it != mpi_recvs.end(); ++recv_it){
        out << "Value: " << **recv_it << endl;
    }

    return out.str();
}
