#include "mpi_chunk.hpp"

MpiSimulatorChunk::MpiSimulatorChunk(int rank, int n_processors, MPI_Comm comm, bool mpi_merged)
:time(0.0), dt(0.001), n_steps(0), rank(rank), n_processors(n_processors), comm(comm), mpi_merged(mpi_merged){
    stringstream ss;
    ss << "Chunk " << rank;
    label = ss.str();
}

void MpiSimulatorChunk::from_file(string filename){
    // Use parallel property lists
    hid_t file_plist = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(file_plist, comm, MPI_INFO_NULL);

    hid_t read_plist = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(read_plist, H5FD_MPIO_INDEPENDENT);

    from_file(filename, file_plist, read_plist);

    H5Pclose(file_plist);
    H5Pclose(read_plist);
}

void MpiSimulatorChunk::finalize_build(){
    sim_log = unique_ptr<SimulationLog>(
        new MpiSimulationLog(n_processors, rank, probe_info, dt, comm));

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
            stable_sort(content_prime.begin(), content_prime.end(), compare_first);

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

void MpiSimulatorChunk::add_op(OpSpec op_spec){
    try{
        SimulatorChunk::add_op(op_spec);
    }catch(const runtime_error& e){
        try{
            string type_string = op_spec.type_string;
            vector<string>& args = op_spec.arguments;
            float index = op_spec.index;

            if(type_string.compare("MpiSend") == 0){

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

            throw boost::bad_lexical_cast(msg.str());
        }
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
