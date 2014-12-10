#include "chunk.hpp"

MpiSimulatorChunk::MpiSimulatorChunk()
    :time(0.0), dt(0.001), n_steps(0) {
}

MpiSimulatorChunk::MpiSimulatorChunk(double dt)
    :time(0.0), dt(dt), n_steps(0){
}

void MpiSimulatorChunk::run_n_steps(int steps){

    map<key_type, Probe<Matrix>*>::iterator probe_it;
    for(probe_it = probe_map.begin(); probe_it != probe_map.end(); ++probe_it){
        probe_it->second->init_for_simulation(steps);
    }

    for(unsigned step = 0; step < steps; ++step){
        if(step % 100 == 0){
            cout << "Starting step: " << step << endl;
        }

        list<Operator*>::const_iterator it;
        for(it = operator_list.begin(); it != operator_list.end(); ++it){
            run_dbg("Before calling: " << **it << endl);

            //Call the operator
            (**it)();

            run_dbg("After calling: " << **it << endl);
        }

        map<key_type, Probe<Matrix>*>::iterator probe_it;
        for(probe_it = probe_map.begin(); probe_it != probe_map.end(); ++probe_it){
            //Call the operator
            probe_it->second->gather(step);
            run_dbg("After gathering: " << *(probe_it->second) << endl);
        }

        run_dbg(print_signal_pointers());
        time += dt;
        n_steps++;
    }
}

void MpiSimulatorChunk::add_signal(key_type key, string label, Matrix* sig){
    signal_map[key] = sig;
    signal_labels[key] = label;
}

void MpiSimulatorChunk::add_probe(key_type key, Probe<Matrix>* probe){
    probe_map[key] = probe;
}

Matrix* MpiSimulatorChunk::get_signal(key_type key){
    try{
        Matrix* mat = signal_map.at(key);
        return mat;
    }catch(const out_of_range& e){
        cerr << "Error accessing MpiSimulatorChunk :: signal with key " << key << endl;
        throw e;
    }
}

Probe<Matrix>* MpiSimulatorChunk::get_probe(key_type key){
    try{
        Probe<Matrix>* probe = probe_map.at(key);
        return probe;
    }catch(const out_of_range& e){
        cerr << "Error accessing MpiSimulatorChunk :: probe with key " << key << endl;
        throw e;
    }
}

void MpiSimulatorChunk::add_operator(Operator *op){
    operator_list.push_back(op);
}

void MpiSimulatorChunk::add_mpi_send(MPISend* mpi_send){
    operator_list.push_back(mpi_send);

    mpi_sends[mpi_send->tag] = mpi_send;
}

void MpiSimulatorChunk::add_mpi_recv(MPIRecv* mpi_recv){
    operator_list.push_back(mpi_recv);

    mpi_recvs[mpi_recv->tag] = mpi_recv;
}

void MpiSimulatorChunk::add_mpi_wait(MPIWait* mpi_wait){
    operator_list.push_back(mpi_wait);

    mpi_waits[mpi_wait->tag] = mpi_wait;
}

void MpiSimulatorChunk::fix_mpi_waits(){

    map<int, MPIWait*>::iterator wait_it;

    for(wait_it = mpi_waits.begin(); wait_it != mpi_waits.end(); ++wait_it){
        try{
            MPISend* send = mpi_sends.at(wait_it->first);
            wait_it->second->request = send->get_request_pointer();
        }catch(const out_of_range& e){
            try{
                MPIRecv* recv = mpi_recvs.at(wait_it->first);
                wait_it->second->request = recv->get_request_pointer();
            }catch(const out_of_range& e){
                cerr << "Found MPIWait with no matching operator. tag = "
                     << wait_it->first << "." << endl;
                throw e;
            }
        }
    }
}

MPIWait* MpiSimulatorChunk::find_wait(int tag){

    MPIWait* mpi_wait;

    try{
        mpi_wait = mpi_waits.at(tag);
    }catch(const out_of_range& e){
        stringstream error;
        error << "MPIWait object with tag " << tag << " does not exist.";
        throw invalid_argument(error.str());
    }

    return mpi_wait;
}

string MpiSimulatorChunk::to_string() const{
    stringstream out;

    out << "<MpiSimulatorChunk" << endl;

    map<key_type, Matrix*>::const_iterator signal_it = signal_map.begin();

    out << "** Matrices: **" << endl;
    for(; signal_it != signal_map.end(); signal_it++){
        out << "Key: " << signal_it->first << endl;
        out << "Label: " << signal_labels.at(signal_it->first);
        out << "Matrix: " << *(signal_it->second) << endl;
    }
    out << endl;

    map<key_type, Probe<Matrix>*>::const_iterator probe_it = probe_map.begin();

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

    return out.str();
}

string MpiSimulatorChunk::print_maps(){
    stringstream out;

    map<int, MPISend*>::iterator send_it;
    out << "SENDS" << endl;
    for(send_it = mpi_sends.begin(); send_it != mpi_sends.end(); ++send_it){
        out << "key: " << send_it->first <<  ", value: " << *(send_it->second) << endl;
    }

    map<int, MPIRecv*>::iterator recv_it;
    out << "RECVS" << endl;
    for(recv_it = mpi_recvs.begin(); recv_it != mpi_recvs.end(); ++recv_it){
        out << "key: " << recv_it->first <<  ", value: " << *(recv_it->second) << endl;
    }

    return out.str();
}

string MpiSimulatorChunk::print_signal_pointers(){
    stringstream out;

    out << "Printing signal pointers: " << endl;
    map<key_type, Matrix*>::iterator signal_it;
    int count = 0;
    for(signal_it = signal_map.begin(); signal_it != signal_map.end(); ++signal_it){
        out << "Count: " << count << ", pointer: " << signal_it->second << endl;
        out << "Label: " << signal_labels.at(signal_it->first) << endl;
        out << "Value: " << *(signal_it->second) << endl << endl;
        count++;
    }

    return out.str();
}

string MpiSimulatorChunk::print_signals(){
    stringstream out;

    out << "Printing signals: " << endl;
    map<key_type, Matrix*>::iterator signal_it;
    int index = 0;
    for(signal_it = signal_map.begin(); signal_it != signal_map.end(); ++signal_it){
        out << "Index: " << index << endl;
        out << "Label: " << signal_labels.at(signal_it->first) << endl;
        out << "Value: " << *(signal_it->second) << endl << endl;
        index++;
    }

    return out.str();
}
