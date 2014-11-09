#include "simulator.hpp"

// The first chunk is kept on the master process,
// subsequent chunks will be sent out to other processes.
MpiSimulatorChunk* MpiSimulator::add_chunk(){
    MpiSimulatorChunk* chunk = new MpiSimulatorChunk();

    if(master_chunk == NULL){
        master_chunk = chunk;
    }else{
        remote_chunks.push_back(chunk);
    }

    return chunk;
}

MpiSimulator::MpiSimulator():
    master_chunk(NULL){
}

void MpiSimulator::finalize(){
    // Use MPI DPM to setup processes on other nodes
    // Then pass the chunks to those nodes.
    mpi_interface.initialize_chunks(master_chunk, remote_chunks);
}

void MpiSimulator::run_n_steps(int steps){
    mpi_interface.start_simulation(steps);
}

void MpiSimulator::write_to_file(string filename){
    ofstream ofs(filename);

    boost::archive::text_oarchive oa(ofs);
    oa << *this;
}

void MpiSimulator::read_from_file(string filename){
    ifstream ifs(filename);

    boost::archive::text_iarchive ia(ifs);
    ia >> *this;
}

string MpiSimulator::to_string() const{
    stringstream out;

    out << "<MpiSimulator" << endl;

    out << "**master chunk**" << endl;
    out << master_chunk << endl;

    out << "**remote chunks**" << endl;
    list<MpiSimulatorChunk*>::const_iterator it;
    for(it = remote_chunks.begin(); it != remote_chunks.end(); ++it){
        out << (**it) << endl;
    }

    return out.str();
}
