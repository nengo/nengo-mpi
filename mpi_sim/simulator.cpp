#include "simulator.hpp"

MpiSimulatorChunk* MpiSimulator::add_chunk(){
    MpiSimulatorChunk* chunk = new MpiSimulatorChunk();
    chunks.push_back(chunk);
    return chunk;
}

MpiSimulator::MpiSimulator(){
}

void MpiSimulator::finalize(){
    // Use MPI DPM to setup processes on other nodes
    // Then pass the chunks to those nodes.
    mpi_interface.send_chunks(chunks);
}

void MpiSimulator::run_n_steps(int steps){
    cout << "In MPI run_n_steps: " << steps << endl;
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

    list<MpiSimulatorChunk*>::const_iterator it;
    for(it = chunks.begin(); it != chunks.end(); ++it){
        out << (**it) << endl;
    }

    return out.str();
}
