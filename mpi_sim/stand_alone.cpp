#include<iostream>

#include "simulator.hpp"

using namespace std;

int main(int argc, char **argv){
    if(argc > 1){
        if(argc > 2){
            MpiSimulator mpi_sim(argv[1]);
            int num_steps = boost::lexical_cast<int>(argv[2]);
            mpi_sim.run_n_steps(num_steps, true);
            key_type key = mpi_sim.get_probe_keys()[0];
            vector<BaseMatrix*> data = mpi_sim.get_probe_data(key);
            vector<BaseMatrix*>::iterator it;
            int i = 0;
            for(it=data.begin(); it < data.end(); it++, i++){
                cout << i << endl;
                cout << **it << endl;
            }
        }else{
            cout << "Please specify a simulation length" << endl;
        }
    }else{
        cout << "Please specify a file to load" << endl;
    }

    return 0;
}