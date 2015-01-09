#include "probe.hpp"

Probe::Probe(Matrix signal, float period)
:signal(signal), period(period), index(0), step_offset(0){
}

void Probe::init_for_simulation(int n_steps){

    if(!data.empty()){
        stringstream error;
        error << "Probe must be empty before it can be initialized. "
              << "Call Probe.clear first";

        throw logic_error(error.str());
    }

    step_offset = index;
    index = 0;

    int num_samples = (int) floor(n_steps / period);

    data.reserve(num_samples);

    for(unsigned i = 0; i < num_samples; i++){
        data.push_back(new BaseMatrix(signal));
    }
}

void Probe::gather(int step){
    if(fmod(step + step_offset, period) < 1){
        *(data[index]) = signal;
        index++;
    }
}

vector<BaseMatrix*> Probe::get_data(){
    return data;
}

void Probe::clear(bool del){
    index = 0;

    if(del){
        vector<BaseMatrix*>::iterator it;
        for(it = data.begin(); it < data.end(); it++){
            delete *it;
        }
    }

    data.clear();
}

void Probe::reset(){
    clear(true);
    step_offset = 0;
}

string Probe::to_string() const{
    stringstream out;
    out << "Probe:" << endl;
    out << "period: " << period << endl;
    out << "size: " << data.size() << endl;
    out << "signal: " << signal << endl;
    out << "index: " << index << endl;
    out << "step_offset: " << step_offset << endl;

    out << "data: " << endl;
    for(unsigned i = 0; i < data.size(); i++){
         out << "index: " << i << ", signal: " << *(data[i]) << endl;
    }

    return out.str();
}

