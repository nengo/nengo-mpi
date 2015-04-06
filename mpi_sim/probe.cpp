#include "probe.hpp"

Probe::Probe(SignalView signal, dtype period)
:signal(signal), period(period), data_index(0), time_index(0){
}

void Probe::init_for_simulation(int n_steps, int fe){

    if(!data.empty()){
        stringstream error;
        error << "Probe must be empty before it can be initialized. "
              << "Call Probe.clear first";

        throw logic_error(error.str());
    }

    time_index = data_index;
    data_index = 0;

    buffer.reset();

    flush_every = fe;
    if(flush_every > 0){
        buffer = shared_ptr<dtype>(new dtype[signal.size1() * flush_every]);
    }else{
        buffer = shared_ptr<dtype>(NULL);
    }

    int num_samples = (int) floor(n_steps / period);

    if(flush_every > 0){
        num_samples = min(num_samples, flush_every);
    }

    data.reserve(num_samples);

    for(unsigned i = 0; i < num_samples; i++){
        data.push_back(unique_ptr<BaseSignal>(new BaseSignal(signal)));
    }
}

void Probe::gather(int step){
    if(fmod(step + time_index, period) < 1){
        *(data[data_index]) = signal;
        data_index++;
    }
}

shared_ptr<dtype> Probe::flush_to_buffer(int &n_rows){
    if(flush_every <= 0){
        throw logic_error(
            "Calling flush_to_buffer, but Probe has flush_every <= 0.");
    }

    int n_cols = data[0]->size1();

    int idx = 0;
    for(int i = 0; i < data_index; i++){
        for(int j = 0; j < n_cols; j++){
            buffer.get()[idx] = (*data[i])(j, 0);
            idx++;
        }
    }

    n_rows = data_index;

    data_index = 0;

    return buffer;
}

vector<unique_ptr<BaseSignal>> Probe::harvest_data(){
    auto d = move(data);
    clear();
    return d;
}

void Probe::clear(){
    data_index = 0;
    data.clear();
}

void Probe::reset(){
    clear();
    time_index = 0;
}

string Probe::to_string() const{
    stringstream out;
    out << "Probe:" << endl;
    out << "period: " << period << endl;
    out << "size: " << data.size() << endl;
    out << "signal: " << signal << endl;
    out << "data_index: " << data_index << endl;
    out << "time_index: " << time_index << endl;

    out << "data: " << endl;
    for(unsigned i = 0; i < data.size(); i++){
         out << "index: " << i << ", signal: " << *(data[i]) << endl;
    }

    return out.str();
}

