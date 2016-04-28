#include "probe.hpp"

Probe::Probe(Signal signal, dtype period)
:signal(signal), period(period), data_index(0), time_index(0){

}

void Probe::init_for_simulation(unsigned n_steps, unsigned flush_every_){

    if(!data.empty()){
        stringstream error;
        error << "Probe must be empty before it can be initialized. "
              << "Call Probe.clear first";

        throw logic_error(error.str());
    }

    time_index = data_index;
    data_index = 0;

    buffer.reset();

    flush_every = flush_every_;
    if(flush_every > 0){
        buffer = shared_ptr<dtype>(new dtype[signal.size * flush_every]);
    }else{
        buffer = shared_ptr<dtype>(NULL);
    }

    unsigned n_samples = (unsigned) floor(n_steps / period);

    if(flush_every > 0){
        n_samples = min(n_samples, flush_every);
    }

    data.reserve(n_samples);
    for(unsigned i = 0; i < n_samples; i++){
        data.push_back(signal.deep_copy());
    }
}

void Probe::gather(unsigned step){
    if(fmod(step + time_index, period) < 1){
        data[data_index].fill_with(signal);
        data_index++;
    }
}

shared_ptr<dtype> Probe::flush_to_buffer(unsigned &n_rows){
    if(flush_every <= 0){
        throw logic_error(
            "Calling flush_to_buffer, but Probe has flush_every <= 0.");
    }

    unsigned offset = 0;
    for(unsigned i = 0; i < data_index; i++){
        data[i].copy_to_buffer(buffer.get()+offset);
        offset += data[i].size;
    }

    n_rows = data_index;
    data_index = 0;

    return buffer;
}

vector<Signal> Probe::harvest_data(){
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

    /*
    out << "data: " << endl;
    for(unsigned i = 0; i < data.size(); i++){
         out << "index: " << i << ", signal: " << data[i] << endl;
    }
    */

    return out.str();
}
