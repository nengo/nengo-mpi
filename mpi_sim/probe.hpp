
#ifndef NENGO_MPI_PROBE_HPP
#define NENGO_MPI_PROBE_HPP

#include <vector>

#include "operator.hpp"

template<class T> class Probe;

template<class T>
ostream& operator << (ostream &out, const Probe<T> &probe);

template<class T>
class Probe {
public:
    Probe(){};
    Probe(T* signal, float period);
    void init_for_simulation(int n_steps);
    void gather(int n_steps);
    vector<T*>* get_data();
    void clear();
    void reset();
    friend ostream& operator << <> (ostream &out, const Probe<T> &probe);

protected:
    vector<T*>* data;
    T* signal;
    float period;
    int index;
    int step_offset;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){
        ar & data;
        ar & signal;
        ar & period;
        ar & index;
        ar & step_offset;
    }
};

template<class T>
Probe<T>::Probe(T* signal, float period)
:signal(signal), period(period), index(0), step_offset(0){
    data = new vector<T*>();
}

template<class T>
void Probe<T>::init_for_simulation(int n_steps){

    if(!data->empty()){
        stringstream error;
        error << "Probe must be empty before it can be initialized. "
              << "Call Probe.clear first";

        throw logic_error(error.str());
    }

    step_offset = index;
    index = 0;

    int num_samples = (int) floor(n_steps / period);

    data->reserve(num_samples);

    for(unsigned i = 0; i < num_samples; i++){
        data->push_back(new T(*signal));
    }
}

template<class T>
void Probe<T>::gather(int step){
    if(fmod(step + step_offset,  period) < 1){
        *((*data)[index]) = *signal;
        index++;
    }
}

template<class T>
vector<T*>* Probe<T>::get_data(){
    vector<T*>* temp = data;
    data = new vector<T*>();
    index = 0;

    return temp;
}

template<class T>
void Probe<T>::clear(){
    data->clear();
    index = 0;
}

template<class T>
void Probe<T>::reset(){
    clear();
    step_offset = 0;
}

template<class T>
ostream& operator << (ostream &out, const Probe<T> &probe){
    out << "Probe:" << endl;

    out << "Period: " << probe.period << endl;

    out << "Size: " << (probe.data)->size() << endl;

    out << "Probed signal: " << *(probe.signal) << endl;

    out << "Index: " << probe.index << endl;

    out << "Step offset: " << probe.step_offset << endl;

    out << "Data: " << endl;
    for(unsigned i = 0; i < (probe.data)->size(); i++){
         out << "index: " << i << ", signal: " << *((*(probe.data))[i]) << endl;
    }

    return out;
}

#endif
