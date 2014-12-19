
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
    friend ostream& operator << <> (ostream &out, const Probe<T> &probe);

protected:
    vector<T*>* data;
    T* signal;
    float period;
    int index;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){
        ar & data;
        ar & signal;
        ar & period;
        ar & index;
    }
};

template<class T>
Probe<T>::Probe(T* signal, float period)
:signal(signal), period(period), index(0){
    data = new vector<T*>();
}

template<class T>
void Probe<T>::init_for_simulation(int n_steps){

    index = 0;

    // Initialize the probes storage resources
    if(!data->empty()){
        stringstream error;
        error << "Probe must be empty before it can be initialized. "
              << "Call Probe.clear first";

        throw logic_error(error.str());
    }

    int num_samples = (int) floor(n_steps / period);
    data->resize(num_samples, NULL);

    for(unsigned i = 0; i < num_samples; i++){
        (*data)[i] = new T(*signal);
    }
}

template<class T>
void Probe<T>::gather(int step){
    if(fmod(step,  period) < 1){
        *((*data)[index]) = *signal;
        index++;
    }
}

template<class T>
vector<T*>* Probe<T>::get_data(){
    return data;
}

template<class T>
void Probe<T>::clear(){
    data->clear();
}

template<class T>
ostream& operator << (ostream &out, const Probe<T> &probe){
    out << "Probe:" << endl;

    out << "Period: " << probe.period << endl;

    out << "Size: " << (probe.data)->size() << endl;

    out << "Probed signal: " << *(probe.signal) << endl;

    out << "Index: " << probe.index << endl;

    out << "Data: " << endl;
    //for(unsigned i = 0; i < (probe.data)->size(); i++){
    //     out << "index: " << i << ", signal: " << *((*(probe.data))[i]) << endl;
    //}

    return out;
}

#endif
