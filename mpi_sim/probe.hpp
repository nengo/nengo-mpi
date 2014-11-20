
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
    Probe(T* signal, int period);
    void init_for_simulation(int n_steps);
    void gather(int n_steps);
    vector<T*>* get_data();
    void clear();
    friend ostream& operator << <> (ostream &out, const Probe<T> &probe);

protected:
    vector<T*>* data;
    T* signal;
    int period;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){
        ar & data;
        ar & signal;
        ar & period;
    }
};

template<class T>
Probe<T>::Probe(T* signal, int period)
:signal(signal), period(period){
    data = new vector<T*>();
}

template<class T>
void Probe<T>::init_for_simulation(int n_steps){

    // Initialize the probes storage resources
    if(!data->empty()){
        stringstream error;
        error << "Probe must be empty before it can be initialized. "
              << "Call Probe.clear first";

        throw logic_error(error.str());
    }

    int num_samples = n_steps / period;
    data->resize(num_samples, NULL);

    for(unsigned i = 0; i < n_steps / period; i++){
        (*data)[i] = new T(*signal);
    }
}

template<class T>
void Probe<T>::gather(int step){
    if(step % period == 0){
        *((*data)[step / period]) = *signal;
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
    out << "Signal: " << endl;
    out << *(probe.signal) << endl;
    out << "Period: " << probe.period << endl;
    out << "Points collected: " << endl;
    out << (probe.data)->size() << endl;
    return out;
}

#endif
