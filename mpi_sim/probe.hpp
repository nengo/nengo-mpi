
#ifndef NENGO_MPI_PROBE_HPP
#define NENGO_MPI_PROBE_HPP

#include <list>

#include "operator.hpp"

template<class T> class Probe;

template<class T>
ostream& operator << (ostream &out, const Probe<T> &probe);

template<class T>
class Probe {
public:
    Probe(){};
    Probe(T* signal, int period);
    void gather(int n_steps);
    list<T*> get_data();
    friend ostream& operator << <> (ostream &out, const Probe<T> &probe);

protected:
    list<T*> data;
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
}

template<class T>
void Probe<T>::gather(int n_steps){
    if(n_steps % period == 0){
        T* new_signal = new T();
        *new_signal = *signal;
        data.push_back(new_signal);
    }
}

template<class T>
list<T*> Probe<T>::get_data(){
    return data;
}

template<class T>
ostream& operator << (ostream &out, const Probe<T> &probe){
    out << "Probe:" << endl;
    out << "Signal: " << endl;
    out << *(probe.signal) << endl;
    out << "Points collected: " << endl;
    out << probe.data.size() << endl;
    return out;
}

#endif
