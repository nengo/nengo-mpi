#include "probe.hpp"

template<class T> 
Probe<T>::Probe(T* signal)
    :signal(signal){
}

template<class T> 
Probe<T>::gather(){
    T* new_signal = new T();
    new_signal = signal;
    data.push_back(new_signal);
}
