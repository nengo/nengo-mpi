
#ifndef NENGO_MPI_PROBE_HPP
#define NENGO_MPI_PROBE_HPP

#include <list>

#include "operator.hpp"

template<class T>
class Probe {
public:
    Probe(T*);
    void gather();

protected:
    list<T*> data;
    T* signal;
};

#endif
