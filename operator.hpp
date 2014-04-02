#ifndef NENGO_MPI_OPERATOR_HPP
#define NENGO_MPI_OPERATOR_HPP

//
//#include <boost/python.hpp>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace boost::numeric::ublas;

typedef boost::numeric::ublas::vector<double> Vector;
typedef boost::numeric::ublas::matrix<double> Matrix;

// Current implementation: Each Operator is essentially a closure.
// At run time, these closures will be in an array, and we simply call
// them sequentially. The order they are called in will be determined by python.
// The () operator is a virtual function, which comes with some overhead. 
// Future optimizations should look at another scheme, either function pointers 
// or, ideally, finding some way to make these functions 
// non-pointers and non-virtual.

class Operator{
public:
    virtual void operator() () = 0;
};

class Reset: public Operator{
public:
    Reset(Vector* dst, float value);
    void operator() ();

private:
    Vector* dst;
    Vector dummy;
    float value;
    int size;
};

class Copy: public Operator{
public:
    Copy(Vector* dst, Vector* src);
    void operator()();

private:
    Vector* dst;
    Vector* src;
};

// Increment signal Y by dot(A,X)
class DotInc: public Operator{
public:
    DotInc(Matrix* A, Vector* X, Vector* Y);
    void operator()();

private:
    Matrix* A;
    Vector* X;
    Vector* Y;
};

// Sets Y <- dot(A, X) + B * Y
class ProdUpdate: public Operator{
public:
    ProdUpdate(Matrix* A, Vector* X, Vector* B, Vector* Y);
    void operator()();

private:
    Matrix* A;
    Vector* X;
    Vector* B;
    Vector* Y;
    int size;
};

//class PyFunc: public Operator{
//public:
//    PyFunc(Vector* output, boost::python::object fn, Vector* time, Vector* input);
//    void operator()();
//};

class SimLIF: public Operator{
public:
    SimLIF(int n_neuron, float tau_rc, float tau_ref, float dt, Vector* J, Vector* output);
    void operator()();

private:
    const float dt;
    const float dt_inv;
    const float tau_rc;
    const float tau_ref;
    const int n_neurons;

    Vector* J;
    Vector* output;

    Vector voltage;
    Vector refractory_time;

    Vector dt_vec;
    Vector mult;
    Vector dV;
    Vector one;
};

class SimLIFRate: public Operator{

public:
    SimLIFRate(int n_neurons, float tau_rc, float tau_ref, float dt, Vector* J, Vector* output);
    void operator()();

private:
    const float dt;
    const float tau_rc;
    const float tau_ref;
    const int n_neurons;

    Vector* J;
    Vector* output;
};

class MPISend: public Operator{
public:
    MPISend();
    void operator()();
};

class MPIReceive: public Operator{
public:
    MPIReceive();
    void operator()();
};

#endif