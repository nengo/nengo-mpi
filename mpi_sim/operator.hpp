#ifndef NENGO_MPI_OPERATOR_HPP
#define NENGO_MPI_OPERATOR_HPP

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace std;

typedef double floattype;

typedef boost::numeric::ublas::vector<double> Vector;
typedef boost::numeric::ublas::scalar_vector<double> ScalarVector;
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
    virtual void print(ostream &out) const = 0;
    friend ostream& operator << (ostream &out, const Operator &op);
};

class Reset: public Operator{
public:
    Reset(Vector* dst, floattype value);
    void operator() ();
    virtual void print(ostream &out) const;

protected:
    Vector* dst;
    Vector dummy;
    floattype value;
    int size;
};

class Copy: public Operator{
public:
    Copy(Vector* dst, Vector* src);
    void operator()();
    virtual void print(ostream &out) const;

protected:
    Vector* dst;
    Vector* src;
};

// Increment signal Y by dot(A,X)
class DotIncMV: public Operator{
public:
    DotIncMV(Matrix* A, Vector* X, Vector* Y);
    void operator()();
    virtual void print(ostream &out) const;

protected:
    Matrix* A;
    Vector* X;
    Vector* Y;
};

// Increment signal Y by dot(A,X)
class DotIncVV: public Operator{
public:
    DotIncVV(Vector* A, Vector* X, Vector* Y);
    void operator()();
    virtual void print(ostream &out) const;

protected:
    Vector* A;
    Vector* X;
    Vector* Y;
    int size;
    bool scalar;
};

// Sets Y <- dot(A, X) + B * Y
class ProdUpdate: public Operator{
public:
    ProdUpdate(Vector* B, Vector* Y);
    void operator()();
    virtual void print(ostream &out) const;

protected:
    Vector* B;
    Vector* Y;
    int size;
    bool scalar;
};

class SimLIF: public Operator{
public:
    SimLIF(int n_neuron, floattype tau_rc, floattype tau_ref, floattype dt, Vector* J, Vector* output);
    void operator()();
    virtual void print(ostream &out) const;

protected:
    const floattype dt;
    const floattype dt_inv;
    const floattype tau_rc;
    const floattype tau_ref;
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
    SimLIFRate(int n_neurons, floattype tau_rc, floattype tau_ref, floattype dt, Vector* J, Vector* output);
    void operator()();
    virtual void print(ostream &out) const;

protected:
    const floattype dt;
    const floattype tau_rc;
    const floattype tau_ref;
    const int n_neurons;

    Vector j;
    Vector one;

    Vector* J;
    Vector* output;
};

class MPISend: public Operator{
public:
    MPISend();
    void operator()();
    virtual void print(ostream &out) const;
};

class MPIReceive: public Operator{
public:
    MPIReceive();
    void operator()();
    virtual void print(ostream &out) const;
};

#endif