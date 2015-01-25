#ifndef NENGO_MPI_OPERATOR_HPP
#define NENGO_MPI_OPERATOR_HPP

#include <vector>
#include <string>
#include <sstream>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/circular_buffer.hpp>

#include "debug.hpp"

using namespace std;

namespace ublas = boost::numeric::ublas;

typedef double floattype;

typedef ublas::matrix<floattype> BaseMatrix;
typedef ublas::matrix_slice<BaseMatrix> Matrix;
typedef ublas::scalar_matrix<floattype> ScalarMatrix;

// Current implementation: Each Operator is essentially a closure.
// At run time, these closures will be in an array, and we simply call
// them sequentially. The order they are called in will be determined by python.
// The () operator is a virtual function, which comes with some overhead.
// Future optimizations should look at another scheme, either function pointers
// or, ideally, finding some way to make these functions
// non-pointers and non-virtual.
//
// All matrices that are basically vectors are generally assumed to be column vectors.

class Operator{

public:
    string classname() const { return "Operator"; }
    virtual void operator() () = 0;
    virtual string to_string() const = 0;

    friend ostream& operator << (ostream &out, const Operator &op){
        out << op.to_string();
        return out;
    }
};

class Reset: public Operator{

public:
    Reset(Matrix dst, floattype value);
    string classname() const { return "Reset"; }

    void operator()();
    virtual string to_string() const;

protected:
    Matrix dst;
    ScalarMatrix dummy;
    floattype value;

};

class Copy: public Operator{
public:
    Copy(Matrix dst, Matrix src);
    string classname() const { return "Copy"; }

    void operator()();
    virtual string to_string() const;

protected:
    Matrix dst;
    Matrix src;
};

// Increment signal Y by dot(A,X)
class DotInc: public Operator{
public:
    DotInc(Matrix A, Matrix X, Matrix Y);
    string classname() const { return "DotInc"; }

    void operator()();
    virtual string to_string() const;

protected:
    Matrix A;
    Matrix X;
    Matrix Y;
};


class ElementwiseInc: public Operator{
public:
    ElementwiseInc(Matrix A, Matrix X, Matrix Y);
    string classname() const { return "ElementwiseInc"; }

    void operator()();
    virtual string to_string() const;

protected:
    Matrix A;
    Matrix X;
    Matrix Y;

    bool broadcast;

    // Strides are 0 or 1, to support broadcasting
    int A_row_stride;
    int A_col_stride;

    int X_row_stride;
    int X_col_stride;
};

class Synapse: public Operator{

public:
    Synapse(Matrix input, Matrix output, BaseMatrix* numer, BaseMatrix* denom);
    string classname() const { return "Synapse"; }

    void operator()();
    virtual string to_string() const;

protected:
    Matrix input;
    Matrix output;

    // TODO: make these into normal Matrices referring to BaseMatrices stored
    // inside the chunk
    BaseMatrix* numer;
    BaseMatrix* denom;

    vector< boost::circular_buffer<floattype> > x;
    vector< boost::circular_buffer<floattype> > y;
};

class SimLIF: public Operator{
public:
    SimLIF(int n_neuron, floattype tau_rc, floattype tau_ref, floattype dt, Matrix J, Matrix output);
    string classname() const { return "SimLIF"; }

    void operator()();
    virtual string to_string() const;

protected:
    floattype dt;
    floattype dt_inv;
    floattype tau_rc;
    floattype tau_ref;
    int n_neurons;

    Matrix J;
    Matrix output;

    BaseMatrix voltage;
    BaseMatrix refractory_time;

    ScalarMatrix dt_vec;
    BaseMatrix mult;
    BaseMatrix dV;
    ScalarMatrix one;
};

class SimLIFRate: public Operator{

public:
    SimLIFRate(int n_neurons, floattype tau_rc, floattype tau_ref, Matrix J, Matrix output);
    string classname() const { return "SimLIFRate"; }

    void operator()();
    virtual string to_string() const;

protected:
    floattype tau_rc;
    floattype tau_ref;
    int n_neurons;

    BaseMatrix j;
    ScalarMatrix one;

    Matrix J;
    Matrix output;
};

class SimRectifiedLinear: public Operator{
public:
    SimRectifiedLinear(int n_neurons, Matrix J, Matrix output);
    string classname() const { return "SimRectifiedLinear"; }

    void operator()();
    virtual string to_string() const;

protected:
    int n_neurons;

    Matrix J;
    Matrix output;
};

class SimSigmoid: public Operator{
public:
    SimSigmoid(int n_neurons, float tau_ref, Matrix J, Matrix output);
    string classname() const { return "SimSigmoid"; }

    void operator()();
    virtual string to_string() const;

protected:
    int n_neurons;
    float tau_ref;
    float tau_ref_inv;

    Matrix J;
    Matrix output;
};


#endif