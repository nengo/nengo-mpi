#ifndef NENGO_MPI_OPERATOR_HPP
#define NENGO_MPI_OPERATOR_HPP

#include <vector>
#include <string>
#include <sstream>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/circular_buffer.hpp>

#include "debug.hpp"

using namespace std;

typedef double floattype;

typedef boost::numeric::ublas::vector<floattype> Vector;
typedef boost::numeric::ublas::matrix<floattype> Matrix;

// A vector/matrix whose elements are all the same
typedef boost::numeric::ublas::scalar_vector<floattype> ScalarVector;
typedef boost::numeric::ublas::scalar_matrix<floattype> ScalarMatrix;

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

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){}
};

class Reset: public Operator{

public:
    Reset(){};
    Reset(Matrix* dst, floattype value);
    string classname() const { return "Reset"; }

    void operator() ();
    virtual string to_string() const;

protected:
    Matrix* dst;
    Matrix dummy;
    floattype value;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){

        dbg("Serializing: " << classname());

        ar & boost::serialization::base_object<Operator>(*this);
        ar & dst;
        ar & dummy;
        ar & value;
    }
};

class Copy: public Operator{
public:
    Copy(){};
    Copy(Matrix* dst, Matrix* src);
    string classname() const { return "Copy"; }

    void operator()();
    virtual string to_string() const;

protected:
    Matrix* dst;
    Matrix* src;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){

        dbg("Serializing: " << classname());

        ar & boost::serialization::base_object<Operator>(*this);
        ar & dst;
        ar & src;
    }
};

// Increment signal Y by dot(A,X)
class DotInc: public Operator{
public:
    DotInc(){};
    DotInc(Matrix* A, Matrix* X, Matrix* Y);
    string classname() const { return "DotInc"; }

    void operator()();
    virtual string to_string() const;

protected:
    Matrix* A;
    Matrix* X;
    Matrix* Y;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){

        dbg("Serializing: " << classname());

        ar & boost::serialization::base_object<Operator>(*this);
        ar & A;
        ar & X;
        ar & Y;
    }
};


class ElementwiseInc: public Operator{
public:
    ElementwiseInc(){};
    ElementwiseInc(Matrix* A, Matrix* X, Matrix* Y);
    string classname() const { return "ElementwiseInc"; }

    void operator()();
    virtual string to_string() const;

protected:
    Matrix* A;
    Matrix* X;
    Matrix* Y;
    int size;
    bool scalar;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){

        dbg("Serializing: " << classname());

        ar & boost::serialization::base_object<Operator>(*this);
        ar & A;
        ar & X;
        ar & Y;
        ar & size;
        ar & scalar;
    }
};

class Filter: public Operator{

public:
    Filter(){};
    Filter(Matrix* input, Matrix* output, Matrix* numer, Matrix* denom);
    string classname() const { return "Filter"; }

    void operator()();
    virtual string to_string() const;

protected:
    Matrix* input;
    Matrix* output;
    Matrix* numer;
    Matrix* denom;

    vector< boost::circular_buffer<floattype> > x;
    vector< boost::circular_buffer<floattype> > y;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void save(Archive & ar, const unsigned int version) const{

        dbg("Serializing: " << classname());

        ar & boost::serialization::base_object<Operator>(*this);

        ar & input;
        ar & output;
        ar & numer;
        ar & denom;
    }

    template<class Archive>
    void load(Archive & ar, const unsigned int version){
        dbg("Serializing: " << classname());

        ar & boost::serialization::base_object<Operator>(*this);

        ar & input;
        ar & output;
        ar & numer;
        ar & denom;

        // Circular buffers do not have serialization method, but they
        // should be empty anyway
        for(int i = 0; i < input->size1(); i++){
            x.push_back(boost::circular_buffer<floattype>(numer->size1()));
            y.push_back(boost::circular_buffer<floattype>(denom->size1()));
        }
    }

    // macro to use separate save/load functions for serialization
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

class SimLIF: public Operator{
public:
    SimLIF(){};
    SimLIF(int n_neuron, floattype tau_rc, floattype tau_ref, floattype dt, Matrix* J, Matrix* output);
    string classname() const { return "SimLIF"; }

    void operator()();
    virtual string to_string() const;

protected:
    floattype dt;
    floattype dt_inv;
    floattype tau_rc;
    floattype tau_ref;
    int n_neurons;

    Matrix* J;
    Matrix* output;

    Matrix voltage;
    Matrix refractory_time;

    Matrix dt_vec;
    Matrix mult;
    Matrix dV;
    Matrix one;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){

        dbg("Serializing: " << classname());

        ar & boost::serialization::base_object<Operator>(*this);
        ar & dt;
        ar & dt_inv;
        ar & tau_rc;
        ar & tau_ref;
        ar & n_neurons;

        ar & J;
        ar & output;

        ar & voltage;
        ar & refractory_time;

        ar & dt_vec;
        ar & mult;
        ar & dV;
        ar & one;
    }
};

class SimLIFRate: public Operator{

public:
    SimLIFRate(){};
    SimLIFRate(int n_neurons, floattype tau_rc, floattype tau_ref, floattype dt, Matrix* J, Matrix* output);
    string classname() const { return "SimLIFRate"; }

    void operator()();
    virtual string to_string() const;

protected:
    floattype dt;
    floattype tau_rc;
    floattype tau_ref;
    int n_neurons;

    Matrix j;
    Matrix one;

    Matrix* J;
    Matrix* output;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){

        dbg("Serializing: " << classname());

        ar & boost::serialization::base_object<Operator>(*this);
        ar & dt;
        ar & tau_rc;
        ar & tau_ref;
        ar & n_neurons;

        ar & j;
        ar & one;

        ar & J;
        ar & output;
    }
};

#endif