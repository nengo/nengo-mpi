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

// A vector whose elements are all the same
typedef boost::numeric::ublas::scalar_vector<floattype> ScalarVector;

// Current implementation: Each Operator is essentially a closure.
// At run time, these closures will be in an array, and we simply call
// them sequentially. The order they are called in will be determined by python.
// The () operator is a virtual function, which comes with some overhead.
// Future optimizations should look at another scheme, either function pointers
// or, ideally, finding some way to make these functions
// non-pointers and non-virtual.

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
    Reset(Vector* dst, floattype value);
    string classname() const { return "Reset"; }

    void operator() ();
    virtual string to_string() const;

protected:
    Vector* dst;
    Vector dummy;
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
    Copy(Vector* dst, Vector* src);
    string classname() const { return "Copy"; }

    void operator()();
    virtual string to_string() const;

protected:
    Vector* dst;
    Vector* src;

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
class DotIncMV: public Operator{
public:
    DotIncMV(){};
    DotIncMV(Matrix* A, Vector* X, Vector* Y);
    string classname() const { return "DotIncMV"; }

    void operator()();
    virtual string to_string() const;

protected:
    Matrix* A;
    Vector* X;
    Vector* Y;

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

// Increment signal Y by dot(A,X)
class DotIncVV: public Operator{
public:
    DotIncVV(){};
    DotIncVV(Vector* A, Vector* X, Vector* Y);
    string classname() const { return "DotIncVV"; }

    void operator()();
    virtual string to_string() const;

protected:
    Vector* A;
    Vector* X;
    Vector* Y;
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

// Sets Y <- dot(A, X) + B * Y
class ProdUpdate: public Operator{
public:
    ProdUpdate(){};
    ProdUpdate(Vector* B, Vector* Y);
    string classname() const { return "ProdUpdate"; }

    void operator()();
    virtual string to_string() const;

protected:
    Vector* B;
    Vector* Y;
    int size;
    bool scalar;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){

        dbg("Serializing: " << classname());

        ar & boost::serialization::base_object<Operator>(*this);
        ar & B;
        ar & Y;
        ar & size;
        ar & scalar;
    }
};

class Filter: public Operator{

public:
    Filter(){};
    Filter(Vector* input, Vector* output, Vector* numer, Vector* denom);
    string classname() const { return "Filter"; }

    void operator()();
    virtual string to_string() const;

protected:
    Vector* input;
    Vector* output;
    Vector* numer;
    Vector* denom;

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
        for(int i = 0; i < input->size(); i++){
            x.push_back(boost::circular_buffer<floattype>(numer->size()));
            y.push_back(boost::circular_buffer<floattype>(denom->size()));
        }
    }

    // macro to use separate save/load functions
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

class SimLIF: public Operator{
public:
    SimLIF(){};
    SimLIF(int n_neuron, floattype tau_rc, floattype tau_ref, floattype dt, Vector* J, Vector* output);
    string classname() const { return "SimLIF"; }

    void operator()();
    virtual string to_string() const;

protected:
    floattype dt;
    floattype dt_inv;
    floattype tau_rc;
    floattype tau_ref;
    int n_neurons;

    Vector* J;
    Vector* output;

    Vector voltage;
    Vector refractory_time;

    Vector dt_vec;
    Vector mult;
    Vector dV;
    Vector one;

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
    SimLIFRate(int n_neurons, floattype tau_rc, floattype tau_ref, floattype dt, Vector* J, Vector* output);
    string classname() const { return "SimLIFRate"; }

    void operator()();
    virtual string to_string() const;

protected:
    floattype dt;
    floattype tau_rc;
    floattype tau_ref;
    int n_neurons;

    Vector j;
    Vector one;

    Vector* J;
    Vector* output;

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