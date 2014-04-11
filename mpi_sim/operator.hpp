#ifndef NENGO_MPI_OPERATOR_HPP
#define NENGO_MPI_OPERATOR_HPP

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace std;

typedef double floattype;

typedef boost::numeric::ublas::vector<floattype> Vector;
typedef boost::numeric::ublas::scalar_vector<floattype> ScalarVector;
typedef boost::numeric::ublas::matrix<floattype> Matrix;

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

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){}
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

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){
        ar & boost::serialization::base_object<Operator>(*this);    
        ar & dst;
        ar & dummy;
        ar & value;
    }
};

class Copy: public Operator{
public:
    Copy(Vector* dst, Vector* src);
    void operator()();
    virtual void print(ostream &out) const;

protected:
    Vector* dst;
    Vector* src;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){
        ar & boost::serialization::base_object<Operator>(*this);    
        ar & dst;
        ar & src;
    }
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

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){
        ar & boost::serialization::base_object<Operator>(*this);    
        ar & A;
        ar & X;
        ar & Y;
    }
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

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){
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
    ProdUpdate(Vector* B, Vector* Y);
    void operator()();
    virtual void print(ostream &out) const;

protected:
    Vector* B;
    Vector* Y;
    int size;
    bool scalar;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){
        ar & boost::serialization::base_object<Operator>(*this);    
        ar & B;
        ar & Y;
        ar & size;
        ar & scalar;
    }
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

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){
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

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){
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