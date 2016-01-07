#ifndef NENGO_MPI_OPERATOR_HPP
#define NENGO_MPI_OPERATOR_HPP

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <memory>
#include <random>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/storage.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation.hpp>

#include <boost/circular_buffer.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "debug.hpp"


using namespace std;

namespace ublas = boost::numeric::ublas;

typedef double dtype;

typedef ublas::unbounded_array<dtype> array_type;
typedef ublas::matrix<dtype> BaseSignal;
typedef ublas::matrix_slice<BaseSignal> SignalView;
typedef ublas::scalar_matrix<dtype> ScalarSignal;

/* Type of keys for various maps in the MpiSimulatorChunk. Keys are typically
 * addresses of python objects, so we need to use long long ints (64 bits). */
typedef long long int key_type;

// Current implementation: Each Operator is essentially a closure.
// At run time, these closures are stored in a list, and we call
// them sequentially each time step. The order they are called in is determined
// by the order they are given to us from python.
//
// Note that the () operator is a virtual function, which comes with some overhead.
// Future optimizations should look at another scheme, either function pointers
// or, ideally, finding some way to make these functions non-pointers and non-virtual.
//
// Note that in general reset must be called before the () operator can be called.

class Operator{

public:
    virtual string classname() const { return "Operator"; }

    virtual void operator() () = 0;
    virtual string to_string() const{
        return classname() + '\n';
    }

    virtual void reset(unsigned seed){}

    friend ostream& operator << (ostream &out, const Operator &op){
        out << "<" << op.to_string() << ">" << endl;
        return out;
    }

    void set_index(float i){ index = i;}
    float get_index() const{ return index; }

    virtual unsigned get_seed_modifier() const{ return unsigned(index); }

protected:
    float index;
};

class Reset: public Operator{

public:
    Reset(SignalView dst, dtype value);
    virtual string classname() const { return "Reset"; }

    void operator()();
    virtual string to_string() const;

protected:
    SignalView dst;
    ScalarSignal dummy;
    dtype value;
};

class Copy: public Operator{
public:
    Copy(SignalView dst, SignalView src);
    virtual string classname() const { return "Copy"; }

    void operator()();
    virtual string to_string() const;

protected:
    SignalView dst;
    SignalView src;
};

class SlicedCopy: public Operator{
public:
    SlicedCopy(
        SignalView B, SignalView A, bool inc,
        int start_A, int stop_A, int step_A,
        int start_B, int stop_B, int step_B,
        vector<int> seq_A, vector<int> seq_B);
    virtual string classname() const { return "SlicedCopy"; }

    void operator()();
    virtual string to_string() const;

protected:
    SignalView B;
    SignalView A;

    int length_A;
    int length_B;

    bool inc;

    int n_assignments;

    int start_A;
    int stop_A;
    int step_A;

    int start_B;
    int stop_B;
    int step_B;

    vector<int> seq_A;
    vector<int> seq_B;
};


// Increment signal Y by dot(A,X)
class DotInc: public Operator{
public:
    DotInc(SignalView A, SignalView X, SignalView Y);
    virtual string classname() const { return "DotInc"; }

    void operator()();
    virtual string to_string() const;

protected:
    bool scalar;

    SignalView A;
    SignalView X;
    SignalView Y;
};


class ElementwiseInc: public Operator{
public:
    ElementwiseInc(SignalView A, SignalView X, SignalView Y);
    virtual string classname() const { return "ElementwiseInc"; }

    void operator()();
    virtual string to_string() const;

protected:
    SignalView A;
    SignalView X;
    SignalView Y;

    bool broadcast;

    // Strides are 0 or 1, to support broadcasting
    int A_row_stride;
    int A_col_stride;

    int X_row_stride;
    int X_col_stride;
};

class NoDenSynapse: public Operator{

public:
    NoDenSynapse(SignalView input, SignalView output, dtype b);

    virtual string classname() const { return "NoDenSynapse"; }

    void operator()();
    virtual string to_string() const;

protected:
    SignalView input;
    SignalView output;

    dtype b;
};

class SimpleSynapse: public Operator{

public:
    SimpleSynapse(SignalView input, SignalView output, dtype a, dtype b);

    virtual string classname() const { return "SimpleSynapse"; }

    void operator()();
    virtual string to_string() const;

protected:
    SignalView input;
    SignalView output;

    dtype a;
    dtype b;
};

class Synapse: public Operator{

public:
    Synapse(
        SignalView input, SignalView output,
        BaseSignal numer, BaseSignal denom);

    virtual string classname() const { return "Synapse"; }

    void operator()();
    virtual string to_string() const;

    virtual void reset(unsigned seed);

protected:
    SignalView input;
    SignalView output;

    BaseSignal numer;
    BaseSignal denom;

    vector< boost::circular_buffer<dtype> > x;
    vector< boost::circular_buffer<dtype> > y;
};

class TriangleSynapse: public Operator{

public:
    TriangleSynapse(SignalView input, SignalView output, dtype n0, dtype ndiff, int n_taps);

    virtual string classname() const { return "TriangleSynapse"; }

    void operator()();
    virtual string to_string() const;

    virtual void reset(unsigned seed);

protected:
    SignalView input;
    SignalView output;

    dtype n0;
    dtype ndiff;
    int n_taps;

    vector< boost::circular_buffer<dtype> > x;
};


class WhiteNoise: public Operator{

public:
    WhiteNoise(
        SignalView output, dtype mean, dtype std,
        bool do_scale, bool inc, dtype dt);

    virtual string classname() const { return "WhiteNoise"; }

    void operator()();
    virtual string to_string() const;

    virtual void reset(unsigned seed);

protected:
    SignalView output;

    dtype mean;
    dtype std;

    dtype alpha;

    bool do_scale;
    bool inc;

    dtype dt;

    default_random_engine rng;
    normal_distribution<dtype> dist;
};

class WhiteSignal: public Operator{

public:
    WhiteSignal(SignalView output, BaseSignal coefs);

    virtual string classname() const { return "WhiteSignal"; }

    void operator()();
    virtual string to_string() const;

    virtual void reset(unsigned seed);

protected:
    SignalView output;
    BaseSignal coefs;
    int idx;
};

class LIF: public Operator{

public:
    LIF(
        int n_neuron, dtype tau_rc, dtype tau_ref, dtype min_voltage,
        dtype dt, SignalView J, SignalView output, SignalView voltage,
        SignalView ref_time);
    virtual string classname() const { return "LIF"; }

    void operator()();
    virtual string to_string() const;

protected:
    dtype dt;
    dtype dt_inv;
    dtype tau_rc;
    dtype tau_ref;
    dtype min_voltage;
    int n_neurons;

    SignalView J;
    SignalView output;
    SignalView voltage;
    SignalView ref_time;

    ScalarSignal dt_vec;
    ScalarSignal one;
    BaseSignal mult;
    BaseSignal dV;
};

class LIFRate: public Operator{
public:
    LIFRate(int n_neurons, dtype tau_rc, dtype tau_ref, SignalView J, SignalView output);
    virtual string classname() const { return "LIFRate"; }

    void operator()();
    virtual string to_string() const;

protected:
    dtype tau_rc;
    dtype tau_ref;
    int n_neurons;

    SignalView J;
    SignalView output;
};

class AdaptiveLIF: public LIF{
public:
    AdaptiveLIF(
        int n_neuron, dtype tau_n, dtype inc_n, dtype tau_rc, dtype tau_ref,
        dtype min_voltage, dtype dt, SignalView J, SignalView output, SignalView voltage,
        SignalView ref_time, SignalView adaptation);
    virtual string classname() const { return "AdaptiveLIF"; }

    void operator()();
    virtual string to_string() const;

protected:
    dtype tau_n;
    dtype inc_n;
    SignalView adaptation;
    BaseSignal temp;
};

class AdaptiveLIFRate: public LIFRate{

public:
    AdaptiveLIFRate(
        int n_neurons, dtype tau_n, dtype inc_n, dtype tau_rc, dtype tau_ref,
        dtype dt, SignalView J, SignalView output, SignalView adaptation);
    virtual string classname() const { return "AdaptiveLIFRate"; }

    void operator()();
    virtual string to_string() const;

protected:
    dtype dt;
    dtype tau_n;
    dtype inc_n;
    SignalView adaptation;
    BaseSignal temp;
};

class RectifiedLinear: public Operator{
public:
    RectifiedLinear(int n_neurons, SignalView J, SignalView output);
    virtual string classname() const { return "RectifiedLinear"; }

    void operator()();
    virtual string to_string() const;

protected:
    int n_neurons;

    SignalView J;
    SignalView output;
};

class Sigmoid: public Operator{
public:
    Sigmoid(int n_neurons, dtype tau_ref, SignalView J, SignalView output);
    virtual string classname() const { return "Sigmoid"; }

    void operator()();
    virtual string to_string() const;

protected:
    int n_neurons;
    dtype tau_ref;
    dtype tau_ref_inv;

    SignalView J;
    SignalView output;
};

class Izhikevich: public Operator{
public:
    Izhikevich(
        int n_neurons, dtype tau_recovery, dtype coupling,
        dtype reset_voltage, dtype reset_recovery, dtype dt,
        SignalView J, SignalView output, SignalView voltage,
        SignalView recovery);
    virtual string classname() const { return "Izhikevich"; }

    void operator()();
    virtual string to_string() const;

protected:
    int n_neurons;
    dtype tau_recovery;
    dtype coupling;
    dtype reset_voltage;
    dtype reset_recovery;
    dtype dt;
    dtype dt_inv;

    SignalView J;
    SignalView output;
    SignalView voltage;
    SignalView recovery;

    BaseSignal dV;
    BaseSignal dU;
    BaseSignal voltage_squared;
    ScalarSignal bias;
};

class BCM: public Operator{
public:
    BCM(
        SignalView pre_filtered, SignalView post_filtered, SignalView weights,
        SignalView delta, dtype learning_rate, dtype dt);
    virtual string classname() const { return "BCM"; }

    void operator()();
    virtual string to_string() const;

protected:
    dtype alpha;

    SignalView pre_filtered;
    SignalView post_filtered;
    SignalView theta;
    SignalView delta;
};

class Oja: public Operator{
public:
    Oja(
        SignalView pre_filtered, SignalView post_filtered, SignalView theta,
        SignalView delta, dtype learning_rate, dtype dt, dtype beta);
    virtual string classname() const { return "Oja"; }

    void operator()();
    virtual string to_string() const;

protected:
    dtype alpha;
    dtype beta;

    SignalView pre_filtered;
    SignalView post_filtered;
    SignalView weights;
    SignalView delta;
};

class Voja: public Operator{
public:
    Voja(
        SignalView pre_decoded, SignalView post_filtered, SignalView scaled_encoders,
        SignalView delta, SignalView learning_signal, BaseSignal scale,
        dtype learning_rate, dtype dt);
    virtual string classname() const { return "Voja"; }

    void operator()();
    virtual string to_string() const;

protected:
    dtype alpha;

    SignalView pre_decoded;
    SignalView post_filtered;
    SignalView scaled_encoders;
    SignalView delta;
    SignalView learning_signal;

    BaseSignal scale;
};




string signal_to_string(const SignalView signal);
string signal_to_string(const BaseSignal signal);
string shape_string(const SignalView signal);

#endif