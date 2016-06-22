#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <memory>
#include <random>
#include <cstdint>

#include <boost/circular_buffer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#ifdef __cplusplus
extern "C"
{
#endif

#include <cblas.h>

#ifdef __cplusplus
}
#endif

#include "signal.hpp"
#include "typedef.hpp"
#include "debug.hpp"


using namespace std;

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
        stringstream ss;
        ss << classname() << endl;
        ss << "index: "<< index << endl;
        return ss.str();
    }

    // Here we only need to reset aspects of operator's state that are *not* stored as signals
    // because resetting signals is handled by the chunk. Consequently, most operators won't
    // need to override this.
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

class TimeUpdate: public Operator{

public:
    TimeUpdate(Signal step, Signal time, dtype t);
    virtual string classname() const { return "TimeUpdate"; }

    void operator()();
    virtual string to_string() const;

protected:
    Signal step;
    Signal time;
    const dtype dt;
};


class Reset: public Operator{

public:
    Reset(Signal dst, dtype value);
    virtual string classname() const { return "Reset"; }

    void operator()();
    virtual string to_string() const;

protected:
    Signal dst;
    const dtype value;
};

class Copy: public Operator{
public:
    Copy(Signal dst, Signal src);
    virtual string classname() const { return "Copy"; }

    void operator()();
    virtual string to_string() const;

protected:
    Signal dst;
    Signal src;
};

class SlicedCopy: public Operator{
public:
    SlicedCopy(
        Signal src, Signal dst,
        int start_src, int stop_src, int step_src,
        int start_dst, int stop_dst, int step_dst,
        vector<int> seq_src, vector<int> seq_dst, bool inc);
    virtual string classname() const { return "SlicedCopy"; }

    void operator()();
    virtual string to_string() const;

protected:
    Signal src;
    Signal dst;

    const unsigned length_src;
    const unsigned length_dst;

    const int start_src;
    const int stop_src;
    const int step_src;

    const int start_dst;
    const int stop_dst;
    const int step_dst;

    const vector<int> seq_src;
    const vector<int> seq_dst;

    const bool inc;
    unsigned n_assignments;
};


// Increment signal Y by dot(A,X)
class DotInc: public Operator{
public:
    DotInc(Signal A, Signal X, Signal Y);
    virtual string classname() const { return "DotInc"; }

    void operator()();
    virtual string to_string() const;

protected:
    const bool scalar;
    bool matrix_vector;

    Signal A;
    Signal X;
    Signal Y;

    CBLAS_TRANSPOSE transpose_A;
    CBLAS_TRANSPOSE transpose_X;

    unsigned leading_dim_A;
    unsigned leading_dim_X;
    unsigned leading_dim_Y;

    unsigned m;
    unsigned n;
    unsigned k;
};


class ElementwiseInc: public Operator{
public:
    ElementwiseInc(Signal A, Signal X, Signal Y);
    virtual string classname() const { return "ElementwiseInc"; }

    void operator()();
    virtual string to_string() const;

protected:
    Signal A;
    Signal X;
    Signal Y;

    // Strides are 0 or 1, to support broadcasting
    const unsigned A_row_stride;
    const unsigned A_col_stride;

    const unsigned X_row_stride;
    const unsigned X_col_stride;
};

class NoDenSynapse: public Operator{

public:
    NoDenSynapse(Signal input, Signal output, dtype b);

    virtual string classname() const { return "NoDenSynapse"; }

    void operator()();
    virtual string to_string() const;

protected:
    Signal input;
    Signal output;

    const dtype b;
};

class SimpleSynapse: public Operator{

public:
    SimpleSynapse(Signal input, Signal output, dtype a, dtype b);

    virtual string classname() const { return "SimpleSynapse"; }

    void operator()();
    virtual string to_string() const;

protected:
    Signal input;
    Signal output;

    const dtype a;
    const dtype b;
};

class Synapse: public Operator{

public:
    Synapse(
        Signal input, Signal output,
        Signal numer, Signal denom);

    virtual string classname() const { return "Synapse"; }

    void operator()();
    virtual string to_string() const;

    virtual void reset(unsigned seed);

protected:
    Signal input;
    Signal output;

    const Signal numer;
    const Signal denom;

    vector< boost::circular_buffer<dtype> > x;
    vector< boost::circular_buffer<dtype> > y;
};

class TriangleSynapse: public Operator{

public:
    TriangleSynapse(Signal input, Signal output, dtype n0, dtype ndiff, unsigned n_taps);

    virtual string classname() const { return "TriangleSynapse"; }

    void operator()();
    virtual string to_string() const;

    virtual void reset(unsigned seed);

protected:
    Signal input;
    Signal output;

    const dtype n0;
    const dtype ndiff;
    const unsigned n_taps;

    vector< boost::circular_buffer<dtype> > x;
};


class WhiteNoise: public Operator{

public:
    WhiteNoise(
        Signal output, dtype mean, dtype std,
        bool do_scale, bool inc, dtype dt);

    virtual string classname() const { return "WhiteNoise"; }

    void operator()();
    virtual string to_string() const;

    virtual void reset(unsigned seed);

protected:
    Signal output;

    const dtype mean;
    const dtype std;

    default_random_engine rng;
    normal_distribution<dtype> dist;

    const dtype alpha;

    const bool do_scale;
    const bool inc;

    const dtype dt;
};

class WhiteSignal: public Operator{

public:
    WhiteSignal(Signal coefs, Signal output, Signal time, dtype dt);

    virtual string classname() const { return "WhiteSignal"; }

    void operator()();
    virtual string to_string() const;

protected:
    const Signal coefs;
    Signal output;
    Signal time;
    dtype dt;
};

class PresentInput: public Operator{

public:
    PresentInput(
        Signal input, Signal output, Signal time,
        dtype presentation_time, dtype dt);

    virtual string classname() const { return "PresentInput"; }

    void operator()();
    virtual string to_string() const;

protected:
    const Signal input;
    Signal output;
    Signal time;

    dtype presentation_time;
    dtype dt;
};


class LIF: public Operator{

public:
    LIF(
        unsigned n_neuron, dtype tau_rc, dtype tau_ref, dtype min_voltage,
        dtype dt, Signal J, Signal output, Signal voltage,
        Signal ref_time);
    virtual string classname() const { return "LIF"; }

    void operator()();
    virtual string to_string() const;

protected:
    const unsigned n_neurons;

    const dtype dt;
    const dtype dt_inv;

    const dtype tau_rc;
    const dtype tau_ref;

    const dtype min_voltage;

    Signal J;
    Signal output;
    Signal voltage;
    Signal ref_time;

    const Signal one;
    Signal mult;
    Signal dV;
};

class LIFRate: public Operator{
public:
    LIFRate(unsigned n_neurons, dtype tau_rc, dtype tau_ref, Signal J, Signal output);
    virtual string classname() const { return "LIFRate"; }

    void operator()();
    virtual string to_string() const;

protected:
    unsigned n_neurons;

    const dtype tau_rc;
    const dtype tau_ref;

    Signal J;
    Signal output;
};

class AdaptiveLIF: public LIF{
public:
    AdaptiveLIF(
        unsigned n_neuron, dtype tau_n, dtype inc_n, dtype tau_rc, dtype tau_ref,
        dtype min_voltage, dtype dt, Signal J, Signal output, Signal voltage,
        Signal ref_time, Signal adaptation);
    virtual string classname() const { return "AdaptiveLIF"; }

    void operator()();
    virtual string to_string() const;

protected:
    const dtype tau_n;
    const dtype inc_n;

    Signal adaptation;
    Signal temp_J;
    Signal dAdapt;
};

class AdaptiveLIFRate: public LIFRate{

public:
    AdaptiveLIFRate(
        unsigned n_neurons, dtype tau_n, dtype inc_n, dtype tau_rc, dtype tau_ref,
        dtype dt, Signal J, Signal output, Signal adaptation);
    virtual string classname() const { return "AdaptiveLIFRate"; }

    void operator()();
    virtual string to_string() const;

protected:
    const dtype dt;
    const dtype tau_n;
    const dtype inc_n;

    Signal adaptation;
    Signal temp_J;
    Signal dAdapt;
};

class RectifiedLinear: public Operator{
public:
    RectifiedLinear(unsigned n_neurons, Signal J, Signal output);
    virtual string classname() const { return "RectifiedLinear"; }

    void operator()();
    virtual string to_string() const;

protected:
    const unsigned n_neurons;

    Signal J;
    Signal output;
};

class Sigmoid: public Operator{
public:
    Sigmoid(unsigned n_neurons, dtype tau_ref, Signal J, Signal output);
    virtual string classname() const { return "Sigmoid"; }

    void operator()();
    virtual string to_string() const;

protected:
    const unsigned n_neurons;

    const dtype tau_ref;
    const dtype tau_ref_inv;

    Signal J;
    Signal output;
};

class BCM: public Operator{
public:
    BCM(
        Signal pre_filtered, Signal post_filtered, Signal weights,
        Signal delta, dtype learning_rate, dtype dt);
    virtual string classname() const { return "BCM"; }

    void operator()();
    virtual string to_string() const;

protected:
    const dtype alpha;

    Signal pre_filtered;
    Signal post_filtered;
    Signal theta;
    Signal delta;

    Signal squared_pf;
};

class Oja: public Operator{
public:
    Oja(
        Signal pre_filtered, Signal post_filtered, Signal theta,
        Signal delta, dtype learning_rate, dtype dt, dtype beta);
    virtual string classname() const { return "Oja"; }

    void operator()();
    virtual string to_string() const;

protected:
    const dtype alpha;
    const dtype beta;

    Signal pre_filtered;
    Signal post_filtered;
    Signal weights;
    Signal delta;
};

class Voja: public Operator{
public:
    Voja(
        Signal pre_decoded, Signal post_filtered, Signal scaled_encoders,
        Signal delta, Signal learning_signal, Signal scale,
        dtype learning_rate, dtype dt);
    virtual string classname() const { return "Voja"; }

    void operator()();
    virtual string to_string() const;

protected:
    const dtype alpha;

    Signal pre_decoded;
    Signal post_filtered;
    Signal scaled_encoders;
    Signal delta;
    Signal learning_signal;

    Signal scale;
};