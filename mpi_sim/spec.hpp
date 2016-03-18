#pragma once

#include <string>
#include <vector>
#include <sstream>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "typedef.hpp"
#include "debug.hpp"


using namespace std;

const string OP_DELIM = ";";
const string SIGNAL_DELIM = ":";
const string PROBE_DELIM = "|";

struct Spec{
    virtual string to_string() const = 0;

    friend ostream& operator << (ostream &out, const Spec &op){
        out << op.to_string();
        return out;
    }
};

struct OpSpec: public Spec {
    OpSpec(){};
    OpSpec(string op_string);

    string type_string;
    vector<string> arguments;
    float index;

    string to_string() const override;
};

/* Expected format of signal_string:
*     key:label:ndim:(shape1, shape2):(stride1, stride2):offset */
struct SignalSpec: public Spec {
    SignalSpec(){};
    SignalSpec(string signal_string);

    key_type key;
    string label;

    unsigned ndim;

    unsigned shape1;
    unsigned shape2;

    unsigned stride1;
    unsigned stride2;

    unsigned offset;

    string to_string() const override;
};

struct ProbeSpec: public Spec {
    ProbeSpec(){};
    ProbeSpec(string probe_string);

    int component;
    key_type probe_key;
    string signal_string;
    SignalSpec signal_spec;
    dtype period;
    string name;

    string to_string() const override;
};
