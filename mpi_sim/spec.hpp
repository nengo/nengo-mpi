#pragma once

#include <string>
#include <vector>
#include <sstream>

#include "operator.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

const string OP_DELIM = ";";
const string SIGNAL_DELIM = ":";
const string PROBE_DELIM = "|";

using namespace std;

class Spec{
    virtual string to_string() const = 0;

    friend ostream& operator << (ostream &out, const Spec &op){
        out << op.to_string();
        return out;
    }
};

class OpSpec: public Spec {
public:
    OpSpec(){};
    OpSpec(string op_string);

    string type_string;
    vector<string> arguments;
    float index;

    string to_string() const override;
};

/* Expected format of signal_string:
*     key:(shape1, shape2):(stride1, stride2):offset */
class SignalSpec: public Spec {
public:
    SignalSpec(){};
    SignalSpec(string signal_string);

    key_type key;
    int shape1;
    int shape2;
    int stride1;
    int stride2;
    int offset;

    string to_string() const override;
};

class ProbeSpec: public Spec {
public:
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
