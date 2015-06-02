#include "spec.hpp"

OpSpec::OpSpec(string op_string){
    try{
        vector<string> tokens;
        boost::split(tokens, op_string, boost::is_any_of(OP_DELIM));
        type_string = tokens[0];
        tokens.erase(tokens.begin());

        index = boost::lexical_cast<float>(tokens.back());
        tokens.pop_back();

        arguments = tokens;

    }catch(const boost::bad_lexical_cast& e){
        stringstream msg;
        msg << "Caught bad lexical cast while extracting OpSpec from string "
                "with error: " << e.what() << endl;
        msg << "The op string was: " << op_string << endl;
        throw logic_error(msg.str());
    }
}

string OpSpec::to_string() const{
    stringstream out;

    out << "OpSpec:" << endl;
    out << "Type: " << type_string << endl;
    out << "Index: " << index << endl;
    out << "Arguments:" << endl;
    for(auto& s : arguments){
        out << s << endl;
    }

    return out.str();
}


SignalSpec::SignalSpec(string signal_string){
    try{
        vector<string> tokens;
        boost::split(tokens, signal_string, boost::is_any_of(SIGNAL_DELIM));

        key = boost::lexical_cast<key_type>(tokens[0]);

        vector<string> shape_tokens;
        boost::trim_if(tokens[1], boost::is_any_of("(,)"));
        boost::split(shape_tokens, tokens[1], boost::is_any_of(","));

        shape1 = boost::lexical_cast<int>(shape_tokens[0]);
        shape2 = shape_tokens.size() == 1 ? 1 : boost::lexical_cast<int>(shape_tokens[1]);

        vector<string> stride_tokens;
        boost::trim_if(tokens[2], boost::is_any_of("(,)"));
        boost::split(stride_tokens, tokens[2], boost::is_any_of(","));

        stride1 = boost::lexical_cast<int>(stride_tokens[0]);
        stride2 = stride_tokens.size() == 1 ? 1 : boost::lexical_cast<int>(stride_tokens[1]);

        offset = boost::lexical_cast<int>(tokens[3]);

    }catch(const boost::bad_lexical_cast& e){
        stringstream msg;
        msg << "Caught bad lexical cast while extracting SignalSpec from string "
                "with error: " << e.what() << endl;
        msg << "The signal string was: " << signal_string << endl;
        throw logic_error(msg.str());
    }
}

string SignalSpec::to_string() const{
    stringstream out;

    out << "SignalSpec:" << endl;
    out << "key: " << key << endl;
    out << "shape: (" << shape1 << ", " << shape2 << ")"<< endl;
    out << "stride: (" << stride1 << ", " << stride2 << ")"<< endl;
    out << "offset: " << offset << endl;

    return out.str();
}

ProbeSpec::ProbeSpec(string probe_string){
    try{
        vector<string> tokens;
        boost::split(tokens, probe_string, boost::is_any_of(PROBE_DELIM), boost::token_compress_on);

        component = boost::lexical_cast<int>(tokens[0]);
        probe_key = boost::lexical_cast<key_type>(tokens[1]);
        signal_string = tokens[2];
        period = boost::lexical_cast<dtype>(tokens[3]);
        name = tokens[4];
        signal_spec = SignalSpec(signal_string);

    }catch(const boost::bad_lexical_cast& e){
        stringstream msg;
        msg << "Caught bad lexical cast while extracting ProbeSpec from string "
                "with error: " << e.what() << endl;
        msg << "The probe string was: " << probe_string << endl;
        throw logic_error(msg.str());
    }
}

string ProbeSpec::to_string() const{
    stringstream out;

    out << "ProbeSpec:" << endl;
    out << "component: " << component << endl;
    out << "probe_key: " << probe_key << endl;
    out << "signal: " << signal_spec << endl;
    out << "period: " << period << endl;
    out << "name: " << name << endl;

    return out.str();
}
