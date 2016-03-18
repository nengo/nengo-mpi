#include "signal.hpp"

Signal::Signal()
:label(""), ndim(0), size(0), shape1(0), shape2(0),
stride1(0), stride2(0), offset(0), is_view(false),
is_contiguous(true), row_major(true){

}

// Create a vector (ndim=1) base signal.
Signal::Signal(
    unsigned n, dtype val, string label)
:label(label), data(shared_ptr<dtype>(new dtype[n])), ndim(1), size(n),
shape1(n), shape2(1), stride1(1), stride2(1), offset(0),
is_view(false), is_contiguous(true), row_major(true){

    raw_data = data.get();
    fill(raw_data, raw_data + n, val);
}

// Create a vector (ndim=1) from an existing buffer.
Signal::Signal(
    unsigned n, shared_ptr<dtype> data, string label)
:label(label), data(data), ndim(1), size(n),
shape1(n), shape2(1), stride1(1), stride2(1), offset(0),
is_view(false), is_contiguous(true), row_major(true){

    raw_data = data.get();
}

// Create a matrix (ndim=2) base signal.
Signal::Signal(
    unsigned m, unsigned n, dtype val, string label)
:label(label), data(shared_ptr<dtype>(new dtype[m*n])), ndim(2), size(m*n),
shape1(m), shape2(n), stride1(n), stride2(1), offset(0), is_view(false),
is_contiguous(true), row_major(true){

    raw_data = data.get();
    fill(raw_data, raw_data + m * n, val);
}

// Create a matrix (ndim=2) from an existing buffer.
Signal::Signal(
    unsigned m, unsigned n, shared_ptr<dtype> data, string label)
:label(label), data(data), ndim(2), size(m*n),
shape1(m), shape2(n), stride1(n), stride2(1), offset(0),
is_view(false), is_contiguous(true), row_major(true){

    raw_data = data.get();
}

string Signal::to_string() const{
    stringstream out;
    out << "<Signal - " << (label.size() > 0 ? label : "(NULL)") << " | "
        << " ndim=" << ndim
        << ", shape=" << shape_string(*this)
        << ", size=" << size
        << ", stride=" << stride_string(*this)
        << ", raw_data=" << raw_data;

    if(RUN_DEBUG_TEST){
        out << endl;
        for(unsigned i = 0; i < shape1; i++){
            out << i << ": ";
            for(unsigned j = 0; j < shape2; j++){
                out << operator()(i, j) << ", ";
            }

            out << endl;
        }
    }

    out << ">";

    return out.str();
}

Signal Signal::deep_copy() const{
    Signal signal(shape1, shape2);
    signal.fill_with(*this);
    return signal;
}

void Signal::copy_to_buffer(dtype* buffer) const{
    if(is_contiguous){
        memcpy(buffer, raw_data, size * sizeof(dtype));
    }else if(stride2 == 1){
        for(unsigned i = 0; i < shape1; i++){
            memcpy(buffer + i * shape2,
                   raw_data + i * stride1,
                   shape2 * sizeof(dtype));
        }
    }else{
        unsigned buffer_offset = 0;
        for(unsigned i = 0; i < shape1; i++){
            for(unsigned j = 0; j < shape2; j++){
                *(buffer + buffer_offset) = operator()(i, j);
                buffer_offset++;
            }
        }
    }
}

Signal Signal::get_view(
        string label_, unsigned ndim_, unsigned shape1_, unsigned shape2_,
        int stride1_, int stride2_, unsigned offset_) const{

    if(is_view){
        throw runtime_error(
            "View of view of signal is not currently supported.");
    }

    Signal view(*this);

    view.label = label_;
    view.ndim = ndim_;
    view.shape1 = shape1_;
    view.shape2 = shape2_;
    view.size = shape1_ * shape2_;
    view.stride1 = stride1_;
    view.stride2 = stride2_;
    view.offset = offset_;
    view.is_view = true;
    view.is_contiguous = _is_contiguous(view);
    view.row_major = (stride2_ == 1);

    view.raw_data += offset_;
    return view;
}

// ********************************************************************************
bool _is_contiguous(const Signal signal){
    if(signal.shape1 == 1){
        return signal.stride2 == 1 || signal.shape2 == 1;
    }

    if(signal.shape2 == 1){
        return signal.stride1 == 1 || signal.shape1 == 1;
    }

    return (signal.stride1 == 1 && signal.stride2 == signal.shape1)
        || (signal.stride2 == 1 && signal.stride1 == signal.shape2);
}

string signal_to_string(const Signal signal){

    stringstream ss;

    if(RUN_DEBUG_TEST){
        ss << signal;
    }else{
        ss << "[" << signal.shape1 << ", " << signal.shape2 << "]";
    }

    return ss.str();
}

string shape_string(const Signal signal){
    stringstream ss;
    ss << "(" << signal.shape1 << ", " << signal.shape2 << ")";
    return ss.str();
}

string stride_string(const Signal signal){
    stringstream ss;
    ss << "(" << signal.stride1 << ", " << signal.stride2 << ")";
    return ss.str();
}

string out_of_range_message(unsigned max, unsigned idx, unsigned axis){
    stringstream ss;
    ss << "In Signal.operator(): index " << idx
       << " out of range along axis " << axis << ". "
       << "Maximum is " << max << ".";

    return ss.str();
}
