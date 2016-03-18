#pragma once

#include <string>
#include <sstream>
#include <iostream>
#include <memory>
#include <cstdint>
#include <cstring>

#include "typedef.hpp"
#include "debug.hpp"


using namespace std;

string out_of_range_message(unsigned max, unsigned idx, unsigned axis);

struct Signal {
    Signal();

    // Create a vector (ndim=1) base signal.
    Signal(unsigned n, dtype val=0.0, string label="");

    // Create a vector (ndim=1) from an existing buffer.
    Signal(unsigned n, shared_ptr<dtype> data, string label="");

    // Create a matrix (ndim=2) base signal.
    Signal(unsigned m, unsigned n, dtype val=0.0, string label="");

    // Create a matrix (ndim=2) from an existing buffer.
    Signal(unsigned m, unsigned n, shared_ptr<dtype> data, string label="");

    Signal(const Signal& s)=default;
    Signal& operator= (const Signal& signal)=default;

    void fill_with(const Signal& signal);
    void fill_with(const dtype& scalar);

    string to_string() const;

    dtype& operator() (unsigned row, unsigned col);
    dtype operator() (unsigned row, unsigned col) const;

    // for vectors
    dtype& operator() (unsigned idx);
    dtype operator() (unsigned idx) const;

    bool operator== (const Signal& other) const;
    bool operator!= (const Signal& other) const;

    Signal deep_copy() const;

    // Copy to a buffer in row-major order.
    void copy_to_buffer(dtype* buffer) const;

    Signal get_view(
            string label_, unsigned ndim_, unsigned shape1_, unsigned shape2_,
            int stride1_, int stride2_, unsigned offset_) const;

    string label;

    unsigned ndim;

    unsigned size;

    unsigned shape1;
    unsigned shape2;

    int stride1;
    int stride2;

    unsigned offset;
    bool is_view;
    bool is_contiguous;
    bool row_major;

    // A pointer to the base array. Offset is not included in this.
    shared_ptr<dtype> data;

    // A pointer to the location in the base array where the current signal starts.
    // i.e. roughly equivalent to ``data.get() + offset``.
    dtype* raw_data;

    friend ostream& operator << (ostream &out, const Signal &sv){
        out << sv.to_string();
        return out;
    }
};

inline
void Signal::fill_with(const Signal& signal){

    if(shape1 != signal.shape1 || shape2 != signal.shape2){
        stringstream out;
        out << "Shape mismatch in Signal.fill_with. "
            << "Attempting to fill " << endl
            << *this << endl
            << " with " << endl
            << signal << endl;
        throw runtime_error(out.str());
    }

    if(is_contiguous){
        signal.copy_to_buffer(raw_data);
    }else{
        for(unsigned i = 0; i < shape1; i++){
            for(unsigned j = 0; j < shape2; j++){
                operator()(i, j) = signal(i, j);
            }
        }
    }
}

inline
void Signal::fill_with(const dtype& scalar){
    if(is_contiguous){
        fill(raw_data, raw_data + size, scalar);
    }else if(stride2 == 1){
        for(unsigned i = 0; i < shape1; i++){
            fill(raw_data + i * stride1, raw_data + i * stride1 + shape2, scalar);
        }
    }else{
        for(unsigned i = 0; i < shape1; i++){
            for(unsigned j = 0; j < shape2; j++){
                operator()(i, j) = scalar;
            }
        }
    }
}

inline
dtype& Signal::operator() (unsigned row, unsigned col){
    if (row >= shape1){
        throw out_of_range(out_of_range_message(shape1, row, 0));
    }

    if (col >= shape2){
        throw out_of_range(out_of_range_message(shape2, col, 1));
    }

    return *(raw_data + int(row) * stride1 + int(col) * stride2);
}

inline
dtype Signal::operator() (unsigned row, unsigned col) const{
    if (row >= shape1){
        throw out_of_range(out_of_range_message(shape1, row, 0));
    }

    if (col >= shape2){
        throw out_of_range(out_of_range_message(shape2, col, 1));
    }

    return *(raw_data + int(row) * stride1 + int(col) * stride2);
}

// for dealing with vectors.
inline
dtype& Signal::operator() (unsigned idx){
    if (idx >= shape1){
        throw out_of_range(out_of_range_message(shape1, idx, 0));
    }

    return *(raw_data + int(idx) * stride1);
}

inline
dtype Signal::operator() (unsigned idx) const{
    if (idx >= shape1){
        throw out_of_range(out_of_range_message(shape1, idx, 0));
    }

    return *(raw_data + int(idx) * stride1);
}

inline
bool Signal::operator== (const Signal& other) const{
    bool identical = true;
    identical &= other.shape1 == shape1;
    identical &= other.shape2 == shape2;

    if(!identical){
        return false;
    }

    for(unsigned i = 0; i < shape1; i++){
        for(unsigned j = 0; j < shape2; j++){
            if (other(i, j) != operator()(i, j)){
                return false;
            }
        }
    }

    return true;
}

inline
bool Signal::operator!= (const Signal& other) const{
    return !(*this == other);
}

bool _is_contiguous(const Signal signal);

string signal_to_string(const Signal signal);
string shape_string(const Signal signal);
string stride_string(const Signal signal);
