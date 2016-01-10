#pragma once

#include <mpi.h>
#include "operator.hpp"

using namespace std;

class MPIOperator: public Operator{

public:
    MPIOperator():first_call(true){}
    MPIOperator(int tag):tag(tag), first_call(true){}
    string classname() const { return "MPIOperator"; }

    virtual void operator() () = 0;
    virtual string to_string() const = 0;
    void complete(){ MPI_Wait(&request, &status); }
    void set_communicator(MPI_Comm comm){ this->comm = comm; }

protected:
    bool first_call;

    int tag;
    MPI_Comm comm;
    MPI_Request request;
    MPI_Status status;

    unique_ptr<dtype> buffer;
    int size;
};

class MPISend: public MPIOperator{

public:
    MPISend(int dst, int tag, SignalView content);
    string classname() const { return "MPISend"; }

    void operator()();
    virtual string to_string() const;

private:
    SignalView content;
    dtype* content_data;

    int dst;
};

class MPIRecv: public MPIOperator{

public:
    MPIRecv(int src, int tag, SignalView content);
    string classname() const { return "MPIRecv"; }

    void operator()();
    virtual string to_string() const;

private:
    SignalView content;
    dtype* content_data;

    int src;
};

// *************************************
// Merged operators, used in MERGED mode.
// MERGED makes an effort to reduce the number of messages that
// are sent per time step. Each component should only be sending one large
// message to each component that it has to communicate with. That message
// will contain all data from all signals that have to be sent. The default
// (i.e. non-merged) mode instead sends one message per signal which can
// introduce significant overhead.
class MergedMPISend: public MPIOperator{

public:
    // `buffer` will be a pointer into a buffer mainted by the chunk
    MergedMPISend(int dst, int tag, vector<SignalView> content);

    string classname() const { return "MergedMPISend"; }

    void operator()();
    virtual string to_string() const;

private:
    vector<SignalView> content;
    vector<int> sizes;
    vector<dtype*> content_data;

    int dst;
};

class MergedMPIRecv: public MPIOperator{

public:
    MergedMPIRecv(int src, int tag, vector<SignalView> content);

    string classname() const { return "MergedMPIRecv"; }

    void operator()();
    virtual string to_string() const;

private:
    vector<SignalView> content;
    vector<int> sizes;
    vector<dtype*> content_data;

    int src;
};
