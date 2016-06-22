#pragma once

#include <mpi.h>

#include "signal.hpp"
#include "operator.hpp"

#include "typedef.hpp"
#include "debug.hpp"


using namespace std;

class MPIOperator: public Operator{

public:
    MPIOperator():first_call(true){}
    MPIOperator(int tag):first_call(true), tag(tag){}

    string classname() const { return "MPIOperator"; }

    virtual void operator() () = 0;
    virtual string to_string() const = 0;

    virtual void reset(unsigned seed){first_call = true;}

    virtual void complete(){ MPI_Wait(&request, &status); }
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
    MPISend(int dst, int tag, Signal content);
    string classname() const { return "MPISend"; }

    virtual void operator()();
    virtual string to_string() const;

private:
    int dst;
    Signal content;
    dtype* content_data;
};

class MPIRecv: public MPIOperator{

public:
    MPIRecv(int src, int tag, Signal content, bool is_update);
    string classname() const { return "MPIRecv"; }

    virtual void operator()();
    void init();
    virtual void complete();
    virtual string to_string() const;

private:
    int src;
    Signal content;
    dtype* content_data;
    bool is_update;
};
