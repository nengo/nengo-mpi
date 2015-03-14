#ifndef NENGO_MPI_MPI_OPS_HPP
#define NENGO_MPI_MPI_OPS_HPP

#include <mpi.h>

#include "operator.hpp"

using namespace std;

class MPISend: public Operator{

public:
    MPISend(int dst, int tag, SignalView content);
    string classname() const { return "MPISend"; }

    void operator()();
    void complete();
    void set_communicator(MPI_Comm comm);
    virtual string to_string() const;

private:
    bool first_call;
    int size;

    int tag;
    MPI_Comm comm;
    MPI_Request request;
    MPI_Status status;

    SignalView content;
    dtype* content_data;
    dtype* buffer;

    int dst;
};

class MPIRecv: public Operator{

public:
    MPIRecv(int src, int tag, SignalView content);
    string classname() const { return "MPIRecv"; }

    void operator()();
    void complete();
    void set_communicator(MPI_Comm comm);
    virtual string to_string() const;

private:
    bool first_call;
    int size;

    int tag;
    MPI_Comm comm;
    MPI_Request request;
    MPI_Status status;

    SignalView content;
    dtype* content_data;
    dtype* buffer;

    int src;
};

static const int BARRIER_PERIOD = 50;

class MPIBarrier: public Operator{

public:
    MPIBarrier(MPI_Comm comm):comm(comm), step(0){};
    string classname() const { return "MPIBarrier"; }

    void operator()();
    virtual string to_string() const;

    MPI_Comm comm;

private:
    int step;
};

#endif
