#ifndef NENGO_MPI_MPI_OPS_HPP
#define NENGO_MPI_MPI_OPS_HPP

#include <mpi.h>

#include "operator.hpp"

using namespace std;

class MPISend: public Operator{

public:
    MPISend(){};
    MPISend(int dst, int tag, BaseMatrix* content);
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

    BaseMatrix* content;
    floattype* content_data;
    floattype* buffer;

    int dst;
};

class MPIRecv: public Operator{

public:
    MPIRecv(){};
    MPIRecv(int src, int tag, BaseMatrix* content);
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

    BaseMatrix* content;
    floattype* content_data;
    floattype* buffer;

    int src;
};

static const int BARRIER_PERIOD = 50;

class MPIBarrier: public Operator{

public:
    MPIBarrier():comm(NULL), step(0){};
    MPIBarrier(MPI_Comm comm):comm(comm), step(0){};
    string classname() const { return "MPIBarrier"; }

    void operator()();
    virtual string to_string() const;

    MPI_Comm comm;

private:
    int step;
};

#endif
