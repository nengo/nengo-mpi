#ifndef NENGO_MPI_MPI_OPS_HPP
#define NENGO_MPI_MPI_OPS_HPP

#include <mpi.h>
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/intercommunicator.hpp>

#include "operator.hpp"

namespace mpi = boost::mpi;
using namespace std;

class MPIWait;

// Each MPISend and MPIRecv operator has a corresponding MPIWait
// operator, which completes the isend/irecv. The MPIWait operator
// should occur before the MPISend/Receive in the operator ordering
// in MpiSimulatorChunk.

//This all assumes that communication happens on nengo signals that
//are UPDATED (rather than just incremented or set).

//We use mpi tags to identify signals. The tags will be the address of
//the corresponding python signal.

//The MPISend operator should be called directly
//after its content vector is updated by an operation. The corresponding
//MPIWait operator should be called directly before the content vector is
//updated.
//

class MPISend: public Operator{

    friend class MPIWait;

public:
    MPISend(){};
    MPISend(int dst, int tag, BaseMatrix* content);
    string classname() const { return "MPISend"; }

    void operator()();
    virtual string to_string() const;

    mpi::request* get_request_pointer(){ return &request;}

    int tag;
    mpi::communicator* comm;
    BaseMatrix* content;

private:
    int dst;
    mpi::request request;
};

//In INTERLEAVED mode the MPIRecv operator should be called directly
//after all operators that make use of its content vector have been called.
//The corresponding MPIWait operator should be called directly before any of
//the operators that make use of the content vector.
class MPIRecv: public Operator{

    friend class MPIWait;

public:
    MPIRecv(){};
    MPIRecv(int src, int tag, BaseMatrix* content);
    string classname() const { return "MPIRecv"; }

    void operator()();
    virtual string to_string() const;

    mpi::request* get_request_pointer(){ return &request;}

    int tag;
    mpi::communicator* comm;
    BaseMatrix* content;

private:
    int src;
    mpi::request request;
};

// Used to complete isend/irecv operations.
class MPIWait: public Operator{

public:
    MPIWait(){};
    MPIWait(int tag);
    string classname() const { return "MPIWait"; }

    void operator()();
    virtual string to_string() const;

    int tag;
    mpi::request* request;

private:
    bool first_call;

};

#endif
