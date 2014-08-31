#include "mpi_operators.hpp"

MPISend::MPISend(int dst, int tag, Vector* content):
    dst(dst), tag(tag), content(content){
}

MPIRecv::MPIRecv(int src, int tag, Vector* content):
    src(src), tag(tag), content(content){
}

MPIWait::MPIWait(int tag)
    :tag(tag), first_call(false){
}

void MPISend::operator() (){

    request = comm.isend(dst, tag, *content);

#ifdef _RUN_DEBUG
    cout << *this;
#endif
}

void MPIRecv::operator() (){

    request = comm.irecv(src, tag, *content);

#ifdef _RUN_DEBUG
    cout << *this;
#endif
}

void MPIWait::operator() (){

    if(not first_call){
        request->wait();
    }else{
        first_call = true;
    }

#ifdef _RUN_DEBUG
    cout << *this;
#endif
}

void MPISend::print(ostream &out) const{
}

void MPIRecv::print(ostream &out) const{
}

void MPIWait::print(ostream &out) const{
}

void MPISend::set_waiter(MPIWait* mpi_wait){
    mpi_wait->request = &request;
}

void MPIRecv::set_waiter(MPIWait* mpi_wait){
    mpi_wait->request = &request;
}
