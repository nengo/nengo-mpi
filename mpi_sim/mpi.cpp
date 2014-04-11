
#include "mpi.hpp"

MPISend::MPISend(){
    waiter->request = &request;
}

void MPISend::operator() (){

    request = comm.isend(dst, tag, *content);

#ifdef _DEBUG
    cout << *this;
#endif
}

void MPISend::print(ostream &out) const{
}

MPIRecv::MPIRecv(){
    waiter->request = &request;
}

void MPIRecv::operator() (){

    request = comm.irecv(src, tag, *content);

#ifdef _DEBUG
    cout << *this;
#endif
}

void MPIRecv::print(ostream &out) const{
}

MPIWait::MPIWait()
    :first_call(false){
}

void MPIWait::operator() (){

    if(not first_call){
        request->wait();
    }

    first_call = true;

#ifdef _DEBUG
    cout << *this;
#endif
}

void MPIWait::print(ostream &out) const{
}