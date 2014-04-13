#include "mpi_operators.hpp"

MPISend::MPISend(){
    waiter->request = &request;
}

MPIRecv::MPIRecv(){
    waiter->request = &request;
}

MPIWait::MPIWait()
    :first_call(false){
}

void MPISend::operator() (){

    request = comm.isend(dst, tag, *content);

#ifdef _DEBUG
    cout << *this;
#endif
}

void MPIRecv::operator() (){

    request = comm.irecv(src, tag, *content);

#ifdef _DEBUG
    cout << *this;
#endif
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

void MPISend::print(ostream &out) const{
}

void MPIRecv::print(ostream &out) const{
}

void MPIWait::print(ostream &out) const{
}

