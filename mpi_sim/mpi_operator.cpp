#include "mpi_operator.hpp"

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

    run_dbg(*this);
}

void MPIRecv::operator() (){

    request = comm.irecv(src, tag, *content);

    run_dbg(*this);
}

void MPIWait::operator() (){

    if(not first_call){
        request->wait();
    }else{
        first_call = true;
    }

    run_dbg(*this);
}

string MPISend::to_string() const{
    stringstream out;

    out << "MPISend:" << endl;
    out << "tag: " << tag << endl;
    out << "dst: " << dst << endl;
    out << "content: " << endl;
    out << *content << endl;

    string out_string;
    out >> out_string;

    return out_string;
}

string MPIRecv::to_string() const{
    stringstream out;

    out << "MPIRecv:" << endl;
    out << "tag: " << tag << endl;
    out << "src: " << src << endl;
    out << "content: " << endl;
    out << *content << endl;

    string out_string;
    out >> out_string;

    return out_string;
}

string MPIWait::to_string() const{
    stringstream out;

    out << "MPIWait:" << endl;
    out << "tag: " << tag << endl;
    out << "first_call: " << first_call << endl;

    string out_string;
    out >> out_string;

    return out_string;
}

void MPISend::set_waiter(MPIWait* mpi_wait){
    //mpi_wait->request = &request;
}

void MPIRecv::set_waiter(MPIWait* mpi_wait){
    //mpi_wait->request = &request;
}
