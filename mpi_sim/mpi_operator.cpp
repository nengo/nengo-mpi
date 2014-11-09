#include "mpi_operator.hpp"

MPISend::MPISend(int dst, int tag, Matrix* content):
    dst(dst), tag(tag), content(content){
}

MPIRecv::MPIRecv(int src, int tag, Matrix* content):
    src(src), tag(tag), content(content){
}

MPIWait::MPIWait(int tag)
    :tag(tag), first_call(true){
}

void MPISend::operator() (){

    request = comm->isend(dst+1, tag, content->data());

    run_dbg(*this);
}

void MPIRecv::operator() (){

    request = comm->irecv(src+1, tag, content->data());

    run_dbg(*this);
}

void MPIWait::operator() (){
    if(first_call){
        first_call = false;
    }else{
        request->wait();
    }

    run_dbg(*this);
}

string MPISend::to_string() const{
    stringstream out;

    out << "MPISend:" << endl;
    out << "tag: " << tag << endl;
    out << "dst: " << dst << endl;
    out << "content:" << endl;
    out << *content << endl;

    return out.str();
}

string MPIRecv::to_string() const{
    stringstream out;

    out << "MPIRecv:" << endl;
    out << "tag: " << tag << endl;
    out << "src: " << src << endl;
    out << "content:" << endl;
    out << *content << endl;

    return out.str();
}

string MPIWait::to_string() const{
    stringstream out;

    out << "MPIWait:" << endl;
    out << "tag: " << tag << endl;
    out << "first_call: " << first_call << endl;

    return out.str();
}
