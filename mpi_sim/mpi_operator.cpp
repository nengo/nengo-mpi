#include "mpi_operator.hpp"

MPISend::MPISend(int dst, int tag, BaseMatrix* content):
    dst(dst), tag(tag), content(content){
}

MPIRecv::MPIRecv(int src, int tag, BaseMatrix* content):
    src(src), tag(tag), content(content){
}

MPIWait::MPIWait(int tag)
    :tag(tag), first_call(true){
}

void MPISend::operator() (){

    request = comm->isend(dst, tag, content->data());

    run_dbg(*this);
}

void MPIRecv::operator() (){

    request = comm->irecv(src, tag, content->data());

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

void MPIBarrier::operator() (){
    if(step != 0 && step % BARRIER_PERIOD == 0){
        comm->barrier();
    }

    step++;

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

string MPIBarrier::to_string() const{
    stringstream out;

    out << "MPIBarrier:" << endl;
    out << "step: " << step << endl;
    out << "barrier period: " << BARRIER_PERIOD << endl;

    return out.str();
}
