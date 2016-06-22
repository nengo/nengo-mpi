#include "mpi_operator.hpp"

MPISend::MPISend(int dst, int tag, Signal content)
:MPIOperator(tag), dst(dst), content(content){

    if(!content.is_contiguous){
        throw runtime_error("MPISend got a non-contiguous signal.");
    }

    content_data = content.raw_data;
    size = content.size;
    buffer = unique_ptr<dtype>(new dtype[size]);
}

void MPISend::operator() (){
    if(first_call){
        first_call = false;
    }else{
        MPI_Wait(&request, &status);
    }

    memcpy(buffer.get(), content_data, size * sizeof(dtype));

    MPI_Isend(buffer.get(), size, MPI_DOUBLE, dst, tag, comm, &request);

    mpi_dbg(*this);
}

string MPISend::to_string() const{
    stringstream out;

    out << "MPISend:" << endl;
    out << "tag: " << tag << endl;
    out << "dst: " << dst << endl;
    out << "size: " << size << endl;
    out << "content:" << endl;
    out << signal_to_string(content) << endl;

    /*
    out << "buffer:" << endl;
    for(int i = 0; i < size; i++){
        out << buffer.get()[i] << ", " << endl;
    }
    */

    return out.str();
}

MPIRecv::MPIRecv(int src, int tag, Signal content, bool is_update)
:MPIOperator(tag), src(src), content(content), is_update(is_update){

    if(!content.is_contiguous){
        throw runtime_error("MPIRecv got a non-contiguous signal.");
    }

    content_data = content.raw_data;
    size = content.size;
    buffer = unique_ptr<dtype>(new dtype[size]);
}

void MPIRecv::operator() (){
    if(is_update && first_call){
        first_call = false;
    }else{
        MPI_Wait(&request, &status);
        memcpy(content_data, buffer.get(), size * sizeof(dtype));
        MPI_Irecv(buffer.get(), size, MPI_DOUBLE, src, tag, comm, &request);
    }

    mpi_dbg(*this);
}

void MPIRecv::init(){
    MPI_Irecv(buffer.get(), size, MPI_DOUBLE, src, tag, comm, &request);
}

void MPIRecv::complete(){
    if(!is_update){
        MPI_Cancel(&request);
    }
    MPI_Wait(&request, &status);
}

string MPIRecv::to_string() const{
    stringstream out;

    out << "MPIRecv:" << endl;
    out << "tag: " << tag << endl;
    out << "src: " << src << endl;
    out << "size: " << size << endl;
    out << "is_update: " << is_update << endl;
    out << "content:" << endl;
    out << signal_to_string(content) << endl;

    /*
    out << "buffer:" << endl;
    for(int i = 0; i < size; i++){
        out << buffer.get()[i] << ", " << endl;
    }
    */

    return out.str();
}
