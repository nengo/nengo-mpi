#include "mpi_operator.hpp"

MPISend::MPISend(int dst, int tag, SignalView content)
:MPIOperator(tag), dst(dst), content(content){

    content_data = &(content.data().expression().data()[0]);
    size = content.size1() * content.size2();
    buffer = unique_ptr<dtype>(new dtype[size]);
}

MPIRecv::MPIRecv(int src, int tag, SignalView content)
:MPIOperator(tag), src(src), content(content){

    content_data = &(content.data().expression().data()[0]);
    size = content.size1() * content.size2();
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

void MPIRecv::operator() (){

    if(first_call){
        first_call = false;
    }else{
        MPI_Wait(&request, &status);

        memcpy(content_data, buffer.get(), size * sizeof(dtype));
    }

    MPI_Irecv(buffer.get(), size, MPI_DOUBLE, src, tag, comm, &request);

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

string MPIRecv::to_string() const{
    stringstream out;

    out << "MPIRecv:" << endl;
    out << "tag: " << tag << endl;
    out << "src: " << src << endl;
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

// *************************************
// Merged operators, used in MERGED mode.

MergedMPISend::MergedMPISend(int dst, int tag, vector<SignalView> content)
:MPIOperator(tag), dst(dst), content(content){

    sizes = vector<int>();
    content_data = vector<dtype*>();

    size = 0;

    for(SignalView& c : content){
        int s = c.size1() * c.size2();
        sizes.push_back(s);

        size += s;

        content_data.push_back(&(c.data().expression().data()[0]));
    }

    buffer = unique_ptr<dtype>(new dtype[size]);
}

MergedMPIRecv::MergedMPIRecv(int src, int tag, vector<SignalView> content)
:MPIOperator(tag), src(src), content(content){

    sizes = vector<int>();
    content_data = vector<dtype*>();

    size = 0;

    for(SignalView& c : content){
        int s = c.size1() * c.size2();
        sizes.push_back(s);

        size += s;

        content_data.push_back(&(c.data().expression().data()[0]));
    }

    buffer = unique_ptr<dtype>(new dtype[size]);
}

void MergedMPISend::operator() (){

    if(first_call){
        first_call = false;
    }else{
        MPI_Wait(&request, &status);
    }

    dtype* buffer_offset = buffer.get();

    int i = 0;
    for(auto& c : content_data){
        int s = sizes[i];
        memcpy(buffer_offset, c, s * sizeof(dtype));

        buffer_offset += s;
        i++;
    }

    MPI_Isend(buffer.get(), size, MPI_DOUBLE, dst, tag, comm, &request);

    mpi_dbg(*this);
}

void MergedMPIRecv::operator() (){

    if(first_call){
        first_call = false;
    }else{
        MPI_Wait(&request, &status);

        dtype* buffer_offset = buffer.get();

        int i = 0;
        for(auto& c : content_data){
            int s = sizes[i];
            memcpy(c, buffer_offset, s * sizeof(dtype));

            buffer_offset += s;
            i++;
        }
    }

    MPI_Irecv(buffer.get(), size, MPI_DOUBLE, src, tag, comm, &request);

    mpi_dbg(*this);
}

string MergedMPISend::to_string() const{
    stringstream out;

    out << "MergedMPISend:" << endl;
    out << "tag: " << tag << endl;
    out << "dst: " << dst << endl;
    out << "size: " << size << endl;

    /*
    out << "content:" << endl;
    for(const auto& c : content){
        out << signal_to_string(c) << endl;
    }

    dtype* buffer_offset = buffer.get();
    out << "buffer:" << endl;
    for(int s : sizes){
        out << "[" << s << "]: ";
        for(int i = 0; i < s ; i++){
            out << *buffer_offset << ", ";
            buffer_offset++;
        }

        out << endl;
    }
    */

    return out.str();
}

string MergedMPIRecv::to_string() const{
    stringstream out;

    out << "MergedMPIRecv:" << endl;
    out << "tag: " << tag << endl;
    out << "src: " << src << endl;
    out << "size: " << size << endl;

    /*
    out << "content:" << endl;
    for(const auto& c : content){
        out << signal_to_string(c) << endl;
    }

    dtype* buffer_offset = buffer.get();
    out << "buffer:" << endl;
    for(int s : sizes){
        out << "[" << s << "]: ";
        for(int i = 0; i < s ; i++){
            out << *buffer_offset << ", ";
            buffer_offset++;
        }

        out << endl;
    }
    */

    return out.str();
}


