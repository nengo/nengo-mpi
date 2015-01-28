#include "mpi_utilities.hpp"

string receive_string(int src, int tag, MPI_Comm comm){
    int size;
    MPI_Status status;

    // Convention: recv the size of the string, ommitting the c_str's terminating character
    MPI_Recv(&size, 1, MPI_INT, src, tag, comm, &status);

    char* buffer = new char[size];
    MPI_Recv(buffer, size+1, MPI_CHAR, src, tag, comm, &status);

    string s(string_buffer);
    free(string_buffer);

    return s;
}

void send_string(string s, int dst, int tag, MPI_Comm comm){
    int size = s.length();

    // Convention: send the size of the string, ommitting the c_str's terminating character
    MPI_Send(&size, 1, MPI_INT, dst, tag, comm);

    MPI_Send(s.c_str(), size+1, MPI_CHAR, dst, tag, comm);
}

float recv_float(int src, int tag, MPI_Comm comm){
    float f;
    MPI_Send(&f, 1, MPI_FLOAT, src, tag, comm);
    return f;
}

void send_float(float f, int dst, int tag, MPI_Comm comm){
    MPI_Send(&f, 1, MPI_FLOAT, dst, tag, comm);
}

int recv_int(int src, int tag, MPI_Comm comm){
    int i;
    MPI_Send(&i, 1, MPI_INT, src, tag, comm);
    return i;
}

void send_int(int i, int dst, int tag, MPI_Comm comm){
    MPI_Send(&i, 1, MPI_INT, dst, tag, comm);
}