#include<mpi.h>
#include<string>

string receive_string(int src, int tag, MPI_Comm comm);
void send_string(string s, int dst, int tag, MPI_Comm comm);

float recv_float(int src, int tag, MPI_Comm comm);
void send_float(float f, int dst, int tag, MPI_Comm comm);

int recv_int(int src, int tag, MPI_Comm comm);
void send_int(int i, int dst, int tag, MPI_Comm comm);
