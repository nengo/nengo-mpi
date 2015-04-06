#ifndef NENGO_MPI_UTILS_HPP
#define NENGO_MPI_UTILS_HPP

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

#include "simulator.hpp"
#include "mpi_simulator.hpp"

using namespace std;

unique_ptr<Simulator> create_simulator_from_file(string filename);

#endif
