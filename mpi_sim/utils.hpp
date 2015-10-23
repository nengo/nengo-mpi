#ifndef NENGO_MPI_UTILS_HPP
#define NENGO_MPI_UTILS_HPP

#include <vector>
#include <string>
#include <memory>

#include "operator.hpp"

/* Helper function to extract a BaseSignal from a string.
 * If get_size is true, assumes the data is encoded in the format:
 * size_1, size_2, data_0, data_1, ..., data_(size_1 * size_2 - 1)
 * Otherwise, assumes the data is encoded in the format:
 * data_0, data_1, ..., data_(n-1)
 * In the latter case, the shape of the returned signal is (n, 1) */
unique_ptr<BaseSignal> python_list_to_signal(string s, bool get_size=false);

/* Helper function to extract a vector of indices from a string.
 * Assumes the data for the index vector is encoded in the format:
 * index_0, index_1, ..., index_(n-1)
 * The length of the returned vector is n */
vector<int> python_list_to_index_vector(string s);

#endif
