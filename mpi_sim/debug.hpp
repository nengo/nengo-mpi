#ifndef NENGO_MPI_DEBUG_HPP
#define NENGO_MPI_DEBUG_HPP

/*
Adapted by Eric Crawford from stackoverflow:

http://stackoverflow.com/questions/1644868/c-define-macro-for-debug-printing/1644898#1644898
*/

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

#ifdef DEBUG
#define BASIC_DEBUG_TEST 1
#else
#define BASIC_DEBUG_TEST 0
#endif

#ifdef BUILD_DEBUG
#define BUILD_DEBUG_TEST 1
#else
#define BUILD_DEBUG_TEST 0
#endif

#ifdef RUN_DEBUG
#define RUN_DEBUG_TEST 1
#else
#define RUN_DEBUG_TEST 0
#endif

#ifdef MPI_DEBUG
#define MPI_DEBUG_TEST 1
#else
#define MPI_DEBUG_TEST 0
#endif

extern ostream* debug_stream;

inline void init_debug_file(string filename){
    debug_stream = new ofstream();
    ((ofstream*) debug_stream)->open(filename);
}

inline ostream* get_debug_stream(){
    return debug_stream;
}

inline void close_debug_file(){
    ((ofstream*) debug_stream)->close();
    delete debug_stream;
    debug_stream = &(cerr);
}

//#define debug_print(fmt, ...) \
//        do { if (DEBUG_TEST) fprintf(stderr, "%s:%d:%s(): " fmt, __FILE__, \
//                                __LINE__, __func__, __VA_ARGS__); } while (0)

//#define debug_print(x) \
//        do { if (DEBUG_TEST) std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << ": " << x << std::endl; } while (0)
//
#define dbgfile(x) do { if(BASIC_DEBUG_TEST) init_debug_file(x); } while (0)
#define dbg(x) do { if (BASIC_DEBUG_TEST) (*get_debug_stream()) << x << endl; } while (0)
#define build_dbg(x) do { if (BUILD_DEBUG_TEST) (*get_debug_stream()) << x << endl; } while (0)
#define run_dbg(x) do { if (RUN_DEBUG_TEST) (*get_debug_stream()) << x << endl; } while (0)
#define mpi_dbg(x) do { if (MPI_DEBUG_TEST) (*get_debug_stream()) << x << endl; } while (0)
#define clsdbgfile(x) do { if(BASIC_DEBUG_TEST) close_debug_file(x); } while (0)

// #define dbg(x) do { if (BASIC_DEBUG_TEST) std::cerr << "mpi_sim: " << x << std::endl; } while (0)
// #define build_dbg(x) do { if (BUILD_DEBUG_TEST) std::cerr << "mpi_sim: " << x << std::endl; } while (0)
// #define run_dbg(x) do { if (RUN_DEBUG_TEST) std::cerr << "mpi_sim: " << x << std::endl; } while (0)
// #define mpi_dbg(x) do { if (MPI_DEBUG_TEST) std::cerr << "mpi_sim: " << x << std::endl; } while (0)
#endif