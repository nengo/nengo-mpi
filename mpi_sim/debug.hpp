/*
Adapted by Eric Crawford from stackoverflow:

http://stackoverflow.com/questions/1644868/c-define-macro-for-debug-printing/1644898#1644898
*/

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

//#define debug_print(fmt, ...) \
//        do { if (DEBUG_TEST) fprintf(stderr, "%s:%d:%s(): " fmt, __FILE__, \
//                                __LINE__, __func__, __VA_ARGS__); } while (0)

//#define debug_print(x) \
//        do { if (DEBUG_TEST) std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << ": " << x << std::endl; } while (0)
//
#define dbg(x) do { if (BASIC_DEBUG_TEST) std::cerr << "mpi_sim: " << x << std::endl; } while (0)
#define build_dbg(x) do { if (BUILD_DEBUG_TEST) std::cerr << "mpi_sim: " << x << std::endl; } while (0)
#define run_dbg(x) do { if (RUN_DEBUG_TEST) std::cerr << "mpi_sim: " << x << std::endl; } while (0)