/**
 * @file execution_strategies.hpp
 * @brief Include file for all execution strategy implementations
 * @author UobinoPino
 * @date 2024
 *
 * This file includes all available execution strategies for DTW
 * computation, providing a single include point for strategy headers.
 */

#ifndef DTW_ACCELERATOR_EXECUTION_STRATEGIES_HPP
#define DTW_ACCELERATOR_EXECUTION_STRATEGIES_HPP

#include "base_strategy.hpp"
#include "dtw_accelerator/execution/sequential/standard_strategy.hpp"
#include "dtw_accelerator/execution/sequential/blocked_strategy.hpp"

#ifdef USE_OPENMP
#include "dtw_accelerator/execution/parallel/openmp/openmp_strategy.hpp"
#endif

#ifdef USE_MPI
#include "dtw_accelerator/execution/parallel/mpi/mpi_strategy.hpp"
#endif

#include "auto_strategy.hpp"

#endif // DTW_ACCELERATOR_EXECUTION_STRATEGIES_HPP