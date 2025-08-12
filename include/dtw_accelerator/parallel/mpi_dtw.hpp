#ifndef DTW_ACCELERATOR_PARALLEL_MPI_DTW_HPP
#define DTW_ACCELERATOR_PARALLEL_MPI_DTW_HPP

#include <vector>
#include <utility>
#include <limits>
#include <algorithm>
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <unordered_map>
#include <string>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <immintrin.h>

#include "../distance_metrics.hpp"
#include "../constraints.hpp"
#include "../path_processing.hpp"
#include "../core_dtw.hpp"
#include "../fast_dtw.hpp"

namespace dtw_accelerator {
    namespace parallel {
        namespace mpi {

            // Optimized DTW data structure
            struct DTWData {
                int n, m, dim;
                std::vector<std::vector<double>> A;
                std::vector<std::vector<double>> B;
                std::vector<double> D_flat;

                std::vector<double> prev_row;
                std::vector<double> curr_row;

                DTWData(int n_, int m_, int dim_, bool full_matrix = false)
                        : n(n_), m(m_), dim(dim_) {
                    if (full_matrix) {
                        D_flat.resize((n+1)*(m+1), std::numeric_limits<double>::infinity());
                        D_flat[0] = 0;
                    } else {
                        // For row-based computation
                        prev_row.resize(m+1, std::numeric_limits<double>::infinity());
                        curr_row.resize(m+1, std::numeric_limits<double>::infinity());
                        prev_row[0] = 0;
                    }
                }

                inline double& D(int i, int j) {
                    return D_flat[i*(m+1) + j];
                }

                inline const double& D(int i, int j) const {
                    return D_flat[i*(m+1) + j];
                }
            };
            // Optimized distance computation with vectorization hints (still needs to be tested in practice)
            inline double compute_distance_optimized(const double* a, const double* b, int dim) {
                double sum = 0.0;
                int i = 0;

                // Vectorized loop for dimensions divisible by 4
                for (; i <= dim - 4; i += 4) {
                    double diff0 = a[i] - b[i];
                    double diff1 = a[i+1] - b[i+1];
                    double diff2 = a[i+2] - b[i+2];
                    double diff3 = a[i+3] - b[i+3];
                    sum += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3;
                }

                // Handle remaining dimensions
                for (; i < dim; ++i) {
                    double diff = a[i] - b[i];
                    sum += diff * diff;
                }

                return std::sqrt(sum);
            }
            // Main MPI DTW implementation
            // Optimized wavefront DTW with row-based parallelization
            inline std::pair<double, std::vector<std::pair<int, int>>> dtw_mpi(
                    const std::vector<std::vector<double>>& A,
                    const std::vector<std::vector<double>>& B,
                    MPI_Comm comm = MPI_COMM_WORLD) {

                int rank, size;
                MPI_Comm_rank(comm, &rank);
                MPI_Comm_size(comm, &size);

                int n = A.size();
                int m = B.size();
                int dim = A.empty() ? 0 : A[0].size();

                // For single process, use sequential algorithm
                if (size == 1) {
                    return core::dtw_cpu(A, B);
                }

                // Broadcast dimensions
                int dims[3] = {n, m, dim};
                MPI_Bcast(dims, 3, MPI_INT, 0, comm);
                n = dims[0]; m = dims[1]; dim = dims[2];

                // Calculate row distribution
                int rows_per_proc = n / size;
                int extra_rows = n % size;
                int my_start_row = rank * rows_per_proc + std::min(rank, extra_rows);
                int my_num_rows = rows_per_proc + (rank < extra_rows ? 1 : 0);
                int my_end_row = my_start_row + my_num_rows;


                // Distribute data efficiently

                // Only distribute the rows each process needs
                std::vector<std::vector<double>> local_A(my_num_rows, std::vector<double>(dim));
                std::vector<std::vector<double>> B_local = B;

                if (rank == 0) {
                    // Send each process its rows
                    for (int p = 1; p < size; ++p) {
                        int p_start = p * rows_per_proc + std::min(p, extra_rows);
                        int p_rows = rows_per_proc + (p < extra_rows ? 1 : 0);

                        for (int i = 0; i < p_rows; ++i) {
                            MPI_Send(A[p_start + i].data(), dim, MPI_DOUBLE, p, i, comm);
                        }
                    }
                    // Copy local rows
                    for (int i = 0; i < my_num_rows; ++i) {
                        local_A[i] = A[my_start_row + i];
                    }
                } else {
                    // Receive my rows
                    for (int i = 0; i < my_num_rows; ++i) {
                        MPI_Recv(local_A[i].data(), dim, MPI_DOUBLE, 0, i, comm, MPI_STATUS_IGNORE);
                    }
                }

                // Broadcast B to all processes
                std::vector<double> B_flat(m * dim);
                if (rank == 0) {
                    for (int j = 0; j < m; ++j) {
                        std::memcpy(B_flat.data() + j * dim, B[j].data(), dim * sizeof(double));
                    }
                }
                MPI_Bcast(B_flat.data(), m * dim, MPI_DOUBLE, 0, comm);

                if (rank != 0) {
                    B_local.resize(m, std::vector<double>(dim));
                    for (int j = 0; j < m; ++j) {
                        std::memcpy(B_local[j].data(), B_flat.data() + j * dim, dim * sizeof(double));
                    }
                }


                // Row-based computation with pipelining

                // Each process maintains two rows: previous and current
                std::vector<double> prev_row(m + 1, std::numeric_limits<double>::infinity());
                std::vector<double> curr_row(m + 1, std::numeric_limits<double>::infinity());

                // Initialize first row for process 0
                if (rank == 0) {
                    prev_row[0] = 0;
                    for (int j = 1; j <= m; ++j) {
                        prev_row[j] = std::numeric_limits<double>::infinity();
                    }
                }

                // Process rows with pipelining
                for (int global_i = 1; global_i <= n; ++global_i) {

                    if (global_i > my_start_row && global_i <= my_end_row) {
                        int local_i = global_i - my_start_row - 1;

                        // Receive previous row from previous process
                        if (global_i == my_start_row + 1 && rank > 0) {
                            MPI_Recv(prev_row.data(), m + 1, MPI_DOUBLE, rank - 1, global_i - 1, comm, MPI_STATUS_IGNORE);
                        }

                        // Compute current row
                        curr_row[0] = std::numeric_limits<double>::infinity();

#pragma omp simd
                        for (int j = 1; j <= m; ++j) {
                            double cost = compute_distance_optimized(local_A[local_i].data(), B_local[j-1].data(), dim);
                            double best = std::min({prev_row[j], curr_row[j-1], prev_row[j-1]});
                            curr_row[j] = cost + best;
                        }

                        // Send current row to next process if needed
                        if (global_i == my_end_row && rank < size - 1) {
                            MPI_Send(curr_row.data(), m + 1, MPI_DOUBLE, rank + 1, global_i, comm);
                        }

                        // Swap rows
                        std::swap(prev_row, curr_row);
                    }
                }


                // Gather final result
                double final_cost;

                if (rank == size - 1) {
                    final_cost = prev_row[m];
                    if (rank > 0) {
                        MPI_Send(&final_cost, 1, MPI_DOUBLE, 0, 0, comm);
                    }
                } else if (rank == 0) {
                    MPI_Recv(&final_cost, 1, MPI_DOUBLE, size - 1, 0, comm, MPI_STATUS_IGNORE);
                }

                // Broadcast final cost
                MPI_Bcast(&final_cost, 1, MPI_DOUBLE, 0, comm);


                // Only root performs backtracking
                std::vector<std::pair<int, int>> path;
                if (rank == 0) {

                    // Backtracking from final cost
                    int i = n, j = m;
                    const double INF = std::numeric_limits<double>::infinity();
                    while (i > 0 || j > 0) {
                        path.emplace_back(i-1, j-1);

                        double d0 = (i > 0 && j > 0) ? prev_row[j-1] : INF;
                        double d1 = (i > 0) ? prev_row[j] : INF;
                        double d2 = (j > 0) ? curr_row[j-1] : INF;

                        if (d0 <= d1 && d0 <= d2) {
                            --i; --j;
                        } else if (d1 < d2) {
                            --i;
                        } else {
                            --j;
                        }
                    }
                    std::reverse(path.begin(), path.end());
                  //  std::cout << "Final cost: " << final_cost << std::endl;
                }

                return {final_cost, path};
            }

            // MPI version of DTW with constraints
            template<constraints::ConstraintType CT, int R = 1, double S = 2.0>
            inline std::pair<double, std::vector<std::pair<int, int>>> dtw_mpi_with_constraint(
                    const std::vector<std::vector<double>>& A,
                    const std::vector<std::vector<double>>& B,
                    MPI_Comm comm = MPI_COMM_WORLD) {

                using namespace constraints;

                int rank, size;
                MPI_Comm_rank(comm, &rank);
                MPI_Comm_size(comm, &size);

                int n = A.size();
                int m = B.size();
                int dim = A.empty() ? 0 : A[0].size();

                // For single process, use sequential algorithm
                if (size == 1) {
                    return core::dtw_with_constraint<CT, R, S>(A, B);
                }

                // Broadcast dimensions
                int dims[3] = {n, m, dim};
                MPI_Bcast(dims, 3, MPI_INT, 0, comm);
                n = dims[0]; m = dims[1]; dim = dims[2];

                const double INF = std::numeric_limits<double>::infinity();

                // Calculate row distribution
                int rows_per_proc = n / size;
                int extra_rows = n % size;
                int my_start_row = rank * rows_per_proc + std::min(rank, extra_rows);
                int my_num_rows = rows_per_proc + (rank < extra_rows ? 1 : 0);
                int my_end_row = my_start_row + my_num_rows;

                // Distribute data efficiently
                std::vector<std::vector<double>> local_A(my_num_rows, std::vector<double>(dim));
                std::vector<std::vector<double>> B_local = B;

                if (rank == 0) {
                    // Send each process its rows
                    for (int p = 1; p < size; ++p) {
                        int p_start = p * rows_per_proc + std::min(p, extra_rows);
                        int p_rows = rows_per_proc + (p < extra_rows ? 1 : 0);

                        for (int i = 0; i < p_rows; ++i) {
                            MPI_Send(A[p_start + i].data(), dim, MPI_DOUBLE, p, i, comm);
                        }
                    }
                    // Copy local rows
                    for (int i = 0; i < my_num_rows; ++i) {
                        local_A[i] = A[my_start_row + i];
                    }
                } else {
                    // Receive my rows
                    for (int i = 0; i < my_num_rows; ++i) {
                        MPI_Recv(local_A[i].data(), dim, MPI_DOUBLE, 0, i, comm, MPI_STATUS_IGNORE);
                    }
                }

                // Broadcast B to all processes
                std::vector<double> B_flat(m * dim);
                if (rank == 0) {
                    for (int j = 0; j < m; ++j) {
                        std::memcpy(B_flat.data() + j * dim, B[j].data(), dim * sizeof(double));
                    }
                }
                MPI_Bcast(B_flat.data(), m * dim, MPI_DOUBLE, 0, comm);

                if (rank != 0) {
                    B_local.resize(m, std::vector<double>(dim));
                    for (int j = 0; j < m; ++j) {
                        std::memcpy(B_local[j].data(), B_flat.data() + j * dim, dim * sizeof(double));
                    }
                }

                // Pre-compute constraint mask for each row
                // Each process computes valid columns for its rows
                std::vector<std::vector<bool>> local_constraint_mask(my_num_rows, std::vector<bool>(m, false));
                std::vector<int> valid_cols_per_row(my_num_rows, 0);

                for (int local_i = 0; local_i < my_num_rows; ++local_i) {
                    int global_i = my_start_row + local_i;
                    for (int j = 0; j < m; ++j) {
                        bool in_constraint = false;

                        if constexpr (CT == ConstraintType::NONE) {
                            in_constraint = true;
                        }
                        else if constexpr (CT == ConstraintType::SAKOE_CHIBA) {
                            in_constraint = within_sakoe_chiba_band<R>(global_i, j, n, m);
                        }
                        else if constexpr (CT == ConstraintType::ITAKURA) {
                            in_constraint = within_itakura_parallelogram<S>(global_i, j, n, m);
                        }

                        local_constraint_mask[local_i][j] = in_constraint;
                        if (in_constraint) valid_cols_per_row[local_i]++;
                    }
                }

                // Row-based computation with pipelining and constraints
                std::vector<double> prev_row(m + 1, INF);
                std::vector<double> curr_row(m + 1, INF);

                // Initialize first row for process 0
                if (rank == 0) {
                    prev_row[0] = 0;
                    // Special case for first row computation
                    if (my_num_rows > 0) {
                        curr_row[0] = INF;
                        for (int j = 1; j <= m; ++j) {
                            if (local_constraint_mask[0][j-1]) {
                                double cost = compute_distance_optimized(local_A[0].data(), B_local[j-1].data(), dim);
                                double best = std::min({prev_row[j], curr_row[j-1], prev_row[j-1]});
                                if (best != INF) {
                                    curr_row[j] = cost + best;
                                }
                            }
                        }
                        std::swap(prev_row, curr_row);
                    }
                }

                // Process rows with pipelining
                for (int global_i = (rank == 0 ? 2 : 1); global_i <= n; ++global_i) {

                    if (global_i > my_start_row && global_i <= my_end_row) {
                        int local_i = global_i - my_start_row - 1;

                        // Receive previous row from previous process
                        if (global_i == my_start_row + 1 && rank > 0) {
                            MPI_Recv(prev_row.data(), m + 1, MPI_DOUBLE, rank - 1, global_i - 1, comm, MPI_STATUS_IGNORE);
                        }

                        // Compute current row with constraint checking
                        std::fill(curr_row.begin(), curr_row.end(), INF);
                        curr_row[0] = INF;

                        // Only compute cells within the constraint
                        for (int j = 1; j <= m; ++j) {
                            if (local_constraint_mask[local_i][j-1]) {
                                double cost = compute_distance_optimized(local_A[local_i].data(), B_local[j-1].data(), dim);

                                double min_prev = INF;

                                // Check all three predecessors
                                if (prev_row[j] != INF) {
                                    min_prev = std::min(min_prev, prev_row[j]);
                                }
                                if (curr_row[j-1] != INF) {
                                    min_prev = std::min(min_prev, curr_row[j-1]);
                                }
                                if (prev_row[j-1] != INF) {
                                    min_prev = std::min(min_prev, prev_row[j-1]);
                                }

                                if (min_prev != INF) {
                                    curr_row[j] = cost + min_prev;
                                }
                            }
                        }

                        // Send current row to next process if needed
                        if (global_i == my_end_row && rank < size - 1) {
                            MPI_Send(curr_row.data(), m + 1, MPI_DOUBLE, rank + 1, global_i, comm);
                        }

                        // Swap rows
                        std::swap(prev_row, curr_row);
                    }
                }

                // Gather final result
                double final_cost;

                if (rank == size - 1) {
                    final_cost = prev_row[m];
                    if (rank > 0) {
                        MPI_Send(&final_cost, 1, MPI_DOUBLE, 0, 0, comm);
                    }
                } else if (rank == 0) {
                    MPI_Recv(&final_cost, 1, MPI_DOUBLE, size - 1, 0, comm, MPI_STATUS_IGNORE);
                }

                // Broadcast final cost
                MPI_Bcast(&final_cost, 1, MPI_DOUBLE, 0, comm);

                // Check if we have a valid path
                if (final_cost == INF) {
                    return {INF, {}};
                }

                // Only root performs backtracking
                std::vector<std::pair<int, int>> path;
                if (rank == 0) {
                    // Backtrack to find the path
                    int i = n, j = m;
                    while (i > 0 || j > 0) {
                        path.emplace_back(i-1, j-1);

                        double d0 = (i > 0 && j > 0) ? prev_row[j-1] : INF;
                        double d1 = (i > 0) ? prev_row[j] : INF;
                        double d2 = (j > 0) ? curr_row[j-1] : INF;

                        if (d0 <= d1 && d0 <= d2) {
                            --i; --j;
                        } else if (d1 < d2) {
                            --i;
                        } else {
                            --j;
                        }
                    }
                    std::reverse(path.begin(), path.end());
                   // std::cout << "Final cost: " << final_cost << std::endl;
                }

                return {final_cost, path};
            }

            // MPI version of constrained DTW (used by FastDTW MPI)
            inline std::pair<double, std::vector<std::pair<int, int>>> dtw_constrained_mpi(
                    const std::vector<std::vector<double>>& A,
                    const std::vector<std::vector<double>>& B,
                    const std::vector<std::pair<int, int>>& window,
                    MPI_Comm comm = MPI_COMM_WORLD) {

                int rank, size;
                MPI_Comm_rank(comm, &rank);
                MPI_Comm_size(comm, &size);

                int n = A.size();
                int m = B.size();
                int dim = A.empty() ? 0 : A[0].size();

                // For single process, use sequential algorithm
                if (size == 1) {
                    return core::dtw_constrained(A, B, window);
                }

                const double INF = std::numeric_limits<double>::infinity();

                // Broadcast dimensions
                int dims[3] = {n, m, dim};
                MPI_Bcast(dims, 3, MPI_INT, 0, comm);
                n = dims[0]; m = dims[1]; dim = dims[2];

                // Create window mask for efficient lookup
                std::vector<std::vector<bool>> in_window(n, std::vector<bool>(m, false));
                for (const auto& [i, j] : window) {
                    if (i >= 0 && i < n && j >= 0 && j < m) {
                        in_window[i][j] = true;
                    }
                }

                // Calculate row distribution
                int rows_per_proc = n / size;
                int extra_rows = n % size;
                int my_start_row = rank * rows_per_proc + std::min(rank, extra_rows);
                int my_num_rows = rows_per_proc + (rank < extra_rows ? 1 : 0);
                int my_end_row = my_start_row + my_num_rows;

                // Distribute data efficiently
                std::vector<std::vector<double>> local_A(my_num_rows, std::vector<double>(dim));
                std::vector<std::vector<double>> B_local = B;

                if (rank == 0) {
                    // Send each process its rows
                    for (int p = 1; p < size; ++p) {
                        int p_start = p * rows_per_proc + std::min(p, extra_rows);
                        int p_rows = rows_per_proc + (p < extra_rows ? 1 : 0);

                        for (int i = 0; i < p_rows; ++i) {
                            MPI_Send(A[p_start + i].data(), dim, MPI_DOUBLE, p, i, comm);
                        }
                    }
                    // Copy local rows
                    for (int i = 0; i < my_num_rows; ++i) {
                        local_A[i] = A[my_start_row + i];
                    }
                } else {
                    // Receive my rows
                    for (int i = 0; i < my_num_rows; ++i) {
                        MPI_Recv(local_A[i].data(), dim, MPI_DOUBLE, 0, i, comm, MPI_STATUS_IGNORE);
                    }
                }

                // Broadcast B to all processes
                std::vector<double> B_flat(m * dim);
                if (rank == 0) {
                    for (int j = 0; j < m; ++j) {
                        std::memcpy(B_flat.data() + j * dim, B[j].data(), dim * sizeof(double));
                    }
                }
                MPI_Bcast(B_flat.data(), m * dim, MPI_DOUBLE, 0, comm);

                if (rank != 0) {
                    B_local.resize(m, std::vector<double>(dim));
                    for (int j = 0; j < m; ++j) {
                        std::memcpy(B_local[j].data(), B_flat.data() + j * dim, dim * sizeof(double));
                    }
                }

                // Row-based computation with pipelining, only computing cells in window
                std::vector<double> prev_row(m + 1, INF);
                std::vector<double> curr_row(m + 1, INF);

                // Initialize first row for process 0
                if (rank == 0) {
                    prev_row[0] = 0;
                }

                // Process rows with pipelining
                for (int global_i = 1; global_i <= n; ++global_i) {
                    if (global_i > my_start_row && global_i <= my_end_row) {
                        int local_i = global_i - my_start_row - 1;

                        // Receive previous row from previous process
                        if (global_i == my_start_row + 1 && rank > 0) {
                            MPI_Recv(prev_row.data(), m + 1, MPI_DOUBLE, rank - 1, global_i - 1, comm, MPI_STATUS_IGNORE);
                        }

                        // Compute current row
                        std::fill(curr_row.begin(), curr_row.end(), INF);
                        curr_row[0] = INF;

                        for (int j = 1; j <= m; ++j) {
                            // Only compute if this cell is in the window
                            if (global_i - 1 < n && j - 1 < m && in_window[global_i - 1][j - 1]) {
                                double cost = compute_distance_optimized(local_A[local_i].data(), B_local[j-1].data(), dim);

                                double min_prev = INF;
                                if (prev_row[j] != INF) {
                                    min_prev = std::min(min_prev, prev_row[j]);
                                }
                                if (curr_row[j-1] != INF) {
                                    min_prev = std::min(min_prev, curr_row[j-1]);
                                }
                                if (prev_row[j-1] != INF) {
                                    min_prev = std::min(min_prev, prev_row[j-1]);
                                }

                                if (min_prev != INF) {
                                    curr_row[j] = cost + min_prev;
                                }
                            }
                        }

                        // Send current row to next process if needed
                        if (global_i == my_end_row && rank < size - 1) {
                            MPI_Send(curr_row.data(), m + 1, MPI_DOUBLE, rank + 1, global_i, comm);
                        }

                        // Swap rows
                        std::swap(prev_row, curr_row);
                    }
                }

                // Gather final result
                double final_cost;

                if (rank == size - 1) {
                    final_cost = prev_row[m];
                    if (rank > 0) {
                        MPI_Send(&final_cost, 1, MPI_DOUBLE, 0, 0, comm);
                    }
                } else if (rank == 0) {
                    MPI_Recv(&final_cost, 1, MPI_DOUBLE, size - 1, 0, comm, MPI_STATUS_IGNORE);
                }

                // Broadcast final cost
                MPI_Bcast(&final_cost, 1, MPI_DOUBLE, 0, comm);

                // Only root performs backtracking
                std::vector<std::pair<int, int>> path;
                if (rank == 0) {
                    int i = n, j = m;
                    while (i > 0 || j > 0) {
                        path.emplace_back(i-1, j-1);

                        double d0 = (i > 0 && j > 0) ? prev_row[j-1] : INF;
                        double d1 = (i > 0) ? prev_row[j] : INF;
                        double d2 = (j > 0) ? curr_row[j-1] : INF;

                        if (d0 <= d1 && d0 <= d2) {
                            --i; --j;
                        } else if (d1 < d2) {
                            --i;
                        } else {
                            --j;
                        }
                    }
                    std::reverse(path.begin(), path.end());
                }

                return {final_cost, path};
            }

// MPI version of FastDTW
            inline std::pair<double, std::vector<std::pair<int, int>>> fastdtw_mpi(
                    const std::vector<std::vector<double>>& A,
                    const std::vector<std::vector<double>>& B,
                    int radius = 1,
                    int min_size = 100,
                    MPI_Comm comm = MPI_COMM_WORLD) {

                using namespace path;  // For downsample, expand_path, get_window

                int rank, size;
                MPI_Comm_rank(comm, &rank);
                MPI_Comm_size(comm, &size);

                int n = A.size();
                int m = B.size();
                int dim = A.empty() ? 0 : A[0].size();

                // For single process or small problems, use sequential algorithm
                if (size == 1 || (n <= min_size && m <= min_size)) {
                    return fast::fastdtw_cpu(A, B, radius, min_size);
                }

                // Base case: if we can't downsample further
                if (n <= 2 || m <= 2) {
                    return dtw_mpi(A, B, comm);
                }

                // Only master handles the recursive downsampling
                std::vector<std::pair<int, int>> window;
                if (rank == 0) {
                    // Recursive case:
                    // 1. Coarsen the time series
                    auto A_coarse = path::downsample(A);
                    auto B_coarse = path::downsample(B);

                    // 2. Recursively compute FastDTW on the coarsened data
                    // Use sequential version for the recursive call to avoid MPI overhead
                    auto [cost_coarse, path_coarse] = fast::fastdtw_cpu(A_coarse, B_coarse, radius, min_size);

                    // 3. Project the low-resolution path to a higher resolution
                    auto projected_path = path::expand_path(path_coarse, n, m);

                    // 4. Create a search window around the projected path
                    window = path::get_window(projected_path, n, m, radius);
                }

                // Broadcast window size to all processes
                int window_size = (rank == 0) ? window.size() : 0;
                MPI_Bcast(&window_size, 1, MPI_INT, 0, comm);

                // Broadcast the window to all processes
                if (rank != 0) {
                    window.resize(window_size);
                }
                MPI_Bcast(window.data(), window_size * 2, MPI_INT, 0, comm);

                // Now perform constrained DTW within the window using MPI parallelization
                return dtw_constrained_mpi(A, B, window, comm);
            }








        } // namespace mpi
    } // namespace parallel
} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_PARALLEL_MPI_DTW_HPP