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
                    std::cout << "Final cost: " << final_cost << std::endl;
                }

                return {final_cost, path};
            }





        } // namespace mpi
    } // namespace parallel
} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_PARALLEL_MPI_DTW_HPP