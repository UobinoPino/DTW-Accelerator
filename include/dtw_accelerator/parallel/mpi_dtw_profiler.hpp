#ifndef DTW_ACCELERATOR_PARALLEL_MPI_DTW_PROFILER_HPP
#define DTW_ACCELERATOR_PARALLEL_MPI_DTW_PROFILER_HPP

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

namespace dtw_accelerator {
    namespace parallel {
        namespace mpi {

            // DTW data structure
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

            // Optimized distance computation with vectorization hints
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

            // Performance profiler class
            class MPIProfiler {
            public:
                struct TimingEntry {
                    double start_time;
                    double end_time;
                    double duration;
                    int count;

                    TimingEntry() : start_time(0), end_time(0), duration(0), count(0) {}
                };

                std::unordered_map<std::string, TimingEntry> timings;
                int rank;
                bool enabled;
                std::string prefix;
                size_t bytes_sent;
                size_t bytes_received;
                int msg_sent;
                int msg_received;

                MPIProfiler(const std::string& name = "MPIProfiler") :
                        enabled(true), prefix(name), bytes_sent(0), bytes_received(0),
                        msg_sent(0), msg_received(0) {
                    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                }

                void start(const std::string& section) {
                    if (!enabled) return;
                    timings[section].start_time = MPI_Wtime();
                    timings[section].count++;
                }

                void stop(const std::string& section) {
                    if (!enabled) return;
                    timings[section].end_time = MPI_Wtime();
                    timings[section].duration += (timings[section].end_time - timings[section].start_time);
                }

                void trackSend(size_t bytes) {
                    bytes_sent += bytes;
                    msg_sent++;
                }

                void trackReceive(size_t bytes) {
                    bytes_received += bytes;
                    msg_received++;
                }

                void enable() { enabled = true; }
                void disable() { enabled = false; }

                void reset() {
                    timings.clear();
                    bytes_sent = 0;
                    bytes_received = 0;
                    msg_sent = 0;
                    msg_received = 0;
                }

                void reportToConsole() {
                    if (rank != 0) return;

                    std::cout << "\n===== " << prefix << " Performance Report =====\n";
                    std::cout << std::setw(30) << std::left << "Section";
                    std::cout << std::setw(15) << std::right << "Time (ms)";
                    std::cout << std::setw(15) << "Count";
                    std::cout << std::setw(15) << "Avg (ms)" << std::endl;
                    std::cout << std::string(75, '-') << std::endl;

                   // double total_time = 0;


                    for (const auto& [section, entry] : timings) {
                        double time_ms = entry.duration * 1000.0;
                        double avg_ms = time_ms / std::max(1, entry.count);

                        std::cout << std::setw(30) << std::left << section;
                        std::cout << std::setw(15) << std::right << std::fixed << std::setprecision(2) << time_ms;
                        std::cout << std::setw(15) << entry.count;
                        std::cout << std::setw(15) << std::fixed << std::setprecision(2) << avg_ms << std::endl;

                      //  total_time += time_ms;

                    }

                    std::cout << std::string(75, '-') << std::endl;
                    std::cout << "\nCommunication Stats:" << std::endl;
                    std::cout << "  Messages Sent: " << msg_sent << std::endl;
                    std::cout << "  Messages Received: " << msg_received << std::endl;
                    std::cout << "  Data Sent: " << (bytes_sent / (1024.0 * 1024.0)) << " MB" << std::endl;
                    std::cout << "  Data Received: " << (bytes_received / (1024.0 * 1024.0)) << " MB" << std::endl;

                }

                void saveToFile(const std::string& filename) {
                    std::stringstream ss;
                    ss << filename << "_rank" << rank << ".csv";

                    std::ofstream file(ss.str());
                    if (!file.is_open()) {
                        std::cerr << "Error: Could not open file " << ss.str() << std::endl;
                        return;
                    }

                    file << "Section,Time (ms),Count,Avg (ms)\n";

                    for (const auto& [section, entry] : timings) {
                        double time_ms = entry.duration * 1000.0;
                        double avg_ms = time_ms / std::max(1, entry.count);

                        file << section << ","
                             << std::fixed << std::setprecision(2) << time_ms << ","
                             << entry.count << ","
                             << std::fixed << std::setprecision(2) << avg_ms << "\n";
                    }

                    file << "\nCommunication Stats:\n";
                    file << "Messages Sent," << msg_sent << "\n";
                    file << "Messages Received," << msg_received << "\n";
                    file << "Data Sent (MB)," << (bytes_sent / (1024.0 * 1024.0)) << "\n";
                    file << "Data Received (MB)," << (bytes_received / (1024.0 * 1024.0)) << "\n";

                    file.close();
                }
            };

            // Optimized wavefront DTW with row-based parallelization
            inline std::pair<double, std::vector<std::pair<int, int>>> dtw_mpi_profiled(
                    const std::vector<std::vector<double>>& A,
                    const std::vector<std::vector<double>>& B,
                    MPIProfiler& profiler,
                    MPI_Comm comm = MPI_COMM_WORLD) {

                profiler.start("Total Execution");
                int rank, size;
                MPI_Comm_rank(comm, &rank);
                MPI_Comm_size(comm, &size);

                profiler.start("Initialization");
                int n = A.size();
                int m = B.size();
                int dim = A.empty() ? 0 : A[0].size();

                // For single process, use sequential algorithm
                if (size == 1) {
                    profiler.stop("Initialization");
                    profiler.stop("Total Execution");
                    return core::dtw_cpu(A, B);
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

                profiler.stop("Initialization");

                // Distribute data efficiently
                profiler.start("Data Distribution");

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
                            profiler.trackSend(dim * sizeof(double));
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
                        profiler.trackReceive(dim * sizeof(double));
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
                profiler.trackReceive(m * dim * sizeof(double));

                if (rank != 0) {
                    B_local.resize(m, std::vector<double>(dim));
                    for (int j = 0; j < m; ++j) {
                        std::memcpy(B_local[j].data(), B_flat.data() + j * dim, dim * sizeof(double));
                    }
                }

                profiler.stop("Data Distribution");

                // Row-based computation with pipelining
                profiler.start("Row Computation");

                // Each process maintains two rows: previous and current
                std::vector<double> prev_row(m + 1, INF);
                std::vector<double> curr_row(m + 1, INF);

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
                            profiler.trackReceive((m + 1) * sizeof(double));
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
                            profiler.trackSend((m + 1) * sizeof(double));
                        }

                        // Swap rows
                        std::swap(prev_row, curr_row);
                    }
                }

                profiler.stop("Row Computation");

                // Gather final result
                profiler.start("Result Collection");
                double final_cost;

                if (rank == size - 1) {
                    final_cost = prev_row[m];
                    if (rank > 0) {
                        MPI_Send(&final_cost, 1, MPI_DOUBLE, 0, 0, comm);
                        profiler.trackSend(sizeof(double));
                    }
                } else if (rank == 0) {
                    MPI_Recv(&final_cost, 1, MPI_DOUBLE, size - 1, 0, comm, MPI_STATUS_IGNORE);
                    profiler.trackReceive(sizeof(double));
                }

                // Broadcast final cost
                MPI_Bcast(&final_cost, 1, MPI_DOUBLE, 0, comm);

                profiler.stop("Result Collection");

                // Only root performs backtracking
                std::vector<std::pair<int, int>> path;
                if (rank == 0) {
                    profiler.start("Backtracking");

                    // Backtracking from final cost
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
                    profiler.trackSend(path.size() * sizeof(std::pair<int, int>));
                    profiler.trackReceive(path.size() * sizeof(std::pair<int, int>));
                  //  std::cout << "Final cost: " << final_cost << std::endl;
                    profiler.stop("Backtracking");
                }

                profiler.stop("Total Execution");
                return {final_cost, path};
            }

            // MPI version of DTW with constraints profiled
            template<constraints::ConstraintType CT, int R = 1, double S = 2.0>
            inline std::pair<double, std::vector<std::pair<int, int>>> dtw_mpi_constrained_profiled(
                    const std::vector<std::vector<double>>& A,
                    const std::vector<std::vector<double>>& B,
                    MPIProfiler& profiler,
                    MPI_Comm comm = MPI_COMM_WORLD) {
                using namespace constraints;

                profiler.start("Total Execution");
                int rank, size;
                MPI_Comm_rank(comm, &rank);
                MPI_Comm_size(comm, &size);

                profiler.start("Initialization");
                int n = A.size();
                int m = B.size();
                int dim = A.empty() ? 0 : A[0].size();

                // For single process, use sequential algorithm
                if (size == 1) {
                    profiler.stop("Initialization");
                    profiler.stop("Total Execution");
                    return core::dtw_with_constraint<CT, R, S>(A, B);
                }

                // Broadcast dimensions
                int dims[3] = {n, m, dim};
                MPI_Bcast(dims, 3, MPI_INT, 0, comm);
                n = dims[0];
                m = dims[1];
                dim = dims[2];

                const double INF = std::numeric_limits<double>::infinity();

                // Calculate row distribution
                int rows_per_proc = n / size;
                int extra_rows = n % size;
                int my_start_row = rank * rows_per_proc + std::min(rank, extra_rows);
                int my_num_rows = rows_per_proc + (rank < extra_rows ? 1 : 0);
                int my_end_row = my_start_row + my_num_rows;

                profiler.stop("Initialization");

                // Distribute data efficiently
                profiler.start("Data Distribution");

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
                            profiler.trackSend(dim * sizeof(double));
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
                        profiler.trackReceive(dim * sizeof(double));
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
                profiler.trackReceive(m * dim * sizeof(double));
                if (rank != 0) {
                    B_local.resize(m, std::vector<double>(dim));
                    for (int j = 0; j < m; ++j) {
                        std::memcpy(B_local[j].data(), B_flat.data() + j * dim, dim * sizeof(double));
                    }
                }
                profiler.stop("Data Distribution");
                // Pre-compute constraint mask for each row
                profiler.start("Constraint Masking");
                std::vector<std::vector<bool>> local_constraint_mask(my_num_rows, std::vector<bool>(m, false));
                std::vector<int> valid_cols_per_row(my_num_rows, 0);
                for (int local_i = 0; local_i < my_num_rows; ++local_i) {
                    int global_i = my_start_row + local_i;
                    for (int j = 0; j < m; ++j) {
                        bool in_constraint = false;

                        if constexpr (CT == ConstraintType::NONE) {
                            in_constraint = true;
                        } else if constexpr (CT == ConstraintType::SAKOE_CHIBA) {
                            in_constraint = within_sakoe_chiba_band<R>(global_i, j, n, m);
                        } else if constexpr (CT == ConstraintType::ITAKURA) {
                            in_constraint = within_itakura_parallelogram<S>(global_i, j, n, m);
                        }

                        local_constraint_mask[local_i][j] = in_constraint;
                        if (in_constraint) valid_cols_per_row[local_i]++;
                    }
                }
                profiler.stop("Constraint Masking");
                // Row-based computation with pipelining and constraints
                profiler.start("Row Computation");
                // Each process maintains two rows: previous and current

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
                profiler.stop("Row Computation");

                // Gather final result
                profiler.start("Result Collection");
                double final_cost;

                if (rank == size - 1) {
                    final_cost = prev_row[m];
                    if (rank > 0) {
                        MPI_Send(&final_cost, 1, MPI_DOUBLE, 0, 0, comm);
                    }
                } else if (rank == 0) {
                    MPI_Recv(&final_cost, 1, MPI_DOUBLE, size - 1, 0, comm, MPI_STATUS_IGNORE);
                }
                profiler.trackReceive(sizeof(double));
                profiler.trackSend(sizeof(double));
                profiler.stop("Result Collection");

                // Broadcast final cost
                MPI_Bcast(&final_cost, 1, MPI_DOUBLE, 0, comm);

                // Check if we have a valid path
                if (final_cost == INF) {
                    return {INF, {}};
                }


                // Only root performs backtracking
                profiler.start("Backtracking");
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
                profiler.stop("Backtracking");
                profiler.stop("Total Execution");

                return {final_cost, path};

            }






            // Test runner function
            inline void run_dtw_profiler_tests(
                    const std::vector<std::vector<double>>& A,
                    const std::vector<std::vector<double>>& B,
                    bool save_to_file = true,
                    MPI_Comm comm = MPI_COMM_WORLD) {

                int rank;
                MPI_Comm_rank(comm, &rank);

                // Run wavefront implementation
                MPIProfiler mpi_profiler("MPI_DTW");
                auto [cost1, path1] = dtw_mpi_profiled(A, B, mpi_profiler, comm);

                if (rank == 0) {
                    std::cout << "\nMPI DTW result: " << cost1 << std::endl;
                    mpi_profiler.reportToConsole();
                }

                if (save_to_file) {
                    mpi_profiler.saveToFile("dtw_mpi_profile");
                }

                MPI_Barrier(comm);
                if (rank == 0) std::cout << "\n----------------------------------------\n" << std::endl;

                // Run sequential implementation

                MPIProfiler sequential_profiler("Sequential_DTW");
                sequential_profiler.start("Total Execution");
                auto [cost_seq, path_seq] = core::dtw_cpu(A, B);
                sequential_profiler.stop("Total Execution");
                if (rank == 0) {
                    std::cout << "Sequential DTW result: " << cost_seq << std::endl;
                    sequential_profiler.reportToConsole();
                }
                if (save_to_file) {
                    sequential_profiler.saveToFile("dtw_sequential_profile");
                }
                MPI_Barrier(comm);
                if (rank == 0) std::cout << "\n----------------------------------------\n" << std::endl;




                // Performance comparison
                if (rank == 0) {
                    std::cout << "\n===============================================" << std::endl;
                    std::cout << "PERFORMANCE COMPARISON" << std::endl;
                    std::cout << "===============================================" << std::endl;

                    double wavefront_time = 0, blocked_time = 0;
                    for (const auto& [section, entry] : mpi_profiler.timings) {
                        if (section == "Total Execution") {
                            wavefront_time = entry.duration * 1000.0;
                            break;
                        }
                    }

                    for (const auto& [section, entry] : sequential_profiler.timings) {
                        if (section == "Total Execution") {
                            blocked_time = entry.duration * 1000.0;
                            break;
                        }
                    }

                    std::cout << "MPI implementation: " << std::fixed << std::setprecision(1)
                              << wavefront_time << " ms" << std::endl;
                    std::cout << "Sequential implementation: " << std::fixed << std::setprecision(1)
                              << blocked_time << " ms" << std::endl;

                    if (wavefront_time < blocked_time) {
                        std::cout << "The MPI implementation is "
                                  << std::fixed << std::setprecision(1)
                                  << blocked_time / wavefront_time
                                  << "x faster than the Sequential implementation." << std::endl;
                    } else {
                        std::cout << "The Sequential implementation is "
                                  << std::fixed << std::setprecision(1)
                                  << wavefront_time / blocked_time
                                  << "x faster than the MPI implementation." << std::endl;
                    }

                    std::cout << "===============================================" << std::endl;
                }
            }

            template<constraints::ConstraintType CT, int R = 1, double S = 2.0>
            inline void run_dtw_profiler_tests_with_constraints(
                    const std::vector<std::vector<double>>& A,
                    const std::vector<std::vector<double>>& B,
                    bool save_to_file = true,
                    MPI_Comm comm = MPI_COMM_WORLD) {

                int rank;
                MPI_Comm_rank(comm, &rank);

                // Run wavefront implementation
                MPIProfiler mpi_profiler_with_constraints("MPI_DTW_With_Constraints");
                auto [cost1, path1] = dtw_mpi_constrained_profiled<CT, R, S>(A, B, mpi_profiler_with_constraints, comm);

                if (rank == 0) {
                    std::cout << "\nMPI DTW with constraints result: " << cost1 << std::endl;
                    mpi_profiler_with_constraints.reportToConsole();
                }

                if (save_to_file) {
                    mpi_profiler_with_constraints.saveToFile("dtw_profile_with_constraints");
                }

                MPI_Barrier(comm);
                if (rank == 0) std::cout << "\n----------------------------------------\n" << std::endl;

                // Run sequential implementation

                MPIProfiler sequential_profiler("Sequential_DTW_Constrained");
                sequential_profiler.start("Total Execution");
                auto [cost_seq, path_seq] = core::dtw_with_constraint<CT, R, S>(A, B);
                sequential_profiler.stop("Total Execution");
                if (rank == 0) {
                    std::cout << "Sequential DTW with constraints result: " << cost_seq << std::endl;
                    sequential_profiler.reportToConsole();
                }
                if (save_to_file) {
                    sequential_profiler.saveToFile("dtw_sequential_constrained_profile");
                }
                MPI_Barrier(comm);
                if (rank == 0) std::cout << "\n----------------------------------------\n" << std::endl;




                // Performance comparison
                if (rank == 0) {
                    std::cout << "\n===============================================" << std::endl;
                    std::cout << "PERFORMANCE COMPARISON" << std::endl;
                    std::cout << "===============================================" << std::endl;

                    double wavefront_time = 0, blocked_time = 0;
                    for (const auto& [section, entry] : mpi_profiler_with_constraints.timings) {
                        if (section == "Total Execution") {
                            wavefront_time = entry.duration * 1000.0;
                            break;
                        }
                    }

                    for (const auto& [section, entry] : sequential_profiler.timings) {
                        if (section == "Total Execution") {
                            blocked_time = entry.duration * 1000.0;
                            break;
                        }
                    }

                    std::cout << "MPI implementation with constraints: " << std::fixed << std::setprecision(1)
                              << wavefront_time << " ms" << std::endl;
                    std::cout << "Sequential implementation with constraints: " << std::fixed << std::setprecision(1)
                              << blocked_time << " ms" << std::endl;

                    if (wavefront_time < blocked_time) {
                        std::cout << "The MPI implementation is "
                                  << std::fixed << std::setprecision(1)
                                  << blocked_time / wavefront_time
                                  << "x faster than the Sequential implementation." << std::endl;
                    } else {
                        std::cout << "The Sequential implementation is "
                                  << std::fixed << std::setprecision(1)
                                  << wavefront_time / blocked_time
                                  << "x faster than the MPI implementation." << std::endl;
                    }

                    std::cout << "===============================================" << std::endl;
                }
            }

        } // namespace mpi
    } // namespace parallel
} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_PARALLEL_MPI_DTW_PROFILER_HPP