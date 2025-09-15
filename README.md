# DTW Accelerator v1.0

A high-performance Dynamic Time Warping (DTW) library with a modern C++20 concepts-based architecture supporting multiple parallel backends.

## 🚀 Key Features

- **Modern C++20 Concepts**: Type-safe, compile-time polymorphic interface using execution strategies
- **Multiple Backends**: Sequential, Blocked, OpenMP, MPI, and CUDA implementations
- **Unified Interface**: Single algorithm implementation works with all execution strategies
- **Performance Optimized**: Cache-aware blocking, vectorization, wavefront parallelization
- **Memory Efficient**: Contiguous memory layouts with `TimeSeries` and `Matrix` classes
- **Flexible Distance Metrics**: Euclidean, Manhattan, Chebyshev, and Cosine distances
- **Constraint Support**: Sakoe-Chiba band, Itakura parallelogram, and custom windows
- **FastDTW Implementation**: Approximate DTW for large sequences with recursive coarsening

## 🏗️ Architecture Overview

### Concepts-Based Design

The library uses C++20 concepts to define a unified interface for execution strategies:

```cpp
template<typename Strategy>
concept ExecutionStrategy = requires(Strategy s, DoubleMatrix& D, 
                                    const DoubleTimeSeries& A, 
                                    const DoubleTimeSeries& B,
                                    int n, int m, int dim) {
    { s.initialize_matrix(D, n, m) } -> std::same_as<void>;
    { s.template execute_with_contraint<MetricType::EUCLIDEAN>(D, A, B, n, m, dim) } -> std::same_as<void>;
    { s.extract_result(D) } -> std::convertible_to<std::pair<double, std::vector<std::pair<int, int>>>>;
    { s.name() } -> std::convertible_to<std::string_view>;
    { s.is_parallel() } -> std::convertible_to<bool>;
};
```

### Core Components

#### Data Structures
- **`TimeSeries<T>`**: Contiguous memory time series with row-major layout
- **`Matrix<T>`**: Contiguous 2D matrix implementation for DTW cost matrix
- **`DoubleTimeSeries`/`DoubleMatrix`**: Type aliases for double precision

#### Execution Strategies
1. **CPU Strategies** (Concepts-based):
    - `SequentialStrategy`: Simple sequential implementation
    - `BlockedStrategy`: Cache-optimized blocked execution
    - `OpenMPStrategy`: Multi-threaded parallel execution with wavefront parallelization
    - `MPIStrategy`: Distributed computing across nodes with block distribution
    - `AutoStrategy`: Automatic strategy selection based on problem size

2. **GPU Strategy** (Specialized):
    - `CUDAStrategy`: Optimized CUDA kernels

#### Algorithm Variants
- **Standard DTW**: Full dynamic programming solution
- **FastDTW**: Recursive coarsening with constrained refinement
- **Constrained DTW**: Window-based constraints for optimization
- **Templated Constraints**: Compile-time constraint specification

### Directory Structure

```
dtw-accelerator/
├── include/dtw_accelerator/
│   ├── core/
│   │   ├── dtw_concepts.hpp         # C++20 concepts definitions
│   │   ├── time_series.hpp          # TimeSeries class
│   │   ├── matrix.hpp                # Matrix class
│   │   ├── distance_metrics.hpp     # Distance metric implementations
│   │   ├── constraints.hpp          # Constraint definitions
│   │   ├── path_processing.hpp      # Path utilities
│   │   └── dtw_utils.hpp           # Common utilities
│   ├── execution/
│   │   ├── base_strategy.hpp        # CRTP base for strategies
│   │   ├── execution_strategies.hpp        
│   │   ├── sequential/
│   │   │   ├── standard_strategy.hpp
│   │   │   └── blocked_strategy.hpp
│   │   ├── parallel/
│   │   │   ├── openmp/
│   │   │   │   └── openmp_strategy.hpp
│   │   │   ├── mpi/
│   │   │   │   ├── mpi_debug_strategy.hpp 
│   │   │   │   └── mpi_strategy.hpp
│   │   │   └── cuda/
│   │   │         ├── cuda_dtw.hpp
│   │   │         ├── core/
│   │   │         │   ├── cuda_memory.hpp
│   │   │         │   └── device_functions.cuh
│   │   │         ├── execution/
│   │   │         │   ├── cuda_launcher.cu
│   │   │         │   ├── cuda_launcher.hpp
│   │   │         │   └── cuda_strategy.hpp
│   │   │         └── kernels/
│   │   │             ├── dtw_core_kernels.cu
│   │   │             ├── dtw_core_kernels.cuh
│   │   │             ├── matrix_kernels.cu
│   │   │             ├── matrix_kernels.cuh
│   │   │             ├── path_kernels.cu
│   │   │             └── path_kernels.cuh
│   │   └── auto_strategy.hpp
│   ├── algorithms/
│   │   ├── dtw_generic.hpp          # Generic DTW algorithm
│   │   ├── fastdtw_generic.hpp      # FastDTW implementation
│   └── dtw_accelerator.hpp          # Main header file
├── tests/
│   ├── test_core.cpp
│   ├── test_strategies.cpp
│   ├── test_distance_metrics.cpp
│   ├── test_constraints.cpp
│   └── performance/
│       ├── benchmark_cuda_only.cpp
│       ├── benchmark_mpi_only.cpp
│       ├── benchmark_openmp_only.cpp
│       ├── benchmark_sequential_pure.cpp
│       ├── plot_results_updated.py
│       └── benchmark_scaling.cpp
├── build.sh
├── run_fair_benchmarks.sh
├── run_unit_tests.sh
├── run_auto_test.sh
├── setup_python.sh
├── ReadMe.md
└── CMakeLists.txt
```

## 📦 Installation

### Requirements

- C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)
- CMake 3.26 or higher (between 3.26 and 4.0.0 if using Clion) 
- Python 3.10+ (Optional: for benchmarks and visualization)

```bash
# Ubuntu/Debian
  sudo apt-get install cmake
```
- Optional Dependencies:
  Google Test (for unit tests)
```bash
# Ubuntu/Debian
sudo apt-get install libgtest-dev

# From source
git clone https://github.com/google/googletest.git
cd googletest
mkdir build && cd build
cmake ..
make
sudo make install
```
OpenMP (for multi-threaded CPU execution)
```bash
# Ubuntu/Debian
sudo apt-get install libomp-dev
```
MPI (for distributed computing)
```bash
# Ubuntu/Debian (OpenMPI)
sudo apt-get install libopenmpi-dev openmpi-bin
```
CUDA Toolkit 12+ (for GPU acceleration)
```bash
# Download from NVIDIA: https://developer.nvidia.com/cuda-downloads

# Ubuntu example (CUDA 12.3)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-3

# Verify installation
nvcc --version
nvidia-smi
```
## 🐍 Python Environment Setup (Optional: only for benchmarks)

The project includes Python scripts for benchmark visualization and analysis. Follow these steps to set up the Python environment:

### Quick Setup
```bash
# Run the setup script (creates virtual environment and installs dependencies)
chmod +x setup_python.sh
./setup_python.sh

# Activate the environment
source dtw_env/bin/activate

# When done, deactivate
deactivate
```
### Manual Setup
If you prefer to set up the environment manually:
```bash
# Create virtual environment
python3 -m venv dtw_env

# Activate environment
source dtw_env/bin/activate

# Install required packages
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn

# Deactivate when done
deactivate
```


## ⚙️ Building the Project

### Quick Build with Scripts

The project includes convenient build scripts for streamlined compilation:

```bash
# Clone the repository
git clone https://github.com/UobinoPino/DTW-Accelerator.git
cd dtw-accelerator

# Make scripts executable
chmod +x build.sh run_fair_benchmarks.sh run_unit_tests.sh

# Build the entire project
./build.sh

# Run unit tests
./run_unit_tests.sh

# Run comprehensive benchmarks and generate plots
./run_fair_benchmarks.sh
```

### Manual Build with CMake
For more control over the build process:
```bash
# Create build directory
mkdir build && cd build

# Configure with desired backends
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DUSE_OPENMP=ON \
         -DUSE_MPI=ON \
         -DUSE_CUDA=ON \
         -DBUILD_TESTS=ON \
         -DBUILD_BENCHMARKS=ON

# Build
make -j$(nproc)

# Run tests
ctest --output-on-failure

# Install (optional)
sudo make install
```

## 🔧 Troubleshooting

### Script Execution Issues (Windows/WSL Users)

If you encounter an error like `/bin/bash^M: bad interpreter: No such file or directory` when running shell scripts, this indicates Windows line endings (CRLF) in the scripts. Convert them to Unix format:

```bash
# Install dos2unix utility
sudo apt install dos2unix

# Convert all shell scripts to Unix format
dos2unix build.sh
dos2unix run_fair_benchmarks.sh
dos2unix run_unit_tests.sh
dos2unix run_auto_test.sh
dos2unix setup_python.sh

# Make scripts executable
chmod +x *.sh

```
## 🎯 Usage Examples
### Layered Interface Approach
DTW Accelerator provides multiple interface layers, allowing users to choose the appropriate level of control for their needs:
### 1. Simple Interface (Automatic Backend Selection)
   For common use cases, the library automatically selects the best backend:
```cpp
#include "dtw_accelerator/dtw_accelerator.hpp"

using namespace dtw_accelerator;

int main() {
    // Create time series
    DoubleTimeSeries series_a(1000, 3);  // 1000 points, 3 dimensions
    DoubleTimeSeries series_b(1000, 3);
    
    // ... fill with data ...
    
    // Automatic backend selection based on problem size
    auto result = dtw(series_a, series_b);
    
    std::cout << "DTW Distance: " << result.first << std::endl;
    std::cout << "Path length: " << result.second.size() << std::endl;
    
    return 0;
}
```
### 2. Backend-Specific Optimization
   When you need/want control over specific backends:
```cpp
#include "dtw_accelerator/dtw_accelerator.hpp"

using namespace dtw_accelerator;

int main() {
    DoubleTimeSeries series_a(5000, 3);
    DoubleTimeSeries series_b(5000, 3);
    // ... fill with data ...
    
    // CPU: Sequential for small data
    auto seq_result = cpu::dtw<MetricType::EUCLIDEAN>(series_a, series_b);
    
    // CPU: Cache-optimized blocked execution
    auto blocked_result = cpu::dtw_blocked<MetricType::EUCLIDEAN>(
        series_a, series_b, 64);  // 64 = block size
    
#ifdef USE_OPENMP
    // Multi-threaded with OpenMP
    auto omp_result = openmp::dtw<MetricType::MANHATTAN>(
        series_a, series_b, 
        4,    // number of threads
        128   // block size
    );
#endif

#ifdef USE_CUDA
    // GPU acceleration for large sequences
    if (cuda::is_available()) {
        auto gpu_result = cuda::dtw_cuda<MetricType::EUCLIDEAN>(
            series_a, series_b, 64);  // tile size
        
        std::cout << "GPU Device: " << cuda::device_info() << std::endl;
    }
#endif

#ifdef USE_MPI
    // Distributed computation across nodes
    auto mpi_result = mpi::dtw<MetricType::CHEBYSHEV>(
        series_a, series_b,
        64,   // block size
        4,    // threads per process
        MPI_COMM_WORLD
    );
#endif
    
    return 0;
}
```
### 3. Custom Strategy Injection
   For specialized requirements, you can implement custom execution strategies:

```cpp
#include "dtw_accelerator/dtw_accelerator.hpp"

using namespace dtw_accelerator;

// Custom strategy with specialized optimization
class MyCustomStrategy : public execution::BaseStrategy<MyCustomStrategy> {
public:
    template<constraints::ConstraintType CT, int R, double S, 
             distance::MetricType M>
    void execute_with_constraint(DoubleMatrix& D,
                                 const DoubleTimeSeries& A,
                                 const DoubleTimeSeries& B,
                                 int n, int m, int dim,
                                 const WindowConstraint* window) const {
        // Your custom implementation
        // For example: SIMD vectorization, custom memory layout, etc.
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= m; ++j) {
                D(i, j) = utils::compute_cell_cost<M>(
                    A[i-1], B[j-1], dim,
                    D(i-1, j-1), D(i, j-1), D(i-1, j)
                );
            }
        }
    }
    
    std::string_view name() const { return "MyCustom"; }
    bool is_parallel() const { return false; }
};

int main() {
    DoubleTimeSeries series_a(1000, 3);
    DoubleTimeSeries series_b(1000, 3);
    // ... fill with data ...
    
    // Use custom strategy
    MyCustomStrategy custom;
    auto result = dtw_custom<MetricType::EUCLIDEAN>(
        series_a, series_b, custom);
    
    std::cout << "Result with custom strategy: " << result.first << std::endl;
    
    return 0;
}
```

## Automatic Strategy Selection
The AutoStrategy class implements intelligent backend selection based on problem characteristics:
```cpp
class AutoStrategy : public BaseStrategy<AutoStrategy> {
        private:
            /// @brief Problem dimensions for strategy selection
            int n_, m_;

            /// @brief Selected strategy name for reporting
            mutable std::string_view selected_name_;

            /// @brief Flag for parallel execution
            mutable bool is_parallel_;

            /// @brief Strategy variant to hold any strategy type
            using StrategyVariant = std::variant<
                    SequentialStrategy,
                    BlockedStrategy
#ifdef USE_OPENMP
                    , OpenMPStrategy
#endif
#ifdef USE_MPI
                    , MPIStrategy
#endif
#ifdef USE_CUDA
                    , parallel::cuda::CUDAStrategy
#endif
            >;

            /// @brief The actual selected strategy
            mutable std::unique_ptr<StrategyVariant> strategy_;

            /**
             * @brief Select appropriate strategy based on problem size
             *
             * This method implements the heuristics for automatic
             * strategy selection. Called lazily on first use.
             */
            void select_strategy() const {
                if (strategy_) return;  // Already selected

#ifdef USE_CUDA
                // Check CUDA availability for large problems
                if (n_ >= 1000 && m_ >= 1000) {
                    try {
                        if (parallel::cuda::CUDAStrategy::is_available()) {
                            strategy_ = std::make_unique<StrategyVariant>(
                                parallel::cuda::CUDAStrategy(256)  // Default tile size
                            );
                            selected_name_ = "CUDA-Auto";
                            is_parallel_ = true;
                            return;
                        }
                    } catch (...) {
                        // CUDA initialization failed, fall through to next option
                    }
                }
#endif

#ifdef USE_MPI
                // Check for MPI - typically not auto-selected unless explicitly in MPI context
                // MPI requires special initialization, so we generally don't auto-select it
                // unless we detect we're already in an MPI environment
                int flag = 0;
                MPI_Initialized(&flag);
                if (flag && n_ >= 500 && m_ >= 500) {
                    int size;
                    MPI_Comm_size(MPI_COMM_WORLD, &size);
                    if (size > 1) {  // Only use MPI if we have multiple processes
                        strategy_ = std::make_unique<StrategyVariant>(
                            MPIStrategy(64, 0, MPI_COMM_WORLD)
                        );
                        selected_name_ = "MPI-Auto";
                        is_parallel_ = true;
                        return;
                    }
                }
#endif

#ifdef USE_OPENMP
                // Use OpenMP for medium to large problems
                if (n_ >= 100 && m_ >= 100) {
                    int num_threads = omp_get_max_threads();
                    if (num_threads > 1) {  // Only use if we have multiple threads available
                        strategy_ = std::make_unique<StrategyVariant>(
                            OpenMPStrategy(0, 64)  // 0 means auto-detect threads
                        );
                        selected_name_ = "OpenMP-Auto";
                        is_parallel_ = true;
                        return;
                    }
                }
#endif

                // For medium problems, use blocked strategy for cache optimization
                if (n_ >= 50 && m_ >= 50) {
                    strategy_ = std::make_unique<StrategyVariant>(
                            BlockedStrategy(64)
                    );
                    selected_name_ = "Blocked-Auto";
                    is_parallel_ = false;
                    return;
                }

                // Default to sequential for small problems
                strategy_ = std::make_unique<StrategyVariant>(SequentialStrategy{});
                selected_name_ = "Sequential-Auto";
                is_parallel_ = false;
            }

        public:
            /**
             * @brief Construct auto-strategy with problem dimensions
             * @param n First series length
             * @param m Second series length
             */
            explicit AutoStrategy(int n = 0, int m = 0)
                    : n_(n), m_(m), selected_name_("Unknown"), is_parallel_(false) {}

            /**
             * @brief Initialize DTW matrix
             * @param D Matrix to initialize
             * @param n Number of rows
             * @param m Number of columns
             */
            void initialize_matrix(DoubleMatrix& D, int n, int m) const {
                // Update dimensions if they were not set in constructor


                select_strategy();

                std::visit([&D, n, m](auto& strategy) {
                    strategy.initialize_matrix(D, n, m);
                }, *strategy_);
            }

            /**
             * @brief Execute DTW computation with constraints
             * @tparam CT Constraint type
             * @tparam R Sakoe-Chiba radius
             * @tparam S Itakura slope
             * @tparam M Distance metric type
             * @param D Cost matrix
             * @param A First time series
             * @param B Second time series
             * @param n Series A length
             * @param m Series B length
             * @param dim Dimensions per point
             * @param window Optional window constraint
             */
            template<constraints::ConstraintType CT, int R = 1, double S = 2.0,
                    distance::MetricType M = distance::MetricType::EUCLIDEAN>
            void execute_with_constraint(DoubleMatrix& D,
                                         const DoubleTimeSeries& A,
                                         const DoubleTimeSeries& B,
                                         int n, int m, int dim,
                                         const WindowConstraint* window = nullptr) const {
                // Update dimensions if needed


                select_strategy();

                std::visit([&](auto& strategy) {
                    strategy.template execute_with_constraint<CT, R, S, M>(
                            D, A, B, n, m, dim, window);
                }, *strategy_);
            }

            /**
             * @brief Extract result from the cost matrix
             * @param D Computed cost matrix
             * @return Pair of (distance, path)
             */
            std::pair<double, std::vector<std::pair<int, int>>>
            extract_result(const DoubleMatrix& D) const {
                select_strategy();

                return std::visit([&D](auto& strategy) {
                    return strategy.extract_result(D);
                }, *strategy_);
            }

            /**
             * @brief Get the name of the selected strategy
             * @return Strategy name
             */
            std::string_view name() const {
                select_strategy();
                return selected_name_;
            }

            /**
             * @brief Check if selected strategy uses parallel execution
             * @return True if parallel
             */
            bool is_parallel() const {
                select_strategy();
                return is_parallel_;
            }
        };
```

## 🔧 Advanced Features Examples

### Using Path Constraints
```cpp
// Sakoe-Chiba band constraint
auto result_band = dtw_sakoe_chiba<MetricType::EUCLIDEAN, 10>(
    series_a, series_b, execution::AutoStrategy{}
);

// Itakura parallelogram constraint  
auto result_para = dtw_itakura<MetricType::EUCLIDEAN, 2.0>(
    series_a, series_b, execution::OpenMPStrategy{4, 64}
);

// Custom window constraint
WindowConstraint window;
for (int i = 0; i < n; ++i) {
    for (int j = std::max(0, i-5); j <= std::min(m-1, i+5); ++j) {
        window.push_back({i, j});
    }
}
auto result_window = dtw_windowed<MetricType::EUCLIDEAN>(
    series_a, series_b, window, execution::BlockedStrategy{32}
);
```
### FastDTW Approximation
```cpp
// FastDTW for O(n) complexity
int radius = 2;        // Search window radius
int min_size = 100;    // Minimum size for recursion

// Auto-select best backend
auto fast_result = fastdtw(series_a, series_b, radius, min_size);

// Or specify backend explicitly
auto fast_omp = fastdtw_openmp<MetricType::EUCLIDEAN>(
    series_a, series_b, radius, min_size, 4, 64
);
```
### Different Distance Metrics
```cpp
// Euclidean distance (default)
auto euclidean = dtw<MetricType::EUCLIDEAN>(series_a, series_b);

// Manhattan distance
auto manhattan = dtw<MetricType::MANHATTAN>(series_a, series_b);

// Chebyshev distance
auto chebyshev = dtw<MetricType::CHEBYSHEV>(series_a, series_b);

// Cosine distance
auto cosine = dtw<MetricType::COSINE>(series_a, series_b);
```
## Testing
Running Unit Tests
```bash
# Using the convenience script
./run_unit_tests.sh

# Or manually

# Run specific test
./include/tests/test_core
./include/tests/test_strategies
./include/tests/test_distance_metrics
./include/tests/test_constraints
```

Running Benchmarks
```bash
# Complete benchmark suite with visualization
./run_fair_benchmarks.sh

# Without MPI (if not installed)
./run_fair_benchmarks.sh --no-mpi

# Manual benchmark execution
cd build/include/tests/performance
./benchmark_cuda_only
./benchmark_mpi_only
./benchmark_openmp_only
./benchmark_sequential_pure

# Generate plots from results
python3 plot_results.py
```

The benchmark suite will:

- Test all available backends, 
- Compare performance across different problem sizes
- Generate performance plots in benchmark_plots/ directory
- Create a detailed report with speedup analysis


## 📊 Performance Characteristics

### Complexity
- **Time Complexity**: O(n×m) for standard DTW
- **Space Complexity**: O(n×m) for cost matrix
- **FastDTW**: O(n) approximate with controllable accuracy

### Memory Layout Benefits
- **TimeSeries**: Contiguous memory, better cache locality
- **Matrix**: Single allocation, reduced fragmentation
- **Direct pointer access**: Efficient for CUDA/MPI transfers

### Parallelization Strategies
- **OpenMP**: Wavefront parallelization with blocked processing
- **MPI**: Distributed block processing with boundary exchange
- **CUDA**: GPU acceleration for large sequences (>1000 points)

## 🏛️ Design Principles

### Concepts:
1. **Type Safety**: Compile-time verification of strategy interfaces
2. **Zero Overhead**: No virtual function calls or runtime polymorphism
3. **Extensibility**: Easy to add new strategies without modifying core
4. **Clear Contracts**: Concepts explicitly define requirements
5. **Better Error Messages**: Clearer compiler diagnostics

### Memory Efficiency
- Contiguous memory allocation reduces cache misses
- Row-major layout optimizes sequential access patterns
- Blocked processing improves temporal locality
- Minimal memory overhead compared to nested vectors

## 🗺️ Roadmap

- [ ] Distributed CUDA with NCCL
- [ ] Python bindings with pybind11


## 📄 License

See LICENSE file for details

## 📚 Citation

If you use DTW Accelerator in your research, please cite:

```bibtex
@software{dtw_accelerator,
  title = {DTW Accelerator: A Modern C++20 Concepts-Based Dynamic Time Warping Library},
  author = {UobinoPino},
  year = {2024},
  url = {https://github.com/UobinoPino/DTW-Accelerator}
}
```

## 🔗 References

1. [Dynamic Time Warping](https://en.wikipedia.org/wiki/Dynamic_time_warping)
2. [FastDTW Algorithm](https://cs.fit.edu/~pkc/papers/tdm04.pdf)
3. [C++20 Concepts](https://en.cppreference.com/w/cpp/language/constraints)

