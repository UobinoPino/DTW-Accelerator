# DTW Accelerator v1.0 - Concepts-Based Architecture

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
    { s.template execute<MetricType::EUCLIDEAN>(D, A, B, n, m, dim) } -> std::same_as<void>;
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
│       ├── benchmark_strategies.cpp
│       └── benchmark_scaling.cpp
└── CMakeLists.txt
```

## 📦 Installation

### Requirements

- C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)
- CMake 3.20+
- Optional: OpenMP, MPI, CUDA Toolkit 12.0+

### Building from Source

```bash
git clone https://github.com/UobinoPino/DTW-Accelerator.git
cd dtw-accelerator
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
TODO

# Install
sudo make install
```

## 🎯 Usage Examples

TODO

### Custom Strategy Implementation

```cpp
class MyCustomStrategy : public BaseStrategy<MyCustomStrategy> {
public:
    template<distance::MetricType M>
    void execute(DoubleMatrix& D,
                 const DoubleTimeSeries& A,
                 const DoubleTimeSeries& B,
                 int n, int m, int dim) const {
        // Your custom implementation
        //.....
        
        .....//
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

// Use custom strategy
MyCustomStrategy custom;
auto result = dtw<MetricType::EUCLIDEAN>(series_a, series_b, custom);
```

## 🔧 Advanced Features

### MPI Usage

```bash
# Compile with MPI support
mpicxx -std=c++20 -DUSE_MPI your_program.cpp -o your_program

# Run with multiple processes
mpirun -np 4 ./your_program
```

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
4. [Wavefront Parallel Pattern](https://en.wikipedia.org/wiki/Wavefront_pattern)
