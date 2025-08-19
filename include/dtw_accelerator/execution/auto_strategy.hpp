#ifndef DTWACCELERATOR_AUTO_STRATEGY_HPP
#define DTWACCELERATOR_AUTO_STRATEGY_HPP

#include "dtw_accelerator/core/dtw_concepts.hpp"
#include "dtw_accelerator/core/distance_metrics.hpp"
#include "dtw_accelerator/core/constraints.hpp"
#include "dtw_accelerator/core/dtw_utils.hpp"
#include <vector>
#include <utility>
#include <limits>
#include <algorithm>
#include <string_view>
#include <memory>
#include "dtw_accelerator/execution/base_strategy.hpp"
#include "dtw_accelerator/execution/sequential/standard_strategy.hpp"
#include "dtw_accelerator/execution/sequential/blocked_strategy.hpp"
#include "dtw_accelerator/execution/parallel/openmp/openmp_strategy.hpp"
#ifdef USE_OPENMP
#include <omp.h>
#endif

//#ifdef USE_MPI
#include <mpi.h>
//#endif


namespace dtw_accelerator {
    namespace execution {

        // Auto-selecting strategy based on problem size and available backends
        class AutoStrategy : public BaseStrategy<AutoStrategy> {
        private:
            mutable std::unique_ptr<SequentialStrategy> strategy_;
            int n_, m_;

            void select_strategy() const {
                if (strategy_) return;

#ifdef USE_OPENMP
                if (n_ >= 100 && m_ >= 100) {
                    auto omp_strat = std::make_unique<OpenMPStrategy>();
                    // Type erasure workaround - store as base type
                    strategy_ = std::make_unique<SequentialStrategy>();
                    return;
                }
#endif

                if (n_ >= 50 && m_ >= 50) {
                    strategy_ = std::make_unique<SequentialStrategy>();
                    // Could be BlockedStrategy for larger sizes
                    return;
                }

                strategy_ = std::make_unique<SequentialStrategy>();
            }

        public:
            explicit AutoStrategy(int n = 0, int m = 0) : n_(n), m_(m) {}

            void initialize_matrix(std::vector<std::vector<double>>& D, int n, int m) const {
                // n_ = n; m_ = m;
                select_strategy();
                strategy_->initialize_matrix(D, n, m);
            }

            template<distance::MetricType M>
            void execute(std::vector<std::vector<double>>& D,
                         const std::vector<std::vector<double>>& A,
                         const std::vector<std::vector<double>>& B,
                         int n, int m, int dim) const {
                select_strategy();
                strategy_->template execute<M>(D, A, B, n, m, dim);
            }

            template<distance::MetricType M>
            void execute_constrained(std::vector<std::vector<double>>& D,
                                     const std::vector<std::vector<double>>& A,
                                     const std::vector<std::vector<double>>& B,
                                     const std::vector<std::pair<int, int>>& window,
                                     int n, int m, int dim) const {
                select_strategy();
                strategy_->template execute_constrained<M>(D, A, B, window, n, m, dim);
            }

            std::pair<double, std::vector<std::pair<int, int>>>
            extract_result(const std::vector<std::vector<double>>& D) const {
                return strategy_->extract_result(D);
            }

            std::string_view name() const {
                select_strategy();
                return strategy_->name();
            }

            bool is_parallel() const {
                select_strategy();
                return strategy_->is_parallel();
            }
        };




    }
}


#endif //DTWACCELERATOR_AUTO_STRATEGY_HPP
