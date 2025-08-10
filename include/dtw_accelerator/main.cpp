#include "dtw_accelerator.hpp"
#include <iostream>
#include <random>
#include <chrono>

// function to generate random time series data
std::vector<std::vector<double>> generate_random_series(int length, int dimensions, double min_val = 0.0, double max_val = 100.0) {
    std::vector<std::vector<double>> series(length, std::vector<double>(dimensions));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min_val, max_val);

    for (int i = 0; i < length; i++) {
        for (int j = 0; j < dimensions; j++) {
            series[i][j] = dist(gen);
        }
    }

    return series;
}

template<typename Func>
auto measure_time(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    auto result = func();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    return std::make_pair(result, duration);
}

int main() {
    int sequence_length;
    int dimensions = 2;

    std::cout << "Inserisci la lunghezza delle sequenze temporali: ";
    std::cin >> sequence_length;

    std::vector<std::vector<double>> series_a = generate_random_series(sequence_length, dimensions);
    std::vector<std::vector<double>> series_b = generate_random_series(sequence_length, dimensions);

    std::cout << "Serie temporali generate con lunghezza " << sequence_length << std::endl;

    // CPU version (always available)
    std::cout << "=== DTW Base CPU ===" << std::endl;
    auto [result_dtw, time_dtw] = measure_time([&]() {
        return dtw_accelerator::dtw_cpu(series_a, series_b);
    });
    auto& [cost_dtw, path_dtw] = result_dtw;
    std::cout << "DTW CPU result: " << cost_dtw << " (tempo: " << time_dtw << " ms)" << std::endl;


    std::cout << "\n=== DTW with Constraint CPU ===" << std::endl;
    // Using dtw_with_constraint with Sakoe-Chiba
    auto [result_wc_sc, time_wc_sc] = measure_time([&]() {
        return dtw_accelerator::core::dtw_with_constraint<dtw_accelerator::constraints::ConstraintType::SAKOE_CHIBA, 3>(series_a, series_b);
    });
    auto& [cost_dtw_wc_sc, path_dtw_wc_sc] = result_wc_sc;
    std::cout << "DTW with constraint (Sakoe-Chiba R=3): " << cost_dtw_wc_sc << " (tempo: " << time_wc_sc << " ms)" << std::endl;

    // Using dtw_with_constraint with Itakura: TODO: TO FIX (RIGHT NOW IT DOESN'T ALWAYS FIND A PATH)
//    auto [result_wc_it, time_wc_it] = measure_time([&]() {
//        return fastdtw::core::dtw_with_constraint<fastdtw::constraints::ConstraintType::ITAKURA, 0, 1.5>(series_a, series_b);
//    });
//    auto& [cost_dtw_wc_it, path_dtw_wc_it] = result_wc_it;
//    std::cout << "DTW with constraint (Itakura S=1.5): " << cost_dtw_wc_it << " (tempo: " << time_wc_it << " ms)" << std::endl;

    std::cout << "\n=== FastDTW CPU ===" << std::endl;
    // Basic FastDTW
    auto [result_fastdtw, time_fastdtw] = measure_time([&]() {
        return dtw_accelerator::fastdtw_cpu(series_a, series_b);
    });
    auto& [cost_fastdtw, path_fastdtw] = result_fastdtw;
    std::cout << "FastDTW CPU result: " << cost_fastdtw << " (tempo: " << time_fastdtw << " ms)" << std::endl;

    // FastDTW with radius parameter
    auto [result_fastdtw_r2, time_fastdtw_r2] = measure_time([&]() {
        return dtw_accelerator::fastdtw_cpu(series_a, series_b, 3);
    });
    auto& [cost_fastdtw_r2, path_fastdtw_r2] = result_fastdtw_r2;
    std::cout << "FastDTW CPU with radius= 3: " << cost_fastdtw_r2 << " (tempo: " << time_fastdtw_r2 << " ms)" << std::endl;


    return 0;
}



