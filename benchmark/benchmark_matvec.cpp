#include "csr_matrix/csr_matrix.hpp"
#include "dense_matrix/dense_matrix.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

slae::DenseMatrix make_dense_matrix(slae::index_type size, slae::scalar_type density, std::mt19937_64& rng) {
    slae::DenseMatrix matrix(size, size);
    std::uniform_real_distribution<slae::scalar_type> prob(0.0, 1.0);
    std::uniform_real_distribution<slae::scalar_type> value_dist(-1.0, 1.0);

    for (slae::index_type i = 0; i < size; ++i) {
        for (slae::index_type j = 0; j < size; ++j) {
            if (prob(rng) < density) {
                matrix(i, j) = value_dist(rng);
            }
        }
    }
    return matrix;
}

std::vector<slae::DokEntry> dense_to_dok(const slae::DenseMatrix& matrix) {
    std::vector<slae::DokEntry> entries;
    for (slae::index_type i = 0; i < matrix.nrows(); ++i) {
        for (slae::index_type j = 0; j < matrix.ncols(); ++j) {
            const slae::scalar_type value = matrix(i, j);
            if (std::abs(value) > slae::tolerance) {
                entries.push_back({i, j, value});
            }
        }
    }
    return entries;
}

slae::vector_type make_vector(slae::index_type size, std::mt19937_64& rng) {
    std::uniform_real_distribution<slae::scalar_type> value_dist(-1.0, 1.0);
    slae::vector_type x(size, 0.0);
    for (slae::index_type i = 0; i < size; ++i) {
        x[i] = value_dist(rng);
    }
    return x;
}

template<typename Function>
slae::scalar_type measure_ms(Function&& function, slae::index_type repeats) {
    volatile slae::scalar_type sink = 0.0;
    const auto start = std::chrono::steady_clock::now();
    for (slae::index_type r = 0; r < repeats; ++r) {
        const slae::vector_type y = function();
        sink += y.empty() ? 0.0 : y[0];
    }
    const auto finish = std::chrono::steady_clock::now();
    const std::chrono::duration<slae::scalar_type, std::milli> elapsed = finish - start;
    return elapsed.count() / static_cast<slae::scalar_type>(repeats);
}

}

int main(int argc, char** argv) {
    const std::string output_path = (argc > 1) ? argv[1] : "benchmark/matvec_results.csv";

    std::ofstream out(output_path);
    if (!out) {
        std::cerr << "Cannot open output file: " << output_path << '\n';
        return 1;
    }

    out << "size,density,nnz,dense_ms,csr_ms\n";

    std::mt19937_64 rng(42);
    const std::vector<slae::index_type> sizes = {128, 256, 512, 1024};
    const std::vector<slae::scalar_type> densities = {0.05, 0.20, 0.80};

    for (const slae::index_type size : sizes) {
        for (const slae::scalar_type density : densities) {
            const slae::DenseMatrix dense = make_dense_matrix(size, density, rng);
            const std::vector<slae::DokEntry> dok = dense_to_dok(dense);
            const slae::CSRMatrix csr(size, size, dok);
            const slae::vector_type x = make_vector(size, rng);

            const slae::index_type repeats = (size <= 256) ? 200 : (size <= 512 ? 100 : 30);

            const slae::scalar_type dense_ms = measure_ms([&]() {
                return dense * x;
            }, repeats);

            const slae::scalar_type csr_ms = measure_ms([&]() {
                return csr * x;
            }, repeats);

            out << size << ','
                << std::fixed << std::setprecision(2) << density << ','
                << csr.nnz() << ','
                << std::setprecision(6) << dense_ms << ','
                << csr_ms << '\n';
        }
    }

    return 0;
}
