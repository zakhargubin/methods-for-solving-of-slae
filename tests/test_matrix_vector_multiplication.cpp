#include "dense_matrix/dense_matrix.hpp"
#include "csr_matrix/csr_matrix.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

bool close(slae::scalar_type x,
           slae::scalar_type y,
           slae::scalar_type eps = 1e-12) {
    return std::abs(x - y) < eps;
}

bool vectors_close(const slae::vector_type& lhs,
                   const slae::vector_type& rhs,
                   slae::scalar_type eps = 1e-12) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (slae::index_type i = 0; i < lhs.size(); ++i) {
        if (!close(lhs[i], rhs[i], eps)) {
            return false;
        }
    }

    return true;
}

}

int main() {
    slae::DenseMatrix dense(3, 3);
    dense(0, 0) = 1.0;
    dense(0, 2) = 2.0;
    dense(1, 1) = 3.0;
    dense(2, 0) = 4.0;
    dense(2, 2) = 6.0;

    const std::vector<slae::DokEntry> entries = {
        {0, 0, 1.0},
        {0, 2, 2.0},
        {1, 1, 3.0},
        {2, 0, 4.0},
        {2, 2, 6.0}
    };

    const slae::CSRMatrix csr(3, 3, entries);

    const slae::vector_type x = {1.0, 2.0, 3.0};
    const slae::vector_type expected = {7.0, 6.0, 22.0};

    const slae::vector_type dense_result = dense * x;
    const slae::vector_type csr_result = csr * x;

    if (!vectors_close(dense_result, expected)) {
        std::cout << "fail: dense matrix-vector multiplication is wrong\n";
        return 1;
    }

    if (!vectors_close(csr_result, expected)) {
        std::cout << "fail: CSR matrix-vector multiplication is wrong\n";
        return 1;
    }

    if (!vectors_close(dense_result, csr_result)) {
        std::cout << "fail: dense and CSR matrix-vector results differ\n";
        return 1;
    }

    std::cout << "success\n";
    return 0;
}
