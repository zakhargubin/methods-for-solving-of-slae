#include "csr_matrix/csr_matrix.hpp"

#include <cmath>
#include <iostream>

namespace {

bool close(slae::scalar_type lhs, slae::scalar_type rhs, slae::scalar_type eps = 1e-12) {
    return std::abs(lhs - rhs) < eps;
}

}

int main() {
    const std::vector<slae::DokEntry> entries = {
        {0, 0, 1.0},
        {0, 2, 2.0},
        {1, 1, 3.0},
        {2, 0, 4.0},
        {2, 2, 5.0},
        {2, 2, 1.0},
        {1, 0, 0.0}
    };

    slae::CSRMatrix matrix(3, 3, entries);

    if (matrix.nrows() != 3 || matrix.ncols() != 3) {
        std::cout << "fail: wrong CSRMatrix size\n";
        return 1;
    }

    if (matrix.nnz() != 5) {
        std::cout << "fail: wrong CSRMatrix nnz\n";
        return 1;
    }

    if (!close(matrix(2, 2), 6.0) || !close(matrix(1, 2), 0.0)) {
        std::cout << "fail: wrong CSRMatrix element access\n";
        return 1;
    }

    const slae::vector_type x = {1.0, 2.0, 3.0};
    const slae::vector_type y = matrix * x;

    if (y.size() != 3 || !close(y[0], 7.0) || !close(y[1], 6.0) || !close(y[2], 22.0)) {
        std::cout << "fail: wrong CSRMatrix matvec result\n";
        return 1;
    }

    std::cout << "success\n";
    return 0;
}
