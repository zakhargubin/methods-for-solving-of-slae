#include "dense_matrix/dense_matrix.hpp"

#include <cmath>
#include <iostream>

namespace {

bool close(slae::scalar_type lhs, slae::scalar_type rhs, slae::scalar_type eps = 1e-12) {
    return std::abs(lhs - rhs) < eps;
}

}

int main() {
    slae::DenseMatrix matrix(2, 3);
    matrix(0, 0) = 1.0;
    matrix(0, 1) = 2.0;
    matrix(0, 2) = 3.0;
    matrix(1, 0) = 4.0;
    matrix(1, 1) = 5.0;
    matrix(1, 2) = 6.0;

    if (matrix.nrows() != 2 || matrix.ncols() != 3) {
        std::cout << "fail: wrong DenseMatrix size\n";
        return 1;
    }

    if (!close(matrix(1, 2), 6.0)) {
        std::cout << "fail: wrong DenseMatrix element access\n";
        return 1;
    }

    const slae::vector_type x = {1.0, 0.0, -1.0};
    const slae::vector_type y = matrix * x;

    if (y.size() != 2 || !close(y[0], -2.0) || !close(y[1], -2.0)) {
        std::cout << "fail: wrong DenseMatrix matvec result\n";
        return 1;
    }

    std::cout << "success\n";
    return 0;
}
