#include "elliptic_matrix/elliptic_matrix.hpp"

#include <cmath>
#include <iostream>

namespace {

bool close(slae::scalar_type lhs,
           slae::scalar_type rhs,
           slae::scalar_type eps = 1e-10) {
    return std::abs(lhs - rhs) < eps;
}

}

int main() {
    const slae::index_type side = 6;
    const slae::scalar_type a = -1.0;
    const slae::scalar_type b = 2.0;
    const slae::CSRMatrix A = slae::generate_elliptic_matrix(side, a, b);
    const slae::index_type total_size = side * side;

    if (A.nrows() != total_size || A.ncols() != total_size) {
        std::cout << "fail: elliptic matrix generator returned a matrix of wrong size\n";
        return 1;
    }
    if (!close(A(0, 0), 4.0) || !close(A(0, 1), -1.0) || !close(A(0, side), -1.0)) {
        std::cout << "fail: elliptic matrix generator produced wrong stencil coefficients\n";
        return 1;
    }
    if (!close(A(0, side - 1), 0.0) || !close(A(side - 1, side), 0.0)) {
        std::cout << "fail: elliptic matrix generator produced row-wrap couplings\n";
        return 1;
    }

    const slae::scalar_type lambda_min = slae::elliptic_lambda_min(side, a, b);
    const slae::scalar_type lambda_max = slae::elliptic_lambda_max(side, a, b);
    if (!(lambda_min > 0.0 && lambda_max > lambda_min)) {
        std::cout << "fail: elliptic eigenvalue helpers returned invalid spectrum bounds\n";
        return 1;
    }

    const slae::scalar_type omega_opt = slae::optimal_sor_omega(side, a, b);
    if (!(omega_opt > 1.0 && omega_opt < 2.0)) {
        std::cout << "fail: optimal SOR omega is outside the expected range\n";
        return 1;
    }

    std::cout << "success\n";
    return 0;
}
