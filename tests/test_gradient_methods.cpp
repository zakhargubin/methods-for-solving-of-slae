#include "elliptic_matrix/elliptic_matrix.hpp"
#include "gradient_methods/gradient_methods.hpp"
#include "iterative_solvers/iterative_solvers.hpp"

#include <cmath>
#include <iostream>

namespace {

bool close(slae::scalar_type lhs,
           slae::scalar_type rhs,
           slae::scalar_type eps = 1e-8) {
    return std::abs(lhs - rhs) < eps;
}

bool vectors_close(const slae::vector_type& lhs,
                   const slae::vector_type& rhs,
                   slae::scalar_type eps = 1e-8) {
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
    const slae::index_type side = 6;
    const slae::scalar_type a = -1.0;
    const slae::scalar_type b = 2.0;
    const slae::CSRMatrix A = slae::generate_elliptic_matrix(side, a, b);
    const slae::vector_type expected(A.ncols(), 1.0);
    const slae::vector_type rhs = A * expected;
    const slae::vector_type x0(A.ncols(), 0.0);

    const slae::IterativeSolverResult sd_result =
        slae::steepest_descent(A, rhs, x0, 1e-8, 5000, true);
    const slae::IterativeSolverResult cg_result =
        slae::conjugate_gradient(A, rhs, x0, 1e-8, 5000, true);

    if (!sd_result.converged || sd_result.diverged) {
        std::cout << "fail: steepest descent did not converge on the elliptic system\n";
        return 1;
    }
    if (!vectors_close(sd_result.x, expected, 1e-6)) {
        std::cout << "fail: steepest descent returned a wrong solution\n";
        return 1;
    }
    if (sd_result.residual_norm_history.size() != sd_result.elapsed_microseconds_history.size()) {
        std::cout << "fail: steepest descent history sizes do not match\n";
        return 1;
    }

    if (!cg_result.converged || cg_result.diverged) {
        std::cout << "fail: conjugate gradient did not converge on the elliptic system\n";
        return 1;
    }
    if (!vectors_close(cg_result.x, expected, 1e-6)) {
        std::cout << "fail: conjugate gradient returned a wrong solution\n";
        return 1;
    }
    if (cg_result.residual_norm_history.size() != cg_result.elapsed_microseconds_history.size()) {
        std::cout << "fail: conjugate gradient history sizes do not match\n";
        return 1;
    }
    if (cg_result.iterations >= sd_result.iterations) {
        std::cout << "fail: conjugate gradient was not faster than steepest descent\n";
        return 1;
    }

    std::cout << "success\n";
    return 0;
}
