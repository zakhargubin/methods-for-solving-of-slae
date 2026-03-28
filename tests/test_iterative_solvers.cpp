#include "iterative_solvers/iterative_solvers.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

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

slae::CSRMatrix make_test_matrix() {
    const std::vector<slae::DokEntry> entries = {
        {0, 0, 4.0}, {0, 1, -1.0},
        {1, 0, -1.0}, {1, 1, 4.0}, {1, 2, -1.0},
        {2, 1, -1.0}, {2, 2, 4.0}, {2, 3, -1.0},
        {3, 2, -1.0}, {3, 3, 3.0}
    };
    return slae::CSRMatrix(4, 4, entries);
}

slae::CSRMatrix make_tridiagonal(slae::index_type size,
                                          slae::scalar_type diagonal,
                                          slae::scalar_type off_diagonal) {
    std::vector<slae::DokEntry> entries;
    entries.reserve(3 * size - 2);

    for (slae::index_type i = 0; i < size; ++i) {
        entries.push_back({i, i, diagonal});
        if (i > 0) {
            entries.push_back({i, i - 1, off_diagonal});
        }
        if (i + 1 < size) {
            entries.push_back({i, i + 1, off_diagonal});
        }
    }

    return slae::CSRMatrix(size, size, entries);
}

slae::scalar_type lambda_min(slae::index_type size,
                                      slae::scalar_type diagonal,
                                      slae::scalar_type off_diagonal_abs) {
    const slae::scalar_type pi = std::acos(-1.0);
    return diagonal - 2.0 * off_diagonal_abs * std::cos(pi / static_cast<slae::scalar_type>(size + 1));
}

slae::scalar_type lambda_max(slae::index_type size,
                                      slae::scalar_type diagonal,
                                      slae::scalar_type off_diagonal_abs) {
    const slae::scalar_type pi = std::acos(-1.0);
    return diagonal + 2.0 * off_diagonal_abs * std::cos(pi / static_cast<slae::scalar_type>(size + 1));
}

}

int main() {
    const slae::CSRMatrix A = make_test_matrix();
    const slae::vector_type expected = {1.0, 2.0, 3.0, 4.0};
    const slae::vector_type b = A * expected;
    const slae::vector_type x0(4, 0.0);

    const slae::IterativeSolverResult jacobi_result =
        slae::jacobi(A, b, x0, 1e-10, 500, true);
    if (!jacobi_result.converged || jacobi_result.diverged) {
        std::cout << "fail: Jacobi did not converge on a diagonally dominant system\n";
        return 1;
    }
    if (!vectors_close(jacobi_result.x, expected, 1e-7)) {
        std::cout << "fail: Jacobi returned a wrong solution\n";
        return 1;
    }
    if (jacobi_result.residual_norm_history.size() != jacobi_result.elapsed_microseconds_history.size()) {
        std::cout << "fail: Jacobi history sizes do not match\n";
        return 1;
    }

    const slae::IterativeSolverResult gauss_seidel_result =
        slae::gauss_seidel(A, b, x0, 1e-10, 500, true);
    if (!gauss_seidel_result.converged || gauss_seidel_result.diverged) {
        std::cout << "fail: Gauss-Seidel did not converge on an SPD system\n";
        return 1;
    }
    if (!vectors_close(gauss_seidel_result.x, expected, 1e-7)) {
        std::cout << "fail: Gauss-Seidel returned a wrong solution\n";
        return 1;
    }
    if (gauss_seidel_result.residual_norm_history.size() != gauss_seidel_result.elapsed_microseconds_history.size()) {
        std::cout << "fail: Gauss-Seidel history sizes do not match\n";
        return 1;
    }

    const slae::IterativeSolverResult simple_iteration_result =
        slae::simple_iteration(A, b, x0, 0.2, 1e-10, 1000, true);
    if (!simple_iteration_result.converged || simple_iteration_result.diverged) {
        std::cout << "fail: simple iteration did not converge for a valid tau\n";
        return 1;
    }
    if (!vectors_close(simple_iteration_result.x, expected, 1e-7)) {
        std::cout << "fail: simple iteration returned a wrong solution\n";
        return 1;
    }
    if (simple_iteration_result.residual_norm_history.size() != simple_iteration_result.elapsed_microseconds_history.size()) {
        std::cout << "fail: simple iteration history sizes do not match\n";
        return 1;
    }

    try {
        (void)slae::simple_iteration(A, b, x0, 0.0, 1e-10, 10, false);
        std::cout << "fail: simple iteration accepted zero tau\n";
        return 1;
    } catch (const std::runtime_error&) {
    }

    const slae::index_type cheb_size = 40;
    const slae::scalar_type diagonal = 2.2;
    const slae::scalar_type off_diagonal = -1.0;
    const slae::CSRMatrix A_cheb = make_tridiagonal(cheb_size, diagonal, off_diagonal);
    const slae::vector_type cheb_expected(cheb_size, 1.0);
    const slae::vector_type cheb_b = A_cheb * cheb_expected;
    const slae::vector_type cheb_x0(cheb_size, 0.0);
    const slae::scalar_type lda_min = lambda_min(cheb_size, diagonal, std::abs(off_diagonal));
    const slae::scalar_type lda_max = lambda_max(cheb_size, diagonal, std::abs(off_diagonal));
    const slae::scalar_type tau_opt = 2.0 / (lda_min + lda_max);

    const slae::IterativeSolverResult baseline_mpi =
        slae::simple_iteration(A_cheb, cheb_b, cheb_x0, tau_opt, 1e-8, 1000, true);
    const slae::IterativeSolverResult chebyshev_result =
        slae::chebyshev_simple_iteration(A_cheb, cheb_b, cheb_x0,
                                         lda_min, lda_max,
                                         16, 1e-8, 1000, true);

    if (!chebyshev_result.converged || chebyshev_result.diverged) {
        std::cout << "fail: Chebyshev-accelerated MPI did not converge on an SPD system\n";
        return 1;
    }
    if (!vectors_close(chebyshev_result.x, cheb_expected, 1e-6)) {
        std::cout << "fail: Chebyshev-accelerated MPI returned a wrong solution\n";
        return 1;
    }
    if (chebyshev_result.residual_norm_history.size() != chebyshev_result.elapsed_microseconds_history.size()) {
        std::cout << "fail: Chebyshev history sizes do not match\n";
        return 1;
    }
    if (!baseline_mpi.converged || baseline_mpi.diverged) {
        std::cout << "fail: baseline MPI did not converge on the Chebyshev test system\n";
        return 1;
    }
    if (chebyshev_result.iterations >= baseline_mpi.iterations) {
        std::cout << "fail: Chebyshev acceleration was not faster than plain MPI\n";
        return 1;
    }

    const slae::scalar_type lambda_max_estimate =
        slae::power_iteration_max_eigenvalue(A_cheb, 2000, 1e-12);
    if (std::abs(lambda_max_estimate - lda_max) > 1e-6 * lda_max) {
        std::cout << "fail: power iteration produced a poor estimate of lambda_max\n";
        return 1;
    }

    try {
        (void)slae::chebyshev_simple_iteration(A_cheb, cheb_b, cheb_x0,
                                               lda_min, lda_max,
                                               3, 1e-8, 100, false);
        std::cout << "fail: Chebyshev MPI accepted a non-power-of-two root count\n";
        return 1;
    } catch (const std::runtime_error&) {
    }

    std::cout << "success\n";
    return 0;
}
