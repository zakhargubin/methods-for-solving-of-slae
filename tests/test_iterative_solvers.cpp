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

slae::CSRMatrix make_toeplitz_tridiagonal(slae::index_type size,
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

slae::scalar_type toeplitz_lambda_min(slae::index_type size,
                                      slae::scalar_type diagonal,
                                      slae::scalar_type off_diagonal_abs) {
    const slae::scalar_type pi = std::acos(-1.0);
    return diagonal - 2.0 * off_diagonal_abs * std::cos(pi / static_cast<slae::scalar_type>(size + 1));
}

slae::scalar_type toeplitz_lambda_max(slae::index_type size,
                                      slae::scalar_type diagonal,
                                      slae::scalar_type off_diagonal_abs) {
    const slae::scalar_type pi = std::acos(-1.0);
    return diagonal + 2.0 * off_diagonal_abs * std::cos(pi / static_cast<slae::scalar_type>(size + 1));
}


} // namespace

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

    const slae::IterativeSolverResult symmetric_gauss_seidel_result =
        slae::symmetric_gauss_seidel(A, b, x0, 1e-10, 500, true);
    if (!symmetric_gauss_seidel_result.converged || symmetric_gauss_seidel_result.diverged) {
        std::cout << "fail: symmetric Gauss-Seidel did not converge on an SPD system\n";
        return 1;
    }
    if (!vectors_close(symmetric_gauss_seidel_result.x, expected, 1e-7)) {
        std::cout << "fail: symmetric Gauss-Seidel returned a wrong solution\n";
        return 1;
    }
    if (symmetric_gauss_seidel_result.residual_norm_history.size() != symmetric_gauss_seidel_result.elapsed_microseconds_history.size()) {
        std::cout << "fail: symmetric Gauss-Seidel history sizes do not match\n";
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

    // DZ5 checks
    const slae::index_type dz5_size = 40;
    const slae::scalar_type dz5_diagonal = 2.2;
    const slae::scalar_type dz5_off_diagonal = -1.0;
    const slae::CSRMatrix A_dz5 = make_toeplitz_tridiagonal(dz5_size, dz5_diagonal, dz5_off_diagonal);
    const slae::vector_type dz5_expected(dz5_size, 1.0);
    const slae::vector_type dz5_b = A_dz5 * dz5_expected;
    const slae::vector_type dz5_x0(dz5_size, 0.0);
    const slae::scalar_type lambda_min = toeplitz_lambda_min(dz5_size, dz5_diagonal, std::abs(dz5_off_diagonal));
    const slae::scalar_type lambda_max = toeplitz_lambda_max(dz5_size, dz5_diagonal, std::abs(dz5_off_diagonal));
    const slae::scalar_type tau_opt = 2.0 / (lambda_min + lambda_max);

    const slae::IterativeSolverResult baseline_mpi =
        slae::simple_iteration(A_dz5, dz5_b, dz5_x0, tau_opt, 1e-8, 1000, true);
    const slae::IterativeSolverResult chebyshev_mpi =
        slae::chebyshev_simple_iteration(A_dz5, dz5_b, dz5_x0,
                                         lambda_min, lambda_max,
                                         16, 1e-8, 1000, true);

    if (!baseline_mpi.converged || baseline_mpi.diverged) {
        std::cout << "fail: baseline MPI did not converge on the DZ5 system\n";
        return 1;
    }
    if (!chebyshev_mpi.converged || chebyshev_mpi.diverged) {
        std::cout << "fail: Chebyshev-accelerated MPI did not converge on the DZ5 system\n";
        return 1;
    }
    if (!vectors_close(chebyshev_mpi.x, dz5_expected, 1e-6)) {
        std::cout << "fail: Chebyshev-accelerated MPI returned a wrong solution\n";
        return 1;
    }
    if (chebyshev_mpi.residual_norm_history.size() != chebyshev_mpi.elapsed_microseconds_history.size()) {
        std::cout << "fail: Chebyshev MPI history sizes do not match\n";
        return 1;
    }
    if (chebyshev_mpi.iterations >= baseline_mpi.iterations) {
        std::cout << "fail: Chebyshev acceleration was not faster than plain MPI\n";
        return 1;
    }

    const slae::scalar_type lambda_max_estimate =
        slae::power_iteration_max_eigenvalue(A_dz5, 2000, 1e-12);
    if (std::abs(lambda_max_estimate - lambda_max) > 1e-6 * lambda_max) {
        std::cout << "fail: power iteration produced a poor estimate of lambda_max\n";
        return 1;
    }

    try {
        (void)slae::chebyshev_simple_iteration(A_dz5, dz5_b, dz5_x0,
                                               lambda_min, lambda_max,
                                               3, 1e-8, 100, false);
        std::cout << "fail: Chebyshev MPI accepted a non-power-of-two root count\n";
        return 1;
    } catch (const std::runtime_error&) {
    }

    // DZ6 checks
    const slae::index_type dz6_size = 40;
    const slae::scalar_type diagonal = 4.0;
    const slae::scalar_type off_diagonal = -1.0;
    const slae::CSRMatrix A_cheb = make_toeplitz_tridiagonal(dz6_size, diagonal, off_diagonal);
    const slae::vector_type cheb_expected(dz6_size, 1.0);
    const slae::vector_type cheb_b = A_cheb * cheb_expected;
    const slae::vector_type cheb_x0(dz6_size, 0.0);
    const slae::vector_type zero_rhs(dz6_size, 0.0);

    const auto jacobi_step = [&A_cheb, &cheb_b](const slae::vector_type& x) {
        return slae::jacobi_step(A_cheb, cheb_b, x);
    };
    const auto jacobi_transition = [&A_cheb, &zero_rhs](const slae::vector_type& x) {
        return slae::jacobi_step(A_cheb, zero_rhs, x);
    };

    const slae::scalar_type estimated_jacobi_rho =
        slae::estimate_spectral_radius(jacobi_transition, dz6_size, 3000, 1e-12);
    const auto exact_jacobi_rho = [dz6_size, diagonal, off_diagonal]() {
        const slae::scalar_type pi = std::acos(-1.0);
        return 2.0 * std::abs(off_diagonal)
            * std::cos(pi / static_cast<slae::scalar_type>(dz6_size + 1))
            / diagonal;
    }();
    if (std::abs(estimated_jacobi_rho - exact_jacobi_rho) > 1e-6) {
        std::cout << "fail: spectral radius estimate for Jacobi is inaccurate\n";
        return 1;
    }

    const slae::IterativeSolverResult plain_jacobi =
        slae::jacobi(A_cheb, cheb_b, cheb_x0, 1e-8, 1000, true);
    const slae::IterativeSolverResult accelerated_jacobi =
        slae::chebyshev_accelerated_method(A_cheb, cheb_b, cheb_x0,
                                           estimated_jacobi_rho,
                                           jacobi_step,
                                           1e-8, 1000, true);

    if (!plain_jacobi.converged || plain_jacobi.diverged) {
        std::cout << "fail: plain Jacobi did not converge on the Chebyshev test system\n";
        return 1;
    }
    if (!accelerated_jacobi.converged || accelerated_jacobi.diverged) {
        std::cout << "fail: accelerated Jacobi did not converge on the Chebyshev test system\n";
        return 1;
    }
    if (!vectors_close(accelerated_jacobi.x, cheb_expected, 1e-6)) {
        std::cout << "fail: accelerated Jacobi returned a wrong solution\n";
        return 1;
    }
    if (accelerated_jacobi.iterations >= plain_jacobi.iterations) {
        std::cout << "fail: Chebyshev acceleration did not improve Jacobi iteration count\n";
        return 1;
    }
    if (accelerated_jacobi.residual_norm_history.size() != accelerated_jacobi.elapsed_microseconds_history.size()) {
        std::cout << "fail: accelerated Jacobi history sizes do not match\n";
        return 1;
    }

    const auto sgs_step = [&A_cheb, &cheb_b](const slae::vector_type& x) {
        return slae::symmetric_gauss_seidel_step(A_cheb, cheb_b, x);
    };
    const auto sgs_transition = [&A_cheb, &zero_rhs](const slae::vector_type& x) {
        return slae::symmetric_gauss_seidel_step(A_cheb, zero_rhs, x);
    };

    const slae::scalar_type estimated_sgs_rho =
        slae::estimate_spectral_radius(sgs_transition, dz6_size, 3000, 1e-12);
    if (!(estimated_sgs_rho > 0.0 && estimated_sgs_rho < 1.0)) {
        std::cout << "fail: symmetric Gauss-Seidel spectral radius estimate is invalid\n";
        return 1;
    }

    const slae::IterativeSolverResult plain_sgs =
        slae::symmetric_gauss_seidel(A_cheb, cheb_b, cheb_x0, 1e-8, 1000, true);
    const slae::IterativeSolverResult accelerated_sgs =
        slae::chebyshev_accelerated_method(A_cheb, cheb_b, cheb_x0,
                                           estimated_sgs_rho,
                                           sgs_step,
                                           1e-8, 1000, true);

    if (!plain_sgs.converged || plain_sgs.diverged) {
        std::cout << "fail: plain symmetric Gauss-Seidel did not converge\n";
        return 1;
    }
    if (!accelerated_sgs.converged || accelerated_sgs.diverged) {
        std::cout << "fail: accelerated symmetric Gauss-Seidel did not converge\n";
        return 1;
    }
    if (!vectors_close(accelerated_sgs.x, cheb_expected, 1e-6)) {
        std::cout << "fail: accelerated symmetric Gauss-Seidel returned a wrong solution\n";
        return 1;
    }
    if (accelerated_sgs.iterations >= plain_sgs.iterations) {
        std::cout << "fail: Chebyshev acceleration did not improve symmetric Gauss-Seidel iteration count\n";
        return 1;
    }

    try {
        (void)slae::chebyshev_accelerated_method(A_cheb, cheb_b, cheb_x0,
                                                 1.0,
                                                 jacobi_step,
                                                 1e-8, 10, false);
        std::cout << "fail: Chebyshev acceleration accepted rho = 1\n";
        return 1;
    } catch (const std::runtime_error&) {
    }


    const slae::IterativeSolverResult sor_result =
        slae::sor(A, b, x0, 1.1, 1e-10, 500, true);
    if (!sor_result.converged || sor_result.diverged) {
        std::cout << "fail: SOR did not converge on an SPD system\n";
        return 1;
    }
    if (!vectors_close(sor_result.x, expected, 1e-7)) {
        std::cout << "fail: SOR returned a wrong solution\n";
        return 1;
    }
    if (sor_result.residual_norm_history.size() != sor_result.elapsed_microseconds_history.size()) {
        std::cout << "fail: SOR history sizes do not match\n";
        return 1;
    }

    try {
        (void)slae::sor(A, b, x0, 2.0, 1e-10, 10, false);
        std::cout << "fail: SOR accepted omega = 2\n";
        return 1;
    } catch (const std::runtime_error&) {
    }

    std::cout << "success\n";
    return 0;
}
