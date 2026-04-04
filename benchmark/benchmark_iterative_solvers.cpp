#include "csr_matrix/csr_matrix.hpp"
#include "iterative_solvers/iterative_solvers.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

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

void write_history(std::ofstream& out,
                   const std::string& method,
                   const slae::IterativeSolverResult& result) {
    for (slae::index_type i = 0; i < result.residual_norm_history.size(); ++i) {
        out << method << ','
            << i << ','
            << std::setprecision(10) << result.elapsed_microseconds_history[i] << ','
            << result.residual_norm_history[i] << '\n';
    }
}

void write_summary(std::ofstream& out,
                   const std::string& method,
                   const slae::IterativeSolverResult& result) {
    const slae::scalar_type total_elapsed =
        result.elapsed_microseconds_history.empty() ? 0.0 : result.elapsed_microseconds_history.back();

    out << method << ','
        << result.iterations << ','
        << std::setprecision(10) << result.residual_norm << ','
        << (result.converged ? 1 : 0) << ','
        << (result.diverged ? 1 : 0) << ','
        << total_elapsed << '\n';
}

} // namespace

int main(int argc, char** argv) {
    const std::string history_path = (argc > 1) ? argv[1] : "benchmark/iterative_history.csv";
    const std::string summary_path = (argc > 2) ? argv[2] : "benchmark/iterative_summary.csv";

    std::ofstream history_out(history_path);
    std::ofstream summary_out(summary_path);
    if (!history_out) {
        std::cerr << "Cannot open output file: " << history_path << '\n';
        return 1;
    }
    if (!summary_out) {
        std::cerr << "Cannot open output file: " << summary_path << '\n';
        return 1;
    }

    const slae::index_type size = 300;
    const slae::scalar_type diagonal = 4.0;
    const slae::scalar_type off_diagonal = -1.0;
    const slae::CSRMatrix A = make_toeplitz_tridiagonal(size, diagonal, off_diagonal);
    const slae::vector_type x_true(size, 1.0);
    const slae::vector_type b = A * x_true;
    const slae::vector_type x0(size, 0.0);
    const slae::scalar_type tolerance = 1e-8;
    const slae::index_type max_iterations = 1500;

    const slae::scalar_type lambda_min = toeplitz_lambda_min(size, diagonal, std::abs(off_diagonal));
    const slae::scalar_type lambda_max = toeplitz_lambda_max(size, diagonal, std::abs(off_diagonal));
    const slae::scalar_type tau_opt = 2.0 / (lambda_min + lambda_max);
    const slae::index_type chebyshev_roots_count = 1u << 5;

    const slae::IterativeSolverResult mpi_result =
        slae::simple_iteration(A, b, x0, tau_opt, tolerance, max_iterations, true);
    const slae::IterativeSolverResult cheb_mpi_result =
        slae::chebyshev_simple_iteration(A, b, x0,
                                         lambda_min, lambda_max,
                                         chebyshev_roots_count,
                                         tolerance, max_iterations, true);
    const slae::IterativeSolverResult jacobi_result =
        slae::jacobi(A, b, x0, tolerance, max_iterations, true);
    const slae::IterativeSolverResult gs_result =
        slae::gauss_seidel(A, b, x0, tolerance, max_iterations, true);

    const slae::vector_type zero_rhs(size, 0.0);
    const auto jacobi_method = [&A, &b](const slae::vector_type& x) {
        return slae::jacobi_step(A, b, x);
    };
    const auto jacobi_transition = [&A, &zero_rhs](const slae::vector_type& x) {
        return slae::jacobi_step(A, zero_rhs, x);
    };
    const auto sgs_method = [&A, &b](const slae::vector_type& x) {
        return slae::symmetric_gauss_seidel_step(A, b, x);
    };
    const auto sgs_transition = [&A, &zero_rhs](const slae::vector_type& x) {
        return slae::symmetric_gauss_seidel_step(A, zero_rhs, x);
    };

    const slae::scalar_type rho_jacobi =
        slae::estimate_spectral_radius(jacobi_transition, size, 3000, 1e-12);
    const slae::scalar_type rho_sgs =
        slae::estimate_spectral_radius(sgs_transition, size, 3000, 1e-12);

    const slae::IterativeSolverResult sgs_result =
        slae::symmetric_gauss_seidel(A, b, x0, tolerance, max_iterations, true);
    const slae::IterativeSolverResult cheb_jacobi_result =
        slae::chebyshev_accelerated_method(A, b, x0, rho_jacobi,
                                           jacobi_method,
                                           tolerance, max_iterations, true);
    const slae::IterativeSolverResult cheb_sgs_result =
        slae::chebyshev_accelerated_method(A, b, x0, rho_sgs,
                                           sgs_method,
                                           tolerance, max_iterations, true);

    history_out << "method,iteration,elapsed_us,residual_norm\n";
    write_history(history_out, "MPI", mpi_result);
    write_history(history_out, "ChebyshevMPI", cheb_mpi_result);
    write_history(history_out, "Jacobi", jacobi_result);
    write_history(history_out, "GaussSeidel", gs_result);
    write_history(history_out, "SymmetricGaussSeidel", sgs_result);
    write_history(history_out, "ChebyshevJacobi", cheb_jacobi_result);
    write_history(history_out, "ChebyshevSymmetricGaussSeidel", cheb_sgs_result);

    summary_out << "method,iterations,residual_norm,converged,diverged,total_elapsed_us\n";
    write_summary(summary_out, "MPI", mpi_result);
    write_summary(summary_out, "ChebyshevMPI", cheb_mpi_result);
    write_summary(summary_out, "Jacobi", jacobi_result);
    write_summary(summary_out, "GaussSeidel", gs_result);
    write_summary(summary_out, "SymmetricGaussSeidel", sgs_result);
    write_summary(summary_out, "ChebyshevJacobi", cheb_jacobi_result);
    write_summary(summary_out, "ChebyshevSymmetricGaussSeidel", cheb_sgs_result);

    return 0;
}
