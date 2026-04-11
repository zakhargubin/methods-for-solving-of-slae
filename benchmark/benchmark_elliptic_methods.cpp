#include "elliptic_matrix/elliptic_matrix.hpp"
#include "gradient_methods/gradient_methods.hpp"
#include "iterative_solvers/iterative_solvers.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

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

} 

int main(int argc, char** argv) {
    const std::string history_path = (argc > 1) ? argv[1] : "benchmark/elliptic_history.csv";
    const std::string summary_path = (argc > 2) ? argv[2] : "benchmark/elliptic_summary.csv";

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

    const slae::index_type side = 30;
    const slae::scalar_type a = -1.0;
    const slae::scalar_type b_param = 2.0;
    const slae::CSRMatrix A = slae::generate_elliptic_matrix(side, a, b_param);
    const slae::vector_type x_true(A.ncols(), 1.0);
    const slae::vector_type b = A * x_true;
    const slae::vector_type x0(A.ncols(), 0.0);
    const slae::scalar_type tolerance = 1e-8;
    const slae::index_type max_iterations = 5000;

    const slae::scalar_type omega = slae::optimal_sor_omega(side, a, b_param);

    const slae::IterativeSolverResult gs_result =
        slae::gauss_seidel(A, b, x0, tolerance, max_iterations, true);
    const slae::IterativeSolverResult sor_result =
        slae::sor(A, b, x0, omega, tolerance, max_iterations, true);
    const slae::IterativeSolverResult sd_result =
        slae::steepest_descent(A, b, x0, tolerance, max_iterations, true);
    const slae::IterativeSolverResult cg_result =
        slae::conjugate_gradient(A, b, x0, tolerance, max_iterations, true);

    const slae::vector_type zero_rhs(A.ncols(), 0.0);
    const auto sgs_method = [&A, &b](const slae::vector_type& x) {
        return slae::symmetric_gauss_seidel_step(A, b, x);
    };
    const auto sgs_transition = [&A, &zero_rhs](const slae::vector_type& x) {
        return slae::symmetric_gauss_seidel_step(A, zero_rhs, x);
    };
    const slae::scalar_type rho_sgs =
        slae::estimate_spectral_radius(sgs_transition, A.ncols(), 5000, 1e-12);
    const slae::IterativeSolverResult cheb_sgs_result =
        slae::chebyshev_accelerated_method(A, b, x0, rho_sgs,
                                           sgs_method,
                                           tolerance, max_iterations, true);

    history_out << "method,iteration,elapsed_us,residual_norm\n";
    write_history(history_out, "GaussSeidel", gs_result);
    write_history(history_out, "SOR", sor_result);
    write_history(history_out, "ChebyshevSymmetricGaussSeidel", cheb_sgs_result);
    write_history(history_out, "SteepestDescent", sd_result);
    write_history(history_out, "ConjugateGradient", cg_result);

    summary_out << "method,iterations,residual_norm,converged,diverged,total_elapsed_us\n";
    write_summary(summary_out, "GaussSeidel", gs_result);
    write_summary(summary_out, "SOR", sor_result);
    write_summary(summary_out, "ChebyshevSymmetricGaussSeidel", cheb_sgs_result);
    write_summary(summary_out, "SteepestDescent", sd_result);
    write_summary(summary_out, "ConjugateGradient", cg_result);

    return 0;
}
