#include "iterative_solvers/iterative_solvers.hpp"
#include "vector_operations/vector_operations.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

bool nearly_zero(slae::scalar_type x) {
    return std::abs(x) <= slae::tolerance;
}

bool is_finite_vector(const slae::vector_type& x) {
    for (const slae::scalar_type value : x) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

bool is_power_of_two(slae::index_type value) {
    return value != 0 && ((value & (value - 1)) == 0);
}

std::vector<slae::index_type> chebyshev_permutation(slae::index_type roots_count) {
    if (!is_power_of_two(roots_count)) {
        throw std::runtime_error("Chebyshev acceleration requires the number of roots to be a power of two");
    }

    std::vector<slae::index_type> permutation = {0};
    while (permutation.size() < roots_count) {
        const slae::index_type next_size = permutation.size() * 2;
        std::vector<slae::index_type> next;
        next.reserve(next_size);
        for (const slae::index_type index : permutation) {
            next.push_back(index);
            next.push_back(next_size - 1 - index);
        }
        permutation = std::move(next);
    }

    return permutation;
}

std::vector<slae::scalar_type> chebyshev_tau_sequence(slae::scalar_type lambda_min,
                                                      slae::scalar_type lambda_max,
                                                      slae::index_type roots_count) {
    if (!std::isfinite(lambda_min) || !std::isfinite(lambda_max)) {
        throw std::runtime_error("Chebyshev simple iteration requires finite spectrum bounds");
    }
    if (lambda_min <= 0.0 || lambda_max <= 0.0) {
        throw std::runtime_error("Chebyshev simple iteration requires positive spectrum bounds");
    }
    if (lambda_min > lambda_max) {
        throw std::runtime_error("Chebyshev simple iteration requires lambda_min <= lambda_max");
    }
    if (!is_power_of_two(roots_count)) {
        throw std::runtime_error("Chebyshev simple iteration requires roots_count to be a power of two");
    }

    if (roots_count == 1) {
        return {2.0 / (lambda_min + lambda_max)};
    }

    const slae::scalar_type pi = std::acos(-1.0);
    const slae::scalar_type theta0 = pi / (2.0 * static_cast<slae::scalar_type>(roots_count));
    const slae::scalar_type delta = pi / static_cast<slae::scalar_type>(roots_count);
    const slae::scalar_type cos_delta = std::cos(delta);
    const slae::scalar_type sin_delta = std::sin(delta);

    slae::scalar_type cos_theta = std::cos(theta0);
    slae::scalar_type sin_theta = std::sin(theta0);

    const slae::scalar_type center = 0.5 * (lambda_min + lambda_max);
    const slae::scalar_type radius = 0.5 * (lambda_max - lambda_min);

    std::vector<slae::scalar_type> ordered_taus(roots_count, 0.0);
    const std::vector<slae::index_type> permutation = chebyshev_permutation(roots_count);

    for (slae::index_type natural_index = 0; natural_index < roots_count; ++natural_index) {
        const slae::scalar_type mapped_root = center + radius * cos_theta;
        ordered_taus[permutation[natural_index]] = 1.0 / mapped_root;

        const slae::scalar_type next_cos = cos_theta * cos_delta - sin_theta * sin_delta;
        const slae::scalar_type next_sin = sin_theta * cos_delta + cos_theta * sin_delta;
        cos_theta = next_cos;
        sin_theta = next_sin;
    }

    return ordered_taus;
}

void check_system_sizes(const slae::CSRMatrix& A,
                        const slae::vector_type& b,
                        const slae::vector_type& x0) {
    if (A.nrows() != A.ncols()) {
        throw std::runtime_error("Iterative solvers require a square matrix");
    }
    if (b.size() != A.nrows()) {
        throw std::runtime_error("RHS size must match matrix size in iterative solver");
    }
    if (x0.size() != A.ncols()) {
        throw std::runtime_error("Initial guess size must match matrix size in iterative solver");
    }
}

void ensure_finite_tolerance(slae::scalar_type tolerance) {
    if (!std::isfinite(tolerance) || tolerance < 0.0) {
        throw std::runtime_error("Iterative solver requires a finite non-negative tolerance");
    }
}

void ensure_valid_relaxation(slae::scalar_type omega) {
    if (!std::isfinite(omega) || omega <= 0.0 || omega >= 2.0) {
        throw std::runtime_error("SOR requires a finite relaxation parameter omega in (0, 2)");
    }
}

slae::IterativeSolverResult make_initial_result(const slae::CSRMatrix& A,
                                                const slae::vector_type& b,
                                                const slae::vector_type& x0,
                                                slae::scalar_type tolerance,
                                                bool store_history) {
    slae::IterativeSolverResult result{};
    result.x = x0;
    result.iterations = 0;
    result.residual_norm = slae::residual_norm(A, x0, b);
    result.converged = std::isfinite(result.residual_norm) && (result.residual_norm <= tolerance);
    result.diverged = !std::isfinite(result.residual_norm) || !is_finite_vector(x0);

    if (store_history) {
        result.residual_norm_history.push_back(result.residual_norm);
        result.elapsed_microseconds_history.push_back(0.0);
    }

    return result;
}

void append_history(slae::IterativeSolverResult& result,
                    slae::scalar_type residual,
                    const clock_type::time_point& start_time) {
    const auto now = clock_type::now();
    const std::chrono::duration<slae::scalar_type, std::micro> elapsed = now - start_time;
    result.residual_norm_history.push_back(residual);
    result.elapsed_microseconds_history.push_back(elapsed.count());
}

slae::vector_type jacobi_step_impl(const slae::CSRMatrix& A,
                                   const slae::vector_type& b,
                                   const slae::vector_type& x) {
    check_system_sizes(A, b, x);

    slae::vector_type next(A.ncols(), 0.0);
    for (slae::index_type i = 0; i < A.nrows(); ++i) {
        slae::scalar_type diagonal = 0.0;
        slae::scalar_type sum = 0.0;

        for (slae::index_type k = A.rows()[i]; k < A.rows()[i + 1]; ++k) {
            const slae::index_type j = A.cols()[k];
            const slae::scalar_type value = A.values()[k];
            if (j == i) {
                diagonal = value;
            } else {
                sum += value * x[j];
            }
        }

        if (nearly_zero(diagonal)) {
            throw std::runtime_error("Jacobi method requires non-zero diagonal elements");
        }

        next[i] = (b[i] - sum) / diagonal;
    }

    return next;
}

slae::vector_type gauss_seidel_step_impl(const slae::CSRMatrix& A,
                                         const slae::vector_type& b,
                                         const slae::vector_type& x) {
    check_system_sizes(A, b, x);

    slae::vector_type next = x;
    for (slae::index_type i = 0; i < A.nrows(); ++i) {
        slae::scalar_type diagonal = 0.0;
        slae::scalar_type sum = b[i];

        for (slae::index_type k = A.rows()[i]; k < A.rows()[i + 1]; ++k) {
            const slae::index_type j = A.cols()[k];
            const slae::scalar_type value = A.values()[k];
            if (j == i) {
                diagonal = value;
            } else if (j < i) {
                sum -= value * next[j];
            } else {
                sum -= value * x[j];
            }
        }

        if (nearly_zero(diagonal)) {
            throw std::runtime_error("Gauss-Seidel method requires non-zero diagonal elements");
        }

        next[i] = sum / diagonal;
    }

    return next;
}

slae::vector_type sor_step_impl(const slae::CSRMatrix& A,
                                const slae::vector_type& b,
                                const slae::vector_type& x,
                                slae::scalar_type omega) {
    check_system_sizes(A, b, x);
    ensure_valid_relaxation(omega);

    slae::vector_type next = x;
    for (slae::index_type i = 0; i < A.nrows(); ++i) {
        slae::scalar_type diagonal = 0.0;
        slae::scalar_type sum = b[i];

        for (slae::index_type k = A.rows()[i]; k < A.rows()[i + 1]; ++k) {
            const slae::index_type j = A.cols()[k];
            const slae::scalar_type value = A.values()[k];
            if (j == i) {
                diagonal = value;
            } else if (j < i) {
                sum -= value * next[j];
            } else {
                sum -= value * x[j];
            }
        }

        if (nearly_zero(diagonal)) {
            throw std::runtime_error("SOR requires non-zero diagonal elements");
        }

        const slae::scalar_type gauss_seidel_value = sum / diagonal;
        next[i] = (1.0 - omega) * x[i] + omega * gauss_seidel_value;
    }

    return next;
}

slae::IterativeSolverResult run_stationary_solver(const slae::CSRMatrix& A,
                                                  const slae::vector_type& b,
                                                  const slae::vector_type& x0,
                                                  const slae::IterationStep& step,
                                                  slae::scalar_type tolerance,
                                                  slae::index_type max_iterations,
                                                  bool store_history) {
    check_system_sizes(A, b, x0);
    ensure_finite_tolerance(tolerance);

    slae::IterativeSolverResult result = make_initial_result(A, b, x0, tolerance, store_history);
    if (result.converged || result.diverged) {
        return result;
    }

    const clock_type::time_point start_time = clock_type::now();
    slae::vector_type current = x0;

    for (slae::index_type iter = 1; iter <= max_iterations; ++iter) {
        slae::vector_type next = step(current);
        const slae::scalar_type current_residual_norm = slae::residual_norm(A, next, b);

        result.x = next;
        result.iterations = iter;
        result.residual_norm = current_residual_norm;
        result.diverged = !std::isfinite(current_residual_norm) || !is_finite_vector(next);

        if (store_history) {
            append_history(result, current_residual_norm, start_time);
        }

        if (result.diverged) {
            return result;
        }

        if (current_residual_norm <= tolerance) {
            result.converged = true;
            return result;
        }

        current = std::move(next);
    }

    result.x = current;
    return result;
}

}

namespace slae {

scalar_type euclidean_norm(const vector_type& x) {
    scalar_type sum = 0.0;
    for (const scalar_type value : x) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

vector_type residual_vector(const CSRMatrix& A,
                            const vector_type& x,
                            const vector_type& b) {
    if (A.nrows() != b.size()) {
        throw std::runtime_error("RHS size must match matrix size when computing residual");
    }

    const vector_type Ax = A * x;
    vector_type residual(A.nrows(), 0.0);
    for (index_type i = 0; i < A.nrows(); ++i) {
        residual[i] = Ax[i] - b[i];
    }
    return residual;
}

scalar_type residual_norm(const CSRMatrix& A,
                          const vector_type& x,
                          const vector_type& b) {
    return euclidean_norm(residual_vector(A, x, b));
}

scalar_type power_iteration_max_eigenvalue(const CSRMatrix& A,
                                           index_type max_iterations,
                                           scalar_type tolerance) {
    if (A.nrows() != A.ncols()) {
        throw std::runtime_error("Power iteration requires a square matrix");
    }
    if (A.nrows() == 0) {
        throw std::runtime_error("Power iteration requires a non-empty matrix");
    }
    if (max_iterations == 0) {
        throw std::runtime_error("Power iteration requires a positive iteration limit");
    }
    if (!std::isfinite(tolerance) || tolerance < 0.0) {
        throw std::runtime_error("Power iteration requires a finite non-negative tolerance");
    }

    vector_type x(A.ncols(), 0.0);
    for (index_type i = 0; i < A.ncols(); ++i) {
        x[i] = static_cast<scalar_type>(i + 1);
    }
    const scalar_type x_norm = euclidean_norm(x);
    for (scalar_type& value : x) {
        value /= x_norm;
    }

    scalar_type mu = 0.0;

    for (index_type iter = 0; iter < max_iterations; ++iter) {
        const vector_type Ax = A * x;
        const scalar_type Ax_norm = euclidean_norm(Ax);
        if (nearly_zero(Ax_norm) || !std::isfinite(Ax_norm)) {
            throw std::runtime_error("Power iteration failed: matrix-vector product became zero or non-finite");
        }

        vector_type next_x = Ax;
        for (scalar_type& value : next_x) {
            value /= Ax_norm;
        }

        const vector_type Anext_x = A * next_x;
        mu = dot(next_x, Anext_x);
        if (!std::isfinite(mu)) {
            throw std::runtime_error("Power iteration failed: eigenvalue estimate became non-finite");
        }

        vector_type eigen_residual(next_x.size(), 0.0);
        for (index_type i = 0; i < next_x.size(); ++i) {
            eigen_residual[i] = Anext_x[i] - mu * next_x[i];
        }

        const scalar_type residual = euclidean_norm(eigen_residual);
        const scalar_type scale = std::max<scalar_type>(1.0, std::abs(mu));
        if (residual <= tolerance * scale) {
            return mu;
        }

        x = std::move(next_x);
    }

    return mu;
}

IterativeSolverResult chebyshev_simple_iteration(const CSRMatrix& A,
                                                 const vector_type& b,
                                                 const vector_type& x0,
                                                 scalar_type lambda_min,
                                                 scalar_type lambda_max,
                                                 index_type roots_count,
                                                 scalar_type tolerance,
                                                 index_type max_iterations,
                                                 bool store_history) {
    check_system_sizes(A, b, x0);
    ensure_finite_tolerance(tolerance);

    const std::vector<scalar_type> tau_sequence =
        chebyshev_tau_sequence(lambda_min, lambda_max, roots_count);

    IterativeSolverResult result = make_initial_result(A, b, x0, tolerance, store_history);
    if (result.converged || result.diverged) {
        return result;
    }

    const clock_type::time_point start_time = clock_type::now();

    vector_type x_current = x0;
    vector_type x_next(A.ncols(), 0.0);

    for (index_type iter = 1; iter <= max_iterations; ++iter) {
        const scalar_type tau = tau_sequence[(iter - 1) % tau_sequence.size()];
        const vector_type residual = residual_vector(A, x_current, b);

        for (index_type i = 0; i < A.ncols(); ++i) {
            x_next[i] = x_current[i] - tau * residual[i];
        }

        const scalar_type current_residual_norm = residual_norm(A, x_next, b);
        result.x = x_next;
        result.iterations = iter;
        result.residual_norm = current_residual_norm;
        result.diverged = !std::isfinite(current_residual_norm) || !is_finite_vector(x_next);

        if (store_history) {
            append_history(result, current_residual_norm, start_time);
        }

        if (result.diverged) {
            return result;
        }

        if (current_residual_norm <= tolerance) {
            result.converged = true;
            return result;
        }

        x_current = x_next;
    }

    result.x = x_current;
    return result;
}

IterativeSolverResult jacobi(const CSRMatrix& A,
                             const vector_type& b,
                             const vector_type& x0,
                             scalar_type tolerance,
                             index_type max_iterations,
                             bool store_history) {
    const IterationStep step = [&A, &b](const vector_type& x) {
        return jacobi_step_impl(A, b, x);
    };
    return run_stationary_solver(A, b, x0, step, tolerance, max_iterations, store_history);
}

IterativeSolverResult gauss_seidel(const CSRMatrix& A,
                                   const vector_type& b,
                                   const vector_type& x0,
                                   scalar_type tolerance,
                                   index_type max_iterations,
                                   bool store_history) {
    const IterationStep step = [&A, &b](const vector_type& x) {
        return gauss_seidel_step_impl(A, b, x);
    };
    return run_stationary_solver(A, b, x0, step, tolerance, max_iterations, store_history);
}

IterativeSolverResult simple_iteration(const CSRMatrix& A,
                                       const vector_type& b,
                                       const vector_type& x0,
                                       scalar_type tau,
                                       scalar_type tolerance,
                                       index_type max_iterations,
                                       bool store_history) {
    check_system_sizes(A, b, x0);
    ensure_finite_tolerance(tolerance);

    if (nearly_zero(tau) || !std::isfinite(tau)) {
        throw std::runtime_error("Simple iteration requires a finite non-zero tau");
    }

    IterativeSolverResult result = make_initial_result(A, b, x0, tolerance, store_history);
    if (result.converged || result.diverged) {
        return result;
    }

    const clock_type::time_point start_time = clock_type::now();

    vector_type x_old = x0;
    vector_type x_new(A.ncols(), 0.0);

    for (index_type iter = 1; iter <= max_iterations; ++iter) {
        const vector_type residual = residual_vector(A, x_old, b);

        for (index_type i = 0; i < A.ncols(); ++i) {
            x_new[i] = x_old[i] - tau * residual[i];
        }

        const scalar_type current_residual_norm = residual_norm(A, x_new, b);
        result.x = x_new;
        result.iterations = iter;
        result.residual_norm = current_residual_norm;
        result.diverged = !std::isfinite(current_residual_norm) || !is_finite_vector(x_new);

        if (store_history) {
            append_history(result, current_residual_norm, start_time);
        }

        if (result.diverged) {
            return result;
        }

        if (current_residual_norm <= tolerance) {
            result.converged = true;
            return result;
        }

        x_old = x_new;
    }

    result.x = x_old;
    return result;
}

vector_type jacobi_step(const CSRMatrix& A,
                        const vector_type& b,
                        const vector_type& x) {
    return jacobi_step_impl(A, b, x);
}

vector_type gauss_seidel_step(const CSRMatrix& A,
                              const vector_type& b,
                              const vector_type& x) {
    return gauss_seidel_step_impl(A, b, x);
}

vector_type symmetric_gauss_seidel_step(const CSRMatrix& A,
                                        const vector_type& b,
                                        const vector_type& x) {
    const vector_type half_step = gauss_seidel_step_impl(A, b, x);

    vector_type next = half_step;
    for (index_type row_from_bottom = 0; row_from_bottom < A.nrows(); ++row_from_bottom) {
        const index_type i = A.nrows() - 1 - row_from_bottom;
        scalar_type diagonal = 0.0;
        scalar_type sum = b[i];

        for (index_type k = A.rows()[i]; k < A.rows()[i + 1]; ++k) {
            const index_type j = A.cols()[k];
            const scalar_type value = A.values()[k];
            if (j == i) {
                diagonal = value;
            } else if (j < i) {
                sum -= value * half_step[j];
            } else {
                sum -= value * next[j];
            }
        }

        if (nearly_zero(diagonal)) {
            throw std::runtime_error("Symmetric Gauss-Seidel method requires non-zero diagonal elements");
        }

        next[i] = sum / diagonal;
    }

    return next;
}

scalar_type estimate_spectral_radius(const IterationStep& homogeneous_step,
                                     index_type size,
                                     index_type max_iterations,
                                     scalar_type tolerance) {
    if (size == 0) {
        throw std::runtime_error("Spectral radius estimation requires a positive operator size");
    }
    if (max_iterations == 0) {
        throw std::runtime_error("Spectral radius estimation requires a positive iteration limit");
    }
    if (!std::isfinite(tolerance) || tolerance < 0.0) {
        throw std::runtime_error("Spectral radius estimation requires a finite non-negative tolerance");
    }

    vector_type x(size, 0.0);
    for (index_type i = 0; i < size; ++i) {
        x[i] = static_cast<scalar_type>(i + 1);
    }

    scalar_type x_norm = euclidean_norm(x);
    if (nearly_zero(x_norm)) {
        throw std::runtime_error("Spectral radius estimation failed: zero initial vector norm");
    }
    for (scalar_type& value : x) {
        value /= x_norm;
    }

    scalar_type previous_rho = std::numeric_limits<scalar_type>::quiet_NaN();

    for (index_type iter = 0; iter < max_iterations; ++iter) {
        vector_type next_x = homogeneous_step(x);
        const scalar_type next_norm = euclidean_norm(next_x);

        if (!std::isfinite(next_norm)) {
            throw std::runtime_error("Spectral radius estimation failed: non-finite iterate norm");
        }
        if (nearly_zero(next_norm)) {
            return 0.0;
        }

        for (scalar_type& value : next_x) {
            value /= next_norm;
        }

        if (std::isfinite(previous_rho) && std::abs(next_norm - previous_rho) <= tolerance) {
            return next_norm;
        }

        previous_rho = next_norm;
        x = std::move(next_x);
    }

    return previous_rho;
}

IterativeSolverResult symmetric_gauss_seidel(const CSRMatrix& A,
                                             const vector_type& b,
                                             const vector_type& x0,
                                             scalar_type tolerance,
                                             index_type max_iterations,
                                             bool store_history) {
    const IterationStep step = [&A, &b](const vector_type& x) {
        return symmetric_gauss_seidel_step(A, b, x);
    };
    return run_stationary_solver(A, b, x0, step, tolerance, max_iterations, store_history);
}

IterativeSolverResult chebyshev_accelerated_method(const CSRMatrix& A,
                                                   const vector_type& b,
                                                   const vector_type& x0,
                                                   scalar_type rho,
                                                   const IterationStep& step,
                                                   scalar_type tolerance,
                                                   index_type max_iterations,
                                                   bool store_history) {
    check_system_sizes(A, b, x0);
    ensure_finite_tolerance(tolerance);

    if (!std::isfinite(rho) || rho < 0.0 || rho >= 1.0) {
        throw std::runtime_error("Chebyshev acceleration requires a finite spectral radius rho in [0, 1)");
    }

    IterativeSolverResult result = make_initial_result(A, b, x0, tolerance, store_history);
    if (result.converged || result.diverged || max_iterations == 0) {
        return result;
    }

    const clock_type::time_point start_time = clock_type::now();

    vector_type y_prev = x0;
    vector_type y_current = step(y_prev);
    scalar_type current_residual_norm = residual_norm(A, y_current, b);

    result.x = y_current;
    result.iterations = 1;
    result.residual_norm = current_residual_norm;
    result.diverged = !std::isfinite(current_residual_norm) || !is_finite_vector(y_current);
    if (store_history) {
        append_history(result, current_residual_norm, start_time);
    }
    if (result.diverged) {
        return result;
    }
    if (current_residual_norm <= tolerance) {
        result.converged = true;
        return result;
    }
    if (max_iterations == 1) {
        return result;
    }

    scalar_type omega = 2.0 / (2.0 - rho * rho);

    for (index_type iter = 2; iter <= max_iterations; ++iter) {
        const vector_type step_value = step(y_current);
        vector_type y_next(y_current.size(), 0.0);
        for (index_type i = 0; i < y_next.size(); ++i) {
            y_next[i] = omega * (step_value[i] - y_prev[i]) + y_prev[i];
        }

        current_residual_norm = residual_norm(A, y_next, b);
        result.x = y_next;
        result.iterations = iter;
        result.residual_norm = current_residual_norm;
        result.diverged = !std::isfinite(current_residual_norm) || !is_finite_vector(y_next);

        if (store_history) {
            append_history(result, current_residual_norm, start_time);
        }

        if (result.diverged) {
            return result;
        }

        if (current_residual_norm <= tolerance) {
            result.converged = true;
            return result;
        }

        const scalar_type next_omega = 1.0 / (1.0 - (rho * rho * omega / 4.0));
        y_prev = std::move(y_current);
        y_current = std::move(y_next);
        omega = next_omega;
    }

    result.x = y_current;
    return result;
}


vector_type sor_step(const CSRMatrix& A,
                     const vector_type& b,
                     const vector_type& x,
                     scalar_type omega) {
    return sor_step_impl(A, b, x, omega);
}

IterativeSolverResult sor(const CSRMatrix& A,
                          const vector_type& b,
                          const vector_type& x0,
                          scalar_type omega,
                          scalar_type tolerance,
                          index_type max_iterations,
                          bool store_history) {
    const IterationStep step = [&A, &b, omega](const vector_type& x) {
        return sor_step_impl(A, b, x, omega);
    };
    return run_stationary_solver(A, b, x0, step, tolerance, max_iterations, store_history);
}

}
