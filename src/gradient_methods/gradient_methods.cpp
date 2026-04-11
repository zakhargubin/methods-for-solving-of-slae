#include "gradient_methods/gradient_methods.hpp"

#include "vector_operations/vector_operations.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace {

using clock_type = std::chrono::steady_clock;

bool is_finite_vector(const slae::vector_type& x) {
    for (const slae::scalar_type value : x) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
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

}

namespace slae {

IterativeSolverResult steepest_descent(const CSRMatrix& A,
                                       const vector_type& b,
                                       const vector_type& x0,
                                       scalar_type tolerance,
                                       index_type max_iterations,
                                       bool store_history) {
    check_system_sizes(A, b, x0);
    ensure_finite_tolerance(tolerance);

    IterativeSolverResult result = make_initial_result(A, b, x0, tolerance, store_history);
    if (result.converged || result.diverged || max_iterations == 0) {
        return result;
    }

    const clock_type::time_point start_time = clock_type::now();
    vector_type x_current = x0;
    vector_type residual_current = residual_vector(A, x_current, b);

    for (index_type iter = 1; iter <= max_iterations; ++iter) {
        const vector_type Ar = A * residual_current;
        const scalar_type numerator = dot(residual_current, residual_current);
        const scalar_type denominator = dot(residual_current, Ar);

        if (!std::isfinite(numerator) || !std::isfinite(denominator) || denominator <= 0.0) {
            result.diverged = true;
            result.x = x_current;
            result.iterations = iter - 1;
            result.residual_norm = euclidean_norm(residual_current);
            return result;
        }

        const scalar_type alpha = numerator / denominator;
        vector_type x_next(x_current.size(), 0.0);
        vector_type residual_next(residual_current.size(), 0.0);
        for (index_type i = 0; i < x_current.size(); ++i) {
            x_next[i] = x_current[i] - alpha * residual_current[i];
            residual_next[i] = residual_current[i] - alpha * Ar[i];
        }

        const scalar_type current_residual_norm = euclidean_norm(residual_next);
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

        x_current = std::move(x_next);
        residual_current = std::move(residual_next);
    }

    result.x = x_current;
    return result;
}

IterativeSolverResult conjugate_gradient(const CSRMatrix& A,
                                         const vector_type& b,
                                         const vector_type& x0,
                                         scalar_type tolerance,
                                         index_type max_iterations,
                                         bool store_history) {
    check_system_sizes(A, b, x0);
    ensure_finite_tolerance(tolerance);

    IterativeSolverResult result = make_initial_result(A, b, x0, tolerance, store_history);
    if (result.converged || result.diverged || max_iterations == 0) {
        return result;
    }

    const clock_type::time_point start_time = clock_type::now();
    vector_type x_current = x0;
    vector_type residual_current = residual_vector(A, x_current, b);
    vector_type direction = residual_current;
    scalar_type residual_dot = dot(residual_current, residual_current);

    for (index_type iter = 1; iter <= max_iterations; ++iter) {
        const vector_type Ad = A * direction;
        const scalar_type denominator = dot(direction, Ad);
        if (!std::isfinite(denominator) || denominator <= 0.0) {
            result.diverged = true;
            result.x = x_current;
            result.iterations = iter - 1;
            result.residual_norm = euclidean_norm(residual_current);
            return result;
        }

        const scalar_type alpha = residual_dot / denominator;
        vector_type x_next(x_current.size(), 0.0);
        vector_type residual_next(residual_current.size(), 0.0);
        for (index_type i = 0; i < x_current.size(); ++i) {
            x_next[i] = x_current[i] - alpha * direction[i];
            residual_next[i] = residual_current[i] - alpha * Ad[i];
        }

        const scalar_type current_residual_norm = euclidean_norm(residual_next);
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

        const scalar_type residual_dot_next = dot(residual_next, residual_next);
        if (!std::isfinite(residual_dot_next)) {
            result.diverged = true;
            return result;
        }

        const scalar_type beta = residual_dot_next / residual_dot;
        vector_type direction_next(direction.size(), 0.0);
        for (index_type i = 0; i < direction.size(); ++i) {
            direction_next[i] = residual_next[i] + beta * direction[i];
        }

        x_current = std::move(x_next);
        residual_current = std::move(residual_next);
        direction = std::move(direction_next);
        residual_dot = residual_dot_next;
    }

    result.x = x_current;
    return result;
}

}
