#include "iterative_solvers/iterative_solvers.hpp"
#include "vector_operations/vector_operations.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>
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
        throw std::runtime_error("Chebyshev acceleration requires finite spectral bounds");
    }
    if (lambda_min <= 0.0 || lambda_max <= 0.0) {
        throw std::runtime_error("Chebyshev acceleration requires positive spectral bounds");
    }
    if (lambda_min > lambda_max) {
        throw std::runtime_error("Chebyshev acceleration requires lambda_min <= lambda_max");
    }
    if (!is_power_of_two(roots_count)) {
        throw std::runtime_error("Chebyshev acceleration requires the number of roots to be a power of two");
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
    scalar_type x_norm = euclidean_norm(x);
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

IterativeSolverResult jacobi(const CSRMatrix& A,
                             const vector_type& b,
                             const vector_type& x0,
                             scalar_type tolerance,
                             index_type max_iterations,
                             bool store_history) {
    check_system_sizes(A, b, x0);

    IterativeSolverResult result = make_initial_result(A, b, x0, tolerance, store_history);
    if (result.converged || result.diverged) {
        return result;
    }

    const clock_type::time_point start_time = clock_type::now();

    vector_type x_old = x0;
    vector_type x_new(A.ncols(), 0.0);

    for (index_type iter = 1; iter <= max_iterations; ++iter) {
        for (index_type i = 0; i < A.nrows(); ++i) {
            scalar_type diagonal = 0.0;
            scalar_type sum = 0.0;

            for (index_type k = A.rows()[i]; k < A.rows()[i + 1]; ++k) {
                const index_type j = A.cols()[k];
                const scalar_type value = A.values()[k];
                if (j == i) {
                    diagonal = value;
                } else {
                    sum += value * x_old[j];
                }
            }

            if (nearly_zero(diagonal)) {
                throw std::runtime_error("Jacobi method requires non-zero diagonal elements");
            }

            x_new[i] = (b[i] - sum) / diagonal;
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

IterativeSolverResult gauss_seidel(const CSRMatrix& A,
                                   const vector_type& b,
                                   const vector_type& x0,
                                   scalar_type tolerance,
                                   index_type max_iterations,
                                   bool store_history) {
    check_system_sizes(A, b, x0);

    IterativeSolverResult result = make_initial_result(A, b, x0, tolerance, store_history);
    if (result.converged || result.diverged) {
        return result;
    }

    const clock_type::time_point start_time = clock_type::now();

    vector_type x_old = x0;
    vector_type x_new = x0;

    for (index_type iter = 1; iter <= max_iterations; ++iter) {

        for (index_type i = 0; i < A.nrows(); ++i) {
            scalar_type diagonal = 0.0;
            scalar_type sum = b[i];

            for (index_type k = A.rows()[i]; k < A.rows()[i + 1]; ++k) {
                const index_type j = A.cols()[k];
                const scalar_type value = A.values()[k];
                if (j == i) {
                    diagonal = value;
                } else if (j < i) {
                    sum -= value * x_new[j];
                } else {
                    sum -= value * x_old[j];
                }
            }

            if (nearly_zero(diagonal)) {
                throw std::runtime_error("Gauss-Seidel method requires non-zero diagonal elements");
            }

            x_new[i] = sum / diagonal;
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

IterativeSolverResult simple_iteration(const CSRMatrix& A,
                                       const vector_type& b,
                                       const vector_type& x0,
                                       scalar_type tau,
                                       scalar_type tolerance,
                                       index_type max_iterations,
                                       bool store_history) {
    check_system_sizes(A, b, x0);

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

}
