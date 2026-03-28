#pragma once

#include "csr_matrix/csr_matrix.hpp"

namespace slae {

struct IterativeSolverResult {
    vector_type x;
    index_type iterations;
    scalar_type residual_norm;
    bool converged;
    bool diverged;
    vector_type residual_norm_history;
    vector_type elapsed_microseconds_history;
};

scalar_type euclidean_norm(const vector_type& x);
vector_type residual_vector(const CSRMatrix& A,
                            const vector_type& x,
                            const vector_type& b);
scalar_type residual_norm(const CSRMatrix& A,
                          const vector_type& x,
                          const vector_type& b);

scalar_type power_iteration_max_eigenvalue(const CSRMatrix& A,
                                           index_type max_iterations = 1000,
                                           scalar_type tolerance = 1e-10);

IterativeSolverResult jacobi(const CSRMatrix& A,
                             const vector_type& b,
                             const vector_type& x0,
                             scalar_type tolerance,
                             index_type max_iterations,
                             bool store_history = false);

IterativeSolverResult gauss_seidel(const CSRMatrix& A,
                                   const vector_type& b,
                                   const vector_type& x0,
                                   scalar_type tolerance,
                                   index_type max_iterations,
                                   bool store_history = false);

IterativeSolverResult simple_iteration(const CSRMatrix& A,
                                       const vector_type& b,
                                       const vector_type& x0,
                                       scalar_type tau,
                                       scalar_type tolerance,
                                       index_type max_iterations,
                                       bool store_history = false);

IterativeSolverResult chebyshev_simple_iteration(const CSRMatrix& A,
                                                 const vector_type& b,
                                                 const vector_type& x0,
                                                 scalar_type lambda_min,
                                                 scalar_type lambda_max,
                                                 index_type roots_count,
                                                 scalar_type tolerance,
                                                 index_type max_iterations,
                                                 bool store_history = false);

}
