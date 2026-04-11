#pragma once

#include "csr_matrix/csr_matrix.hpp"

#include <functional>

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

IterativeSolverResult chebyshev_simple_iteration(const CSRMatrix& A,
                                                 const vector_type& b,
                                                 const vector_type& x0,
                                                 scalar_type lambda_min,
                                                 scalar_type lambda_max,
                                                 index_type roots_count,
                                                 scalar_type tolerance,
                                                 index_type max_iterations,
                                                 bool store_history = false);

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

using IterationStep = std::function<vector_type(const vector_type&)>;

vector_type jacobi_step(const CSRMatrix& A,
                        const vector_type& b,
                        const vector_type& x);
vector_type gauss_seidel_step(const CSRMatrix& A,
                              const vector_type& b,
                              const vector_type& x);
vector_type symmetric_gauss_seidel_step(const CSRMatrix& A,
                                        const vector_type& b,
                                        const vector_type& x);

scalar_type estimate_spectral_radius(const IterationStep& homogeneous_step,
                                     index_type size,
                                     index_type max_iterations = 1000,
                                     scalar_type tolerance = 1e-10);

IterativeSolverResult symmetric_gauss_seidel(const CSRMatrix& A,
                                             const vector_type& b,
                                             const vector_type& x0,
                                             scalar_type tolerance,
                                             index_type max_iterations,
                                             bool store_history = false);

IterativeSolverResult chebyshev_accelerated_method(const CSRMatrix& A,
                                                   const vector_type& b,
                                                   const vector_type& x0,
                                                   scalar_type rho,
                                                   const IterationStep& step,
                                                   scalar_type tolerance,
                                                   index_type max_iterations,
                                                   bool store_history = false);

vector_type sor_step(const CSRMatrix& A,
                     const vector_type& b,
                     const vector_type& x,
                     scalar_type omega);

IterativeSolverResult sor(const CSRMatrix& A,
                          const vector_type& b,
                          const vector_type& x0,
                          scalar_type omega,
                          scalar_type tolerance,
                          index_type max_iterations,
                          bool store_history = false);


}
