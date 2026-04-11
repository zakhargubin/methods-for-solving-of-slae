#pragma once

#include "iterative_solvers/iterative_solvers.hpp"

namespace slae {

IterativeSolverResult steepest_descent(const CSRMatrix& A,
                                       const vector_type& b,
                                       const vector_type& x0,
                                       scalar_type tolerance,
                                       index_type max_iterations,
                                       bool store_history = false);

IterativeSolverResult conjugate_gradient(const CSRMatrix& A,
                                         const vector_type& b,
                                         const vector_type& x0,
                                         scalar_type tolerance,
                                         index_type max_iterations,
                                         bool store_history = false);

}
