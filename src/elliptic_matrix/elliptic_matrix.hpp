#pragma once

#include "csr_matrix/csr_matrix.hpp"

namespace slae {

CSRMatrix generate_elliptic_matrix(index_type side_length,
                                   scalar_type a = -1.0,
                                   scalar_type b = 2.0);

scalar_type elliptic_lambda_min(index_type side_length,
                                scalar_type a = -1.0,
                                scalar_type b = 2.0);

scalar_type elliptic_lambda_max(index_type side_length,
                                scalar_type a = -1.0,
                                scalar_type b = 2.0);

scalar_type optimal_sor_omega(index_type side_length,
                              scalar_type a = -1.0,
                              scalar_type b = 2.0);

} // namespace slae
