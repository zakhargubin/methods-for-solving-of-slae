#pragma once

#include "common/types.hpp"

namespace slae {

vector_type tridiagonal_solver(
    const vector_type& a,
    const vector_type& b,
    const vector_type& c,
    const vector_type& d
);

}
