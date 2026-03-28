#pragma once

#include "common/types.hpp"

namespace slae {

vector_type operator+(const vector_type& lhs, const vector_type& rhs);
vector_type operator*(const vector_type& x, scalar_type alpha);
vector_type operator*(scalar_type alpha, const vector_type& x);
scalar_type operator*(const vector_type& lhs, const vector_type& rhs);
scalar_type dot(const vector_type& lhs, const vector_type& rhs);

}
