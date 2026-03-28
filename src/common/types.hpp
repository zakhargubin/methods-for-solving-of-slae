#pragma once

#include <cstddef>
#include <vector>

namespace slae {

using scalar_type = double;
using index_type = std::size_t;
using vector_type = std::vector<scalar_type>;

inline constexpr scalar_type tolerance = 1e-12;

}
