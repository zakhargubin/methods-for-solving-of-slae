#include "vector_operations/vector_operations.hpp"

#include <stdexcept>

namespace {

void check_same_size(const slae::vector_type& lhs, const slae::vector_type& rhs, const char* message) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error(message);
    }
}

}

namespace slae {

vector_type operator+(const vector_type& lhs, const vector_type& rhs) {
    check_same_size(lhs, rhs, "Vector sizes must match for addition");

    vector_type out(lhs.size(), 0.0);
    for (index_type i = 0; i < lhs.size(); ++i) {
        out[i] = lhs[i] + rhs[i];
    }
    return out;
}

vector_type operator*(const vector_type& x, scalar_type alpha) {
    vector_type out(x.size(), 0.0);
    for (index_type i = 0; i < x.size(); ++i) {
        out[i] = x[i] * alpha;
    }
    return out;
}

vector_type operator*(scalar_type alpha, const vector_type& x) {
    return x * alpha;
}

scalar_type dot(const vector_type& lhs, const vector_type& rhs) {
    check_same_size(lhs, rhs, "Vector sizes must match for dot product");

    scalar_type out = 0.0;
    for (index_type i = 0; i < lhs.size(); ++i) {
        out += lhs[i] * rhs[i];
    }
    return out;
}

scalar_type operator*(const vector_type& lhs, const vector_type& rhs) {
    return dot(lhs, rhs);
}

}
