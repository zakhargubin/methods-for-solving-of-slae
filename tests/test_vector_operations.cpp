#include "vector_operations/vector_operations.hpp"

#include <cmath>
#include <iostream>

namespace {

bool close(slae::scalar_type x,
           slae::scalar_type y,
           slae::scalar_type eps = 1e-12) {
    return std::abs(x - y) < eps;
}

bool vectors_close(const slae::vector_type& lhs,
                   const slae::vector_type& rhs,
                   slae::scalar_type eps = 1e-12) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (slae::index_type i = 0; i < lhs.size(); ++i) {
        if (!close(lhs[i], rhs[i], eps)) {
            return false;
        }
    }

    return true;
}

}

int main() {
    const slae::vector_type a = {1.0, 2.0, 3.0};
    const slae::vector_type b = {4.0, -1.0, 0.5};

    const slae::vector_type expected_sum = {5.0, 1.0, 3.5};
    const slae::vector_type expected_scaled = {2.0, 4.0, 6.0};
    const slae::scalar_type expected_dot = 3.5;

    const slae::vector_type sum = slae::operator+(a, b);
    if (!vectors_close(sum, expected_sum)) {
        std::cout << "fail: vector addition is wrong\n";
        return 1;
    }

    const slae::vector_type scaled_right = slae::operator*(a, 2.0);
    if (!vectors_close(scaled_right, expected_scaled)) {
        std::cout << "fail: vector * scalar is wrong\n";
        return 1;
    }

    const slae::vector_type scaled_left = slae::operator*(2.0, a);
    if (!vectors_close(scaled_left, expected_scaled)) {
        std::cout << "fail: scalar * vector is wrong\n";
        return 1;
    }

    const slae::scalar_type dot_result = slae::dot(a, b);
    if (!close(dot_result, expected_dot)) {
        std::cout << "fail: dot product is wrong\n";
        return 1;
    }

    const slae::scalar_type dot_operator_result = slae::operator*(a, b);
    if (!close(dot_operator_result, expected_dot)) {
        std::cout << "fail: vector operator* is wrong\n";
        return 1;
    }

    std::cout << "success\n";
    return 0;
}
