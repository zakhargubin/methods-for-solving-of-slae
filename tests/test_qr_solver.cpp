#include "householder_qr/householder_qr.hpp"

#include <cmath>
#include <iostream>

namespace {

bool close(slae::scalar_type lhs, slae::scalar_type rhs, slae::scalar_type eps = 1e-9) {
    return std::abs(lhs - rhs) < eps;
}

}  // namespace

int main() {
    slae::DenseMatrix A(3, 3);
    A(0, 0) = 2.0;
    A(0, 1) = 1.0;
    A(0, 2) = 1.0;
    A(1, 0) = 1.0;
    A(1, 1) = 3.0;
    A(1, 2) = 2.0;
    A(2, 0) = 1.0;
    A(2, 1) = 0.0;
    A(2, 2) = 0.0;

    const slae::vector_type expected = {6.0, -2.0, -3.0};
    const slae::vector_type b = A * expected;
    const slae::vector_type x = slae::qr_solve(A, b);

    if (x.size() != expected.size()) {
        std::cout << "fail: wrong QR solution size\n";
        return 1;
    }

    for (slae::index_type i = 0; i < x.size(); ++i) {
        if (!close(x[i], expected[i], 1e-8)) {
            std::cout << "fail: wrong QR solution\n";
            return 1;
        }
    }

    std::cout << "success\n";
    return 0;
}
