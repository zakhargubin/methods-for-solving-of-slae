#include "tridiagonal_solver/tridiagonal_solver.hpp"

#include <cmath>
#include <iostream>

namespace {

bool close(slae::scalar_type x, slae::scalar_type y, slae::scalar_type eps = 1e-12) {
    return std::abs(x - y) < eps;
}

}

int main() {
    const slae::vector_type a = {1.0, 1.0};
    const slae::vector_type b = {2.0, 2.0, 2.0};
    const slae::vector_type c = {1.0, 1.0};
    const slae::vector_type d = {4.0, 4.0, 4.0};

    const slae::vector_type x = slae::tridiagonal_solver(a, b, c, d);

    if (x.size() != 3) {
        std::cout << "ошибка: неправильная размерность ответа\n";
        return 1;
    }

    if (!close(x[0], 2.0) || !close(x[1], 0.0) || !close(x[2], 2.0)) {
        std::cout << "ошибка: неверное решение\n";
        std::cout << "x = [" << x[0] << ", " << x[1] << ", " << x[2] << "]\n";
        return 1;
    }

    std::cout << "успех\n";
    return 0;
}
