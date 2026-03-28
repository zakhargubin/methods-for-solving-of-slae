#include "tridiagonal_solver/tridiagonal_solver.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace {

bool is_zero(slae::scalar_type x) {
    return std::abs(x) < slae::tolerance;
}

}  

namespace slae {

vector_type tridiagonal_solver(
    const vector_type& a,
    const vector_type& b,
    const vector_type& c,
    const vector_type& d
) {
    if (b.size() != d.size()) {
        throw std::runtime_error("Размеры главной диагонали и свободного столбца не равны.");
    }

    const index_type n = b.size();

    if (n == 0) {
        return {};
    }

    if (b[0] == 0.0) {
        throw std::runtime_error("Нулевой коэффициент при первом x");
    }

    if (a.size() != n - 1 || c.size() != n - 1) {
        throw std::runtime_error("Несоответствие размеров неглавных диагоналей");
    }

    if (n == 1) {
        return {d[0] / b[0]};
    }

    vector_type p(n - 1, 0.0);
    vector_type q(n - 1, 0.0);
    vector_type x(n, 0.0);

    p[0] = -c[0] / b[0];
    q[0] = d[0] / b[0];

    for (index_type i = 1; i < n - 1; ++i) {
        const scalar_type denom = a[i - 1] * p[i - 1] + b[i];

        if (is_zero(denom)) {
            throw std::runtime_error(
                "Коэффициент при x равен 0 в строке " + std::to_string(i)
            );
        }

        p[i] = -c[i] / denom;
        q[i] = (d[i] - a[i - 1] * q[i - 1]) / denom;
    }

    const scalar_type last_denom = a[n - 2] * p[n - 2] + b[n - 1];

    if (is_zero(last_denom)) {
        throw std::runtime_error("Коэффициент при x равен 0 в последней строке");
    }

    x[n - 1] = (d[n - 1] - a[n - 2] * q[n - 2]) / last_denom;

    for (index_type i = n - 1; i-- > 0;) {
        x[i] = p[i] * x[i + 1] + q[i];
    }

    return x;
}

}
