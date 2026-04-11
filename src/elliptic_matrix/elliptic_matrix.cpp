#include "elliptic_matrix/elliptic_matrix.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

bool nearly_zero(slae::scalar_type x) {
    return std::abs(x) <= slae::tolerance;
}

}

namespace slae {

CSRMatrix generate_elliptic_matrix(index_type side_length,
                                   scalar_type a,
                                   scalar_type b_param) {
    if (side_length == 0) {
        throw std::runtime_error("Elliptic matrix generation requires a positive side length");
    }
    if (!std::isfinite(a) || !std::isfinite(b_param)) {
        throw std::runtime_error("Elliptic matrix generation requires finite coefficients");
    }
    if (b_param <= 0.0 || nearly_zero(a)) {
        throw std::runtime_error("Elliptic matrix generation requires finite non-zero stencil coefficients");
    }
    if (elliptic_lambda_min(side_length, a, b_param) <= 0.0) {
        throw std::runtime_error("Elliptic matrix generation requires a positive definite stencil");
    }

    const index_type total_size = side_length * side_length;
    std::vector<DokEntry> entries;
    entries.reserve(total_size * 5);
    const scalar_type diagonal = 2.0 * b_param;

    for (index_type row = 0; row < side_length; ++row) {
        for (index_type col = 0; col < side_length; ++col) {
            const index_type index = row * side_length + col;
            entries.push_back({index, index, diagonal});

            if (col > 0) {
                entries.push_back({index, index - 1, a});
            }
            if (col + 1 < side_length) {
                entries.push_back({index, index + 1, a});
            }
            if (row > 0) {
                entries.push_back({index, index - side_length, a});
            }
            if (row + 1 < side_length) {
                entries.push_back({index, index + side_length, a});
            }
        }
    }

    return CSRMatrix(total_size, total_size, entries);
}

scalar_type elliptic_lambda_min(index_type side_length,
                                scalar_type a,
                                scalar_type b_param) {
    if (side_length == 0) {
        throw std::runtime_error("Elliptic eigenvalue estimation requires a positive side length");
    }
    const scalar_type pi = std::acos(-1.0);
    const scalar_type cosine = std::cos(pi / static_cast<scalar_type>(side_length + 1));
    return 2.0 * b_param - 4.0 * std::abs(a) * cosine;
}

scalar_type elliptic_lambda_max(index_type side_length,
                                scalar_type a,
                                scalar_type b_param) {
    if (side_length == 0) {
        throw std::runtime_error("Elliptic eigenvalue estimation requires a positive side length");
    }
    const scalar_type pi = std::acos(-1.0);
    const scalar_type cosine = std::cos(pi / static_cast<scalar_type>(side_length + 1));
    return 2.0 * b_param + 4.0 * std::abs(a) * cosine;
}

scalar_type optimal_sor_omega(index_type side_length,
                              scalar_type a,
                              scalar_type b_param) {
    if (side_length == 0) {
        throw std::runtime_error("SOR omega estimation requires a positive side length");
    }
    if (b_param <= 0.0 || nearly_zero(a)) {
        throw std::runtime_error("SOR omega estimation requires finite non-zero stencil coefficients");
    }
    if (elliptic_lambda_min(side_length, a, b_param) <= 0.0) {
        throw std::runtime_error("SOR omega estimation requires a positive definite stencil");
    }

    const scalar_type pi = std::acos(-1.0);
    const scalar_type mu = (2.0 * std::abs(a) / b_param)
        * std::cos(pi / static_cast<scalar_type>(side_length + 1));
    if (!(mu >= 0.0 && mu < 1.0)) {
        throw std::runtime_error("SOR omega estimation produced an invalid Jacobi spectral radius");
    }

    const scalar_type denominator = 1.0 + std::sqrt(1.0 - mu * mu);
    return 1.0 + (mu / denominator) * (mu / denominator);
}

}
