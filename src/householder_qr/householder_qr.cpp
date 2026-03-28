#include "householder_qr/householder_qr.hpp"

#include <cmath>
#include <stdexcept>

namespace {

bool nearly_zero(slae::scalar_type x) {
    return std::abs(x) <= slae::tolerance;
}

void apply_reflector_from_left(slae::DenseMatrix& matrix,
                               slae::index_type row_begin,
                               slae::index_type col_begin,
                               const slae::vector_type& v,
                               slae::scalar_type beta) {
    for (slae::index_type j = col_begin; j < matrix.ncols(); ++j) {
        slae::scalar_type dot = 0.0;
        for (slae::index_type offset = 0; offset < v.size(); ++offset) {
            dot += v[offset] * matrix(row_begin + offset, j);
        }

        for (slae::index_type offset = 0; offset < v.size(); ++offset) {
            matrix(row_begin + offset, j) -= beta * v[offset] * dot;
        }
    }
}

void apply_reflector_from_right(slae::DenseMatrix& matrix,
                                slae::index_type row_begin,
                                const slae::vector_type& v,
                                slae::scalar_type beta) {
    for (slae::index_type i = 0; i < matrix.nrows(); ++i) {
        slae::scalar_type dot = 0.0;
        for (slae::index_type offset = 0; offset < v.size(); ++offset) {
            dot += matrix(i, row_begin + offset) * v[offset];
        }

        for (slae::index_type offset = 0; offset < v.size(); ++offset) {
            matrix(i, row_begin + offset) -= beta * dot * v[offset];
        }
    }
}

slae::DenseMatrix make_identity(slae::index_type size) {
    slae::DenseMatrix identity(size, size);
    for (slae::index_type i = 0; i < size; ++i) {
        identity(i, i) = 1.0;
    }
    return identity;
}

slae::vector_type multiply_transposed(const slae::DenseMatrix& matrix,
                                      const slae::vector_type& x) {
    if (matrix.nrows() != x.size()) {
        throw std::runtime_error(
            "Vector size must match number of rows for transposed multiplication"
        );
    }

    slae::vector_type out(matrix.ncols(), 0.0);
    for (slae::index_type j = 0; j < matrix.ncols(); ++j) {
        for (slae::index_type i = 0; i < matrix.nrows(); ++i) {
            out[j] += matrix(i, j) * x[i];
        }
    }

    return out;
}

slae::vector_type back_substitution(const slae::DenseMatrix& matrix,
                                    const slae::vector_type& rhs) {
    if (matrix.nrows() != matrix.ncols()) {
        throw std::runtime_error(
            "Back substitution requires a square upper triangular matrix"
        );
    }

    if (rhs.size() != matrix.nrows()) {
        throw std::runtime_error(
            "RHS size must match matrix size in back substitution"
        );
    }

    const slae::index_type size = matrix.ncols();
    slae::vector_type x(size, 0.0);

    for (slae::index_type i = size; i-- > 0;) {
        slae::scalar_type sum = rhs[i];
        for (slae::index_type j = i + 1; j < size; ++j) {
            sum -= matrix(i, j) * x[j];
        }

        const slae::scalar_type diagonal = matrix(i, i);
        if (nearly_zero(diagonal)) {
            throw std::runtime_error(
                "QR solve failed: matrix is singular or rank-deficient"
            );
        }

        x[i] = sum / diagonal;
    }

    return x;
}

}

namespace slae {

QRDecomposition householder_qr(const DenseMatrix& matrix) {
    const index_type m = matrix.nrows();
    const index_type n = matrix.ncols();

    if (m < n) {
        throw std::runtime_error(
            "Householder QR requires matrix with nrows >= ncols"
        );
    }

    DenseMatrix Q = make_identity(m);
    DenseMatrix R = matrix;

    for (index_type k = 0; k < n; ++k) {
        vector_type x(m - k, 0.0);
        for (index_type i = k; i < m; ++i) {
            x[i - k] = R(i, k);
        }

        scalar_type norm_x = 0.0;
        for (const scalar_type value : x) {
            norm_x += value * value;
        }
        norm_x = std::sqrt(norm_x);

        if (nearly_zero(norm_x)) {
            continue;
        }

        vector_type v = x;
        const scalar_type sign = (x[0] >= 0.0) ? 1.0 : -1.0;
        v[0] += sign * norm_x;

        scalar_type v_norm_sq = 0.0;
        for (const scalar_type value : v) {
            v_norm_sq += value * value;
        }

        if (nearly_zero(v_norm_sq)) {
            continue;
        }

        const scalar_type beta = 2.0 / v_norm_sq;

        apply_reflector_from_left(R, k, k, v, beta);
        apply_reflector_from_right(Q, k, v, beta);
    }

    for (index_type i = 0; i < m; ++i) {
        const index_type max_j = (i < n) ? i : n;
        for (index_type j = 0; j < max_j; ++j) {
            if (std::abs(R(i, j)) <= tolerance) {
                R(i, j) = 0.0;
            }
        }
    }

    return {Q, R};
}

vector_type qr_solve(const DenseMatrix& Q,
                     const DenseMatrix& R,
                     const vector_type& rhs) {
    if (Q.nrows() != Q.ncols()) {
        throw std::runtime_error("Q must be square in QR solve");
    }

    if (R.nrows() != Q.nrows()) {
        throw std::runtime_error("Q and R sizes do not match in QR solve");
    }

    if (rhs.size() != Q.nrows()) {
        throw std::runtime_error("RHS size must match Q size in QR solve");
    }

    if (R.nrows() != R.ncols()) {
        throw std::runtime_error(
            "Current QR solve implementation requires a square matrix"
        );
    }

    const vector_type y = multiply_transposed(Q, rhs);
    return back_substitution(R, y);
}

vector_type qr_solve(const QRDecomposition& decomposition,
                     const vector_type& rhs) {
    return qr_solve(decomposition.Q, decomposition.R, rhs);
}

vector_type qr_solve(const DenseMatrix& matrix,
                     const vector_type& rhs) {
    const QRDecomposition decomposition = householder_qr(matrix);
    const auto& [Q, R] = decomposition;
    return qr_solve(Q, R, rhs);
}

}
