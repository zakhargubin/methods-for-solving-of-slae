#include "householder_qr/householder_qr.hpp"

#include <cmath>
#include <iostream>

namespace {

bool close(slae::scalar_type lhs, slae::scalar_type rhs, slae::scalar_type eps = 1e-12) {
    return std::abs(lhs - rhs) < eps;
}

slae::DenseMatrix multiply(const slae::DenseMatrix& lhs, const slae::DenseMatrix& rhs) {
    if (lhs.ncols() != rhs.nrows()) {
        throw std::runtime_error("Matrix sizes do not match for multiplication");
    }

    slae::DenseMatrix out(lhs.nrows(), rhs.ncols());
    for (slae::index_type i = 0; i < lhs.nrows(); ++i) {
        for (slae::index_type j = 0; j < rhs.ncols(); ++j) {
            slae::scalar_type sum = 0.0;
            for (slae::index_type k = 0; k < lhs.ncols(); ++k) {
                sum += lhs(i, k) * rhs(k, j);
            }
            out(i, j) = sum;
        }
    }
    return out;
}

slae::DenseMatrix transpose(const slae::DenseMatrix& matrix) {
    slae::DenseMatrix out(matrix.ncols(), matrix.nrows());
    for (slae::index_type i = 0; i < matrix.nrows(); ++i) {
        for (slae::index_type j = 0; j < matrix.ncols(); ++j) {
            out(j, i) = matrix(i, j);
        }
    }
    return out;
}

}

int main() {
    const slae::vector_type data = {
        1.0,  2.0,  3.0,
        4.0,  5.0,  6.0,
        7.0,  8.0, 10.0,
        1.0,  3.0,  5.0
    };
    const slae::DenseMatrix A(4, 3, data);

    const slae::QRDecomposition qr = slae::householder_qr(A);

    if (qr.Q.nrows() != 4 || qr.Q.ncols() != 4 || qr.R.nrows() != 4 || qr.R.ncols() != 3) {
        std::cout << "fail: wrong QR matrix sizes\n";
        return 1;
    }

    for (slae::index_type i = 0; i < qr.R.nrows(); ++i) {
        const slae::index_type max_j = (i < qr.R.ncols()) ? i : qr.R.ncols();
        for (slae::index_type j = 0; j < max_j; ++j) {
            if (!close(qr.R(i, j), 0.0)) {
                std::cout << "fail: R is not upper triangular\n";
                return 1;
            }
        }
    }

    const slae::DenseMatrix reconstructed = multiply(qr.Q, qr.R);
    for (slae::index_type i = 0; i < A.nrows(); ++i) {
        for (slae::index_type j = 0; j < A.ncols(); ++j) {
            if (!close(reconstructed(i, j), A(i, j), 1e-8)) {
                std::cout << "fail: QR does not reconstruct A\n";
                return 1;
            }
        }
    }

    const slae::DenseMatrix qtq = multiply(transpose(qr.Q), qr.Q);
    for (slae::index_type i = 0; i < qtq.nrows(); ++i) {
        for (slae::index_type j = 0; j < qtq.ncols(); ++j) {
            const slae::scalar_type expected = (i == j) ? 1.0 : 0.0;
            if (!close(qtq(i, j), expected, 1e-8)) {
                std::cout << "fail: Q is not orthogonal\n";
                return 1;
            }
        }
    }

    std::cout << "success\n";
    return 0;
}
