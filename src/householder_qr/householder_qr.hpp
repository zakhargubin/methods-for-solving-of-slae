#pragma once

#include "dense_matrix/dense_matrix.hpp"

namespace slae {

struct QRDecomposition {
    DenseMatrix Q;
    DenseMatrix R;
};

QRDecomposition householder_qr(const DenseMatrix& matrix);

vector_type qr_solve(const DenseMatrix& Q,
                     const DenseMatrix& R,
                     const vector_type& rhs);

vector_type qr_solve(const QRDecomposition& decomposition,
                     const vector_type& rhs);

vector_type qr_solve(const DenseMatrix& matrix,
                     const vector_type& rhs);

}
