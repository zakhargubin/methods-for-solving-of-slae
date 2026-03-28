#include "dense_matrix/dense_matrix.hpp"

#include <stdexcept>

namespace slae {

DenseMatrix::DenseMatrix(index_type rows, index_type cols)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {
}

DenseMatrix::DenseMatrix(index_type rows, index_type cols, const vector_type& data)
    : rows_(rows), cols_(cols), data_(data) {
    if (data_.size() != rows_ * cols_) {
        throw std::runtime_error("DenseMatrix data size must be rows * cols");
    }
}

index_type DenseMatrix::nrows() const {
    return rows_;
}

index_type DenseMatrix::ncols() const {
    return cols_;
}

index_type DenseMatrix::flat_index(index_type i, index_type j) const {
    if (i >= rows_ || j >= cols_) {
        throw std::out_of_range("DenseMatrix index out of range");
    }
    return i * cols_ + j;
}

scalar_type DenseMatrix::operator()(index_type i, index_type j) const {
    return data_[flat_index(i, j)];
}

scalar_type& DenseMatrix::operator()(index_type i, index_type j) {
    return data_[flat_index(i, j)];
}

vector_type DenseMatrix::get_row(index_type i) const {
    if (i >= rows_) {
        throw std::out_of_range("DenseMatrix row index out of range");
    }

    vector_type row(cols_, 0.0);
    for (index_type j = 0; j < cols_; ++j) {
        row[j] = (*this)(i, j);
    }
    return row;
}

vector_type DenseMatrix::operator*(const vector_type& x) const {
    if (x.size() != cols_) {
        throw std::runtime_error("Vector size must match number of columns in DenseMatrix");
    }

    vector_type out(rows_, 0.0);
    for (index_type i = 0; i < rows_; ++i) {
        scalar_type sum = 0.0;
        for (index_type j = 0; j < cols_; ++j) {
            sum += (*this)(i, j) * x[j];
        }
        out[i] = sum;
    }
    return out;
}

}
