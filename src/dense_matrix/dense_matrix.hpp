#pragma once

#include "common/types.hpp"

namespace slae {

class DenseMatrix {
public:
    DenseMatrix(index_type rows, index_type cols);
    DenseMatrix(index_type rows, index_type cols, const vector_type& data);

    index_type nrows() const;
    index_type ncols() const;

    scalar_type operator()(index_type i, index_type j) const;
    scalar_type& operator()(index_type i, index_type j);

    vector_type get_row(index_type i) const;
    vector_type operator*(const vector_type& x) const;

private:
    index_type rows_;
    index_type cols_;
    vector_type data_;

    index_type flat_index(index_type i, index_type j) const;
};

}
