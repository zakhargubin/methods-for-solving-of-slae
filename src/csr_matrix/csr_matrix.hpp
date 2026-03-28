#pragma once

#include "common/types.hpp"

#include <vector>

namespace slae {

struct DokEntry {
    index_type row;
    index_type col;
    scalar_type value;
};

class CSRMatrix {
public:
    CSRMatrix(index_type rows, index_type cols, const std::vector<DokEntry>& entries);

    index_type nrows() const;
    index_type ncols() const;
    index_type nnz() const;

    scalar_type operator()(index_type i, index_type j) const;
    vector_type operator*(const vector_type& x) const;

    const std::vector<scalar_type>& values() const;
    const std::vector<index_type>& cols() const;
    const std::vector<index_type>& rows() const;

private:
    index_type rows_count_;
    index_type cols_count_;
    std::vector<scalar_type> values_;
    std::vector<index_type> col_indices_;
    std::vector<index_type> row_ptr_;
};

}
