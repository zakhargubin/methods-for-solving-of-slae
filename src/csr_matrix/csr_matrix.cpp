#include "csr_matrix/csr_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

bool nearly_zero(slae::scalar_type x) {
    return std::abs(x) <= slae::tolerance;
}

}

namespace slae {

CSRMatrix::CSRMatrix(index_type rows, index_type cols, const std::vector<DokEntry>& entries)
    : rows_count_(rows), cols_count_(cols), row_ptr_(rows + 1, 0) {
    std::vector<DokEntry> filtered;
    filtered.reserve(entries.size());

    for (const DokEntry& entry : entries) {
        if (entry.row >= rows_count_ || entry.col >= cols_count_) {
            throw std::runtime_error("DOK entry index out of range");
        }
        if (!nearly_zero(entry.value)) {
            filtered.push_back(entry);
        }
    }

    std::sort(filtered.begin(), filtered.end(), [](const DokEntry& lhs, const DokEntry& rhs) {
        if (lhs.row != rhs.row) {
            return lhs.row < rhs.row;
        }
        return lhs.col < rhs.col;
    });

    std::vector<DokEntry> merged;
    merged.reserve(filtered.size());

    for (const DokEntry& entry : filtered) {
        if (!merged.empty() && merged.back().row == entry.row && merged.back().col == entry.col) {
            merged.back().value += entry.value;
            if (nearly_zero(merged.back().value)) {
                merged.pop_back();
            }
        } else {
            merged.push_back(entry);
        }
    }

    values_.reserve(merged.size());
    col_indices_.reserve(merged.size());

    for (const DokEntry& entry : merged) {
        values_.push_back(entry.value);
        col_indices_.push_back(entry.col);
        ++row_ptr_[entry.row + 1];
    }

    for (index_type i = 0; i < rows_count_; ++i) {
        row_ptr_[i + 1] += row_ptr_[i];
    }
}

index_type CSRMatrix::nrows() const {
    return rows_count_;
}

index_type CSRMatrix::ncols() const {
    return cols_count_;
}

index_type CSRMatrix::nnz() const {
    return values_.size();
}

scalar_type CSRMatrix::operator()(index_type i, index_type j) const {
    if (i >= rows_count_ || j >= cols_count_) {
        throw std::out_of_range("CSRMatrix index out of range");
    }

    for (index_type k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
        if (col_indices_[k] == j) {
            return values_[k];
        }
        if (col_indices_[k] > j) {
            break;
        }
    }
    return 0.0;
}

vector_type CSRMatrix::operator*(const vector_type& x) const {
    if (x.size() != cols_count_) {
        throw std::runtime_error("Vector size must match number of columns in CSRMatrix");
    }

    vector_type out(rows_count_, 0.0);
    for (index_type i = 0; i < rows_count_; ++i) {
        for (index_type k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
            out[i] += values_[k] * x[col_indices_[k]];
        }
    }
    return out;
}

const std::vector<scalar_type>& CSRMatrix::values() const {
    return values_;
}

const std::vector<index_type>& CSRMatrix::cols() const {
    return col_indices_;
}

const std::vector<index_type>& CSRMatrix::rows() const {
    return row_ptr_;
}

}
