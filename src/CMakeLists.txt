add_library(methods_for_solving_of_slae)

target_sources(methods_for_solving_of_slae
    PRIVATE
        tridiagonal_solver/tridiagonal_solver.cpp
        dense_matrix/dense_matrix.cpp
        csr_matrix/csr_matrix.cpp
        vector_operations/vector_operations.cpp
        householder_qr/householder_qr.cpp
        iterative_solvers/iterative_solvers.cpp
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            common/types.hpp
            tridiagonal_solver/tridiagonal_solver.hpp
            dense_matrix/dense_matrix.hpp
            csr_matrix/csr_matrix.hpp
            vector_operations/vector_operations.hpp
            householder_qr/householder_qr.hpp
            iterative_solvers/iterative_solvers.hpp
)
