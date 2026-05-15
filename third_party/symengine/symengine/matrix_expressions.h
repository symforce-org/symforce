#ifndef SYMENGINE_MATRIXEXPR_H
#define SYMENGINE_MATRIXEXPR_H

#include <symengine/basic.h>
#include <symengine/assumptions.h>
#include <symengine/integer.h>
#include <symengine/matrices/matrix_expr.h>
#include <symengine/matrices/size.h>
#include <symengine/matrices/identity_matrix.h>
#include <symengine/matrices/zero_matrix.h>
#include <symengine/matrices/matrix_symbol.h>
#include <symengine/matrices/diagonal_matrix.h>
#include <symengine/matrices/immutable_dense_matrix.h>
#include <symengine/matrices/matrix_add.h>
#include <symengine/matrices/hadamard_product.h>
#include <symengine/matrices/matrix_mul.h>
#include <symengine/matrices/trace.h>
#include <symengine/matrices/conjugate_matrix.h>
#include <symengine/matrices/transpose.h>
#include <symengine/matrices/size.h>

namespace SymEngine
{

tribool is_diagonal(const MatrixExpr &m,
                    const Assumptions *assumptions = nullptr);
tribool is_lower(const MatrixExpr &m, const Assumptions *assumptions = nullptr);
tribool is_real(const MatrixExpr &m, const Assumptions *assumptions = nullptr);
tribool is_square(const MatrixExpr &m,
                  const Assumptions *assumptions = nullptr);
tribool is_symmetric(const MatrixExpr &m,
                     const Assumptions *assumptions = nullptr);
tribool is_toeplitz(const MatrixExpr &m,
                    const Assumptions *assumptions = nullptr);
tribool is_upper(const MatrixExpr &m, const Assumptions *assumptions = nullptr);
tribool is_zero(const MatrixExpr &m, const Assumptions *assumptions = nullptr);

}; // namespace SymEngine

#endif
