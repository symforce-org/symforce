#ifndef SYMENGINE_MATRICES_DIAGONAL_MATRIX_H
#define SYMENGINE_MATRICES_DIAGONAL_MATRIX_H

#include <symengine/basic.h>
#include <symengine/matrices/matrix_expr.h>

namespace SymEngine
{

class DiagonalMatrix : public MatrixExpr
{
private:
    vec_basic diag_;

public:
    DiagonalMatrix(const vec_basic &container) : diag_(container)
    {
        SYMENGINE_ASSIGN_TYPEID();
        SYMENGINE_ASSERT(is_canonical(container));
    }

    IMPLEMENT_TYPEID(SYMENGINE_DIAGONALMATRIX)
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    bool is_canonical(const vec_basic &container) const;

    vec_basic get_args() const override
    {
        return vec_basic(diag_.begin(), diag_.end());
    }

    inline const vec_basic &get_container() const
    {
        return diag_;
    }

    inline RCP<const Basic> get(size_t i) const
    {
        return diag_[i];
    }
};

bool is_zero_vec(const vec_basic &container);
bool is_identity_vec(const vec_basic &container);

RCP<const MatrixExpr> diagonal_matrix(const vec_basic &container);

} // namespace SymEngine

#endif
