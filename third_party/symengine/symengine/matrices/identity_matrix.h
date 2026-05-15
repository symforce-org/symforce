#ifndef SYMENGINE_MATRICES_IDENTITY_MATRIX_H
#define SYMENGINE_MATRICES_IDENTITY_MATRIX_H

#include <symengine/basic.h>
#include <symengine/matrices/matrix_expr.h>

namespace SymEngine
{

class IdentityMatrix : public MatrixExpr
{
private:
    RCP<const Basic> n_; // n >= 0

public:
    IdentityMatrix(const RCP<const Basic> &n) : n_(n)
    {
        SYMENGINE_ASSIGN_TYPEID();
        SYMENGINE_ASSERT(is_canonical(n));
    }

    IMPLEMENT_TYPEID(SYMENGINE_IDENTITYMATRIX)
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    vec_basic get_args() const override;
    bool is_canonical(const RCP<const Basic> &n) const;

    inline const RCP<const Basic> &size() const
    {
        return n_;
    }
};

RCP<const MatrixExpr> identity_matrix(const RCP<const Basic> &n);

} // namespace SymEngine

#endif
