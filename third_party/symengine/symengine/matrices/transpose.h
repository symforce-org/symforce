#ifndef SYMENGINE_MATRICES_TRANSPOSE_H
#define SYMENGINE_MATRICES_TRANSPOSE_H

#include <symengine/basic.h>
#include <symengine/matrices/matrix_expr.h>

namespace SymEngine
{

class Transpose : public MatrixExpr
{
private:
    RCP<const MatrixExpr> arg_;

public:
    Transpose(const RCP<const MatrixExpr> &arg) : arg_(arg)
    {
        SYMENGINE_ASSIGN_TYPEID();
        SYMENGINE_ASSERT(is_canonical(arg));
    }

    IMPLEMENT_TYPEID(SYMENGINE_TRANSPOSE)
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    bool is_canonical(const RCP<const MatrixExpr> &arg) const;
    vec_basic get_args() const override;

    inline RCP<const MatrixExpr> get_arg() const
    {
        return arg_;
    }
};

RCP<const MatrixExpr> transpose(const RCP<const MatrixExpr> &arg);
} // namespace SymEngine

#endif
