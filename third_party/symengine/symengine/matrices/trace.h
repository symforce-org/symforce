#ifndef SYMENGINE_MATRICES_TRACE_H
#define SYMENGINE_MATRICES_TRACE_H

#include <symengine/basic.h>
#include <symengine/matrices/matrix_expr.h>

namespace SymEngine
{

class Trace : public Basic
{
private:
    RCP<const Basic> arg_;

public:
    Trace(const RCP<const MatrixExpr> &arg) : arg_(arg)
    {
        SYMENGINE_ASSIGN_TYPEID();
    }

    IMPLEMENT_TYPEID(SYMENGINE_TRACE)
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    vec_basic get_args() const override;
};

RCP<const Basic> trace(const RCP<const MatrixExpr> &arg);
} // namespace SymEngine

#endif
