#ifndef SYMENGINE_MATRICES_MATRIX_ADD_H
#define SYMENGINE_MATRICES_MATRIX_ADD_H

#include <symengine/basic.h>
#include <symengine/matrices/matrix_expr.h>
#include <symengine/matrices/size.h>

namespace SymEngine
{

class MatrixAdd : public MatrixExpr
{
private:
    vec_basic terms_;

public:
    MatrixAdd(const vec_basic &terms) : terms_(terms)
    {
        SYMENGINE_ASSIGN_TYPEID();
        SYMENGINE_ASSERT(is_canonical(terms));
    }

    IMPLEMENT_TYPEID(SYMENGINE_MATRIXADD)
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    bool is_canonical(const vec_basic &terms) const;
    vec_basic get_args() const override
    {
        return vec_basic(terms_.begin(), terms_.end());
    }
    inline const vec_basic &get_terms() const
    {
        return terms_;
    }
};

RCP<const MatrixExpr> matrix_add(const vec_basic &terms);

} // namespace SymEngine

#endif
