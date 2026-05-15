#ifndef SYMENGINE_MATRICES_MATRIX_MUL_H
#define SYMENGINE_MATRICES_MATRIX_MUL_H

#include <symengine/basic.h>
#include <symengine/constants.h>
#include <symengine/matrices/matrix_expr.h>
#include <symengine/matrices/size.h>

namespace SymEngine
{

class MatrixMul : public MatrixExpr
{
private:
    RCP<const Basic> scalar_;
    vec_basic factors_;

public:
    MatrixMul(const RCP<const Basic> &scalar, const vec_basic &factors)
        : scalar_(scalar), factors_(factors)
    {
        SYMENGINE_ASSIGN_TYPEID();
        SYMENGINE_ASSERT(is_canonical(scalar, factors));
    }

    IMPLEMENT_TYPEID(SYMENGINE_MATRIXMUL)
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    bool is_canonical(const RCP<const Basic> &scalar,
                      const vec_basic &factors) const;
    vec_basic get_args() const override
    {
        vec_basic args;
        if (!eq(*scalar_, *one)) {
            args.push_back(scalar_);
        }
        args.insert(args.end(), factors_.begin(), factors_.end());
        return args;
    }
    inline const vec_basic &get_factors() const
    {
        return factors_;
    }
    inline const RCP<const Basic> &get_scalar() const
    {
        return scalar_;
    }
};

RCP<const MatrixExpr> matrix_mul(const vec_basic &factors);

} // namespace SymEngine

#endif
