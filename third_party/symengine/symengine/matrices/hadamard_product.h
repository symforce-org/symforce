#ifndef SYMENGINE_MATRICES_HADAMARD_PRODUCT_H
#define SYMENGINE_MATRICES_HADAMARD_PRODUCT_H

#include <symengine/basic.h>
#include <symengine/matrices/matrix_expr.h>

namespace SymEngine
{

class HadamardProduct : public MatrixExpr
{
private:
    vec_basic factors_;

public:
    HadamardProduct(const vec_basic &factors) : factors_(factors)
    {
        SYMENGINE_ASSIGN_TYPEID();
        SYMENGINE_ASSERT(is_canonical(factors));
    }

    IMPLEMENT_TYPEID(SYMENGINE_HADAMARDPRODUCT)
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    bool is_canonical(const vec_basic &factors) const;
    vec_basic get_args() const override
    {
        return vec_basic(factors_.begin(), factors_.end());
    }
    inline const vec_basic &get_factors() const
    {
        return factors_;
    }
};

RCP<const MatrixExpr> hadamard_product(const vec_basic &factors);

} // namespace SymEngine

#endif
