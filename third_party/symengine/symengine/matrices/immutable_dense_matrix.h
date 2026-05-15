#ifndef SYMENGINE_MATRICES_IMMUTABLE_DENSE_MATRIX_H
#define SYMENGINE_MATRICES_IMMUTABLE_DENSE_MATRIX_H

#include <symengine/basic.h>
#include <symengine/matrices/matrix_expr.h>
#include <symengine/integer.h>

namespace SymEngine
{

class ImmutableDenseMatrix : public MatrixExpr
{
private:
    size_t m_, n_;
    vec_basic values_;

public:
    ImmutableDenseMatrix(size_t m, size_t n, const vec_basic &values)
        : m_(m), n_(n), values_(values)
    {
        SYMENGINE_ASSIGN_TYPEID();
        SYMENGINE_ASSERT(is_canonical(m, n, values));
    }

    IMPLEMENT_TYPEID(SYMENGINE_IMMUTABLEDENSEMATRIX)
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    bool is_canonical(size_t m, size_t n, const vec_basic &values) const;

    vec_basic get_args() const override
    {
        vec_basic args = vec_basic({integer(m_), integer(n_)});
        args.insert(args.begin(), values_.begin(), values_.end());
        return args;
    }

    inline RCP<const Basic> get(size_t i, size_t j) const
    {
        return values_[i * n_ + j];
    }

    inline size_t nrows() const
    {
        return m_;
    }

    inline size_t ncols() const
    {
        return n_;
    }

    inline const vec_basic &get_values() const
    {
        return values_;
    }
};

RCP<const MatrixExpr> immutable_dense_matrix(size_t m, size_t n,
                                             const vec_basic &container);

} // namespace SymEngine

#endif
