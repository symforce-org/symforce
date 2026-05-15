#ifndef SYMENGINE_MATRICES_ZERO_MATRIX_H
#define SYMENGINE_MATRICES_ZERO_MATRIX_H

#include <symengine/basic.h>
#include <symengine/matrices/matrix_expr.h>

namespace SymEngine
{

class ZeroMatrix : public MatrixExpr
{
private:
    RCP<const Basic> m_, n_; // m >= 0, n >= 0

public:
    ZeroMatrix(const RCP<const Basic> &m, const RCP<const Basic> &n)
        : m_(m), n_(n)
    {
        SYMENGINE_ASSIGN_TYPEID();
        SYMENGINE_ASSERT(is_canonical(m, n));
    }

    IMPLEMENT_TYPEID(SYMENGINE_ZEROMATRIX)
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;
    vec_basic get_args() const override;
    bool is_canonical(const RCP<const Basic> &m,
                      const RCP<const Basic> &n) const;

    inline const RCP<const Basic> &nrows() const
    {
        return m_;
    }

    inline const RCP<const Basic> &ncols() const
    {
        return n_;
    }
};

RCP<const MatrixExpr> zero_matrix(const RCP<const Basic> &m,
                                  const RCP<const Basic> &n);

} // namespace SymEngine
#endif
