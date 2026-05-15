#include <symengine/basic.h>
#include <symengine/assumptions.h>
#include <symengine/visitor.h>
#include <symengine/test_visitors.h>

namespace SymEngine
{

class MatrixSymmetricVisitor : public BaseVisitor<MatrixSymmetricVisitor>
{
private:
    tribool is_symmetric_;
    const Assumptions *assumptions_;

    void check_vector(const vec_basic &vec)
    {
        bool found_nonsym = false;
        for (auto &elt : vec) {
            elt->accept(*this);
            if (is_indeterminate(is_symmetric_)) {
                return;
            } else if (is_false(is_symmetric_)) {
                if (found_nonsym) {
                    return;
                } else {
                    found_nonsym = true;
                }
            }
        }
        if (found_nonsym) {
            is_symmetric_ = tribool::trifalse;
        } else {
            is_symmetric_ = tribool::tritrue;
        }
    }

public:
    MatrixSymmetricVisitor(const Assumptions *assumptions)
        : assumptions_(assumptions)
    {
    }

    void bvisit(const Basic &x){};
    void bvisit(const MatrixExpr &x)
    {
        is_symmetric_ = tribool::indeterminate;
    }

    void bvisit(const IdentityMatrix &x)
    {
        is_symmetric_ = tribool::tritrue;
    }

    void bvisit(const ZeroMatrix &x)
    {
        is_symmetric_ = is_square(x, assumptions_);
    }

    void bvisit(const DiagonalMatrix &x)
    {
        is_symmetric_ = tribool::tritrue;
    }

    void bvisit(const ImmutableDenseMatrix &x)
    {
        size_t nrows = x.nrows();
        size_t ncols = x.ncols();
        if (nrows != ncols) {
            is_symmetric_ = tribool::trifalse;
            return;
        }
        ZeroVisitor visitor(assumptions_);
        is_symmetric_ = tribool::tritrue;
        for (size_t i = 0; i < ncols; i++) {
            for (size_t j = 0; j <= i; j++) {
                if (j != i) {
                    auto e1 = x.get(i, j);
                    auto e2 = x.get(j, i);
                    is_symmetric_ = and_tribool(is_symmetric_,
                                                visitor.apply(*sub(e1, e2)));
                }
                if (is_false(is_symmetric_)) {
                    return;
                }
            }
        }
    }

    void bvisit(const MatrixAdd &x)
    {
        check_vector(x.get_terms());
    }

    void bvisit(const HadamardProduct &x)
    {
        check_vector(x.get_factors());
    }

    tribool apply(const MatrixExpr &s)
    {
        s.accept(*this);
        return is_symmetric_;
    }
};

tribool is_symmetric(const MatrixExpr &m, const Assumptions *assumptions)
{
    MatrixSymmetricVisitor visitor(assumptions);
    return visitor.apply(m);
}

} // namespace SymEngine
