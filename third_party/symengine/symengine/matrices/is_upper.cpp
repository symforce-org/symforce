#include <symengine/basic.h>
#include <symengine/assumptions.h>
#include <symengine/visitor.h>
#include <symengine/test_visitors.h>

namespace SymEngine
{

class MatrixUpperVisitor : public BaseVisitor<MatrixUpperVisitor>
{
private:
    tribool is_upper_;
    const Assumptions *assumptions_;

public:
    MatrixUpperVisitor(const Assumptions *assumptions)
        : assumptions_(assumptions)
    {
    }

    void bvisit(const Basic &x){};
    void bvisit(const MatrixExpr &x)
    {
        is_upper_ = tribool::indeterminate;
    }

    void bvisit(const IdentityMatrix &x)
    {
        is_upper_ = tribool::tritrue;
    }

    void bvisit(const ZeroMatrix &x)
    {
        is_upper_ = is_square(x, assumptions_);
    }

    void bvisit(const DiagonalMatrix &x)
    {
        is_upper_ = tribool::tritrue;
    }

    void bvisit(const ImmutableDenseMatrix &x)
    {
        size_t nrows = x.nrows();
        size_t ncols = x.ncols();
        if (nrows != ncols) {
            is_upper_ = tribool::trifalse;
            return;
        }
        ZeroVisitor visitor(assumptions_);
        is_upper_ = tribool::tritrue;
        for (size_t i = 1; i < nrows; i++) {
            for (size_t j = 0; j < i; j++) {
                is_upper_ = and_tribool(is_upper_, visitor.apply(*x.get(i, j)));
                if (is_false(is_upper_)) {
                    return;
                }
            }
        }
    }

    void bvisit(const MatrixAdd &x)
    {
        bool found_nonupper = false;
        for (auto &elt : x.get_terms()) {
            elt->accept(*this);
            if (is_indeterminate(is_upper_)) {
                return;
            } else if (is_false(is_upper_)) {
                if (found_nonupper) {
                    return;
                } else {
                    found_nonupper = true;
                }
            }
        }
        if (found_nonupper) {
            is_upper_ = tribool::trifalse;
        } else {
            is_upper_ = tribool::tritrue;
        }
    }

    void bvisit(const HadamardProduct &x)
    {
        for (auto &elt : x.get_factors()) {
            elt->accept(*this);
            if (is_true(is_upper_)) {
                return;
            }
        }
        is_upper_ = tribool::indeterminate;
    }

    tribool apply(const MatrixExpr &s)
    {
        s.accept(*this);
        return is_upper_;
    }
};

tribool is_upper(const MatrixExpr &m, const Assumptions *assumptions)
{
    MatrixUpperVisitor visitor(assumptions);
    return visitor.apply(m);
}

} // namespace SymEngine
