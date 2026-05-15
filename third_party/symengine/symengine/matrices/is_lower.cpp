#include <symengine/basic.h>
#include <symengine/assumptions.h>
#include <symengine/visitor.h>
#include <symengine/test_visitors.h>

namespace SymEngine
{

class MatrixLowerVisitor : public BaseVisitor<MatrixLowerVisitor>
{
private:
    tribool is_lower_;
    const Assumptions *assumptions_;

public:
    MatrixLowerVisitor(const Assumptions *assumptions)
        : assumptions_(assumptions)
    {
    }

    void bvisit(const Basic &x){};
    void bvisit(const MatrixExpr &x)
    {
        is_lower_ = tribool::indeterminate;
    }

    void bvisit(const IdentityMatrix &x)
    {
        is_lower_ = tribool::tritrue;
    }

    void bvisit(const ZeroMatrix &x)
    {
        is_lower_ = is_square(x, assumptions_);
    }

    void bvisit(const DiagonalMatrix &x)
    {
        is_lower_ = tribool::tritrue;
    }

    void bvisit(const ImmutableDenseMatrix &x)
    {
        size_t nrows = x.nrows();
        size_t ncols = x.ncols();
        if (nrows != ncols) {
            is_lower_ = tribool::trifalse;
            return;
        }
        ZeroVisitor visitor(assumptions_);
        is_lower_ = tribool::tritrue;
        for (size_t i = 0; i < nrows; i++) {
            for (size_t j = i + 1; j < nrows; j++) {
                is_lower_ = and_tribool(is_lower_, visitor.apply(*x.get(i, j)));
                if (is_false(is_lower_)) {
                    return;
                }
            }
        }
    }

    void bvisit(const MatrixAdd &x)
    {
        bool found_nonlower = false;
        for (auto &elt : x.get_terms()) {
            elt->accept(*this);
            if (is_indeterminate(is_lower_)) {
                return;
            } else if (is_false(is_lower_)) {
                if (found_nonlower) {
                    return;
                } else {
                    found_nonlower = true;
                }
            }
        }
        if (found_nonlower) {
            is_lower_ = tribool::trifalse;
        } else {
            is_lower_ = tribool::tritrue;
        }
    }

    void bvisit(const HadamardProduct &x)
    {
        // lower x (lower | nolower | indeterminate) x ... = lower
        // (indet | nolower) x (indet | nocwlower) x ... = indeterminate
        for (auto &elt : x.get_factors()) {
            elt->accept(*this);
            if (is_true(is_lower_)) {
                return;
            }
        }
        is_lower_ = tribool::indeterminate;
    }

    tribool apply(const MatrixExpr &s)
    {
        s.accept(*this);
        return is_lower_;
    }
};

tribool is_lower(const MatrixExpr &m, const Assumptions *assumptions)
{
    MatrixLowerVisitor visitor(assumptions);
    return visitor.apply(m);
}

} // namespace SymEngine
