#include <symengine/basic.h>
#include <symengine/assumptions.h>
#include <symengine/visitor.h>
#include <symengine/test_visitors.h>

namespace SymEngine
{

class MatrixDiagonalVisitor : public BaseVisitor<MatrixDiagonalVisitor>
{
private:
    tribool is_diagonal_;
    const Assumptions *assumptions_;

public:
    MatrixDiagonalVisitor(const Assumptions *assumptions)
        : assumptions_(assumptions)
    {
    }

    void bvisit(const Basic &x){};
    void bvisit(const MatrixExpr &x)
    {
        is_diagonal_ = tribool::indeterminate;
    }

    void bvisit(const IdentityMatrix &x)
    {
        is_diagonal_ = tribool::tritrue;
    }

    void bvisit(const ZeroMatrix &x)
    {
        is_diagonal_ = is_square(x, assumptions_);
    }

    void bvisit(const DiagonalMatrix &x)
    {
        is_diagonal_ = tribool::tritrue;
    }

    void bvisit(const ImmutableDenseMatrix &x)
    {
        if (x.nrows() != x.ncols()) {
            is_diagonal_ = tribool::trifalse;
            return;
        }
        size_t ncols = x.ncols();
        size_t offset;
        ZeroVisitor visitor(assumptions_);
        is_diagonal_ = tribool::tritrue;
        for (size_t i = 0; i < ncols; i++) {
            offset = i * ncols;
            for (size_t j = 0; j < ncols; j++) {
                if (j != i) {
                    auto &e = x.get_values()[offset];
                    is_diagonal_ = and_tribool(is_diagonal_, visitor.apply(*e));
                    if (is_false(is_diagonal_)) {
                        return;
                    }
                }
                offset++;
            }
        }
    }

    void bvisit(const MatrixAdd &x)
    {
        bool found_nondiag = false;
        for (auto &elt : x.get_terms()) {
            elt->accept(*this);
            if (is_indeterminate(is_diagonal_)) {
                return;
            } else if (is_false(is_diagonal_)) {
                if (found_nondiag) {
                    return;
                } else {
                    found_nondiag = true;
                }
            }
        }
        if (found_nondiag) {
            is_diagonal_ = tribool::trifalse;
        } else {
            is_diagonal_ = tribool::tritrue;
        }
    }

    void bvisit(const HadamardProduct &x)
    {
        // diag x (diag | nodiag | indeterminate) x ... = diag
        // (indet | nodiag) x (indet | nodiag) x ... = indeterminate
        for (auto &elt : x.get_factors()) {
            elt->accept(*this);
            if (is_true(is_diagonal_)) {
                return;
            }
        }
        is_diagonal_ = tribool::indeterminate;
    }

    tribool apply(const MatrixExpr &s)
    {
        s.accept(*this);
        return is_diagonal_;
    }
};

tribool is_diagonal(const MatrixExpr &m, const Assumptions *assumptions)
{
    MatrixDiagonalVisitor visitor(assumptions);
    return visitor.apply(m);
}

} // namespace SymEngine
