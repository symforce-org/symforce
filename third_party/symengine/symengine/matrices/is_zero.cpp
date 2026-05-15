#include <symengine/basic.h>
#include <symengine/assumptions.h>
#include <symengine/visitor.h>
#include <symengine/test_visitors.h>

namespace SymEngine
{

class MatrixZeroVisitor : public BaseVisitor<MatrixZeroVisitor>
{
private:
    tribool is_zero_;
    const Assumptions *assumptions_;

public:
    MatrixZeroVisitor(const Assumptions *assumptions)
        : assumptions_(assumptions)
    {
    }

    void bvisit(const Basic &x){};
    void bvisit(const MatrixExpr &x)
    {
        is_zero_ = tribool::indeterminate;
    }

    void bvisit(const IdentityMatrix &x)
    {
        is_zero_ = tribool::trifalse;
    }

    void bvisit(const ZeroMatrix &x)
    {
        is_zero_ = tribool::tritrue;
    }

    void bvisit(const DiagonalMatrix &x)
    {
        tribool current = tribool::tritrue;
        for (auto &e : x.get_container()) {
            tribool next = is_zero(*e, assumptions_);
            if (is_false(next)) {
                is_zero_ = next;
                return;
            }
            current = andwk_tribool(current, next);
        }
        is_zero_ = current;
    }

    void bvisit(const ImmutableDenseMatrix &x)
    {
        ZeroVisitor visitor(assumptions_);
        is_zero_ = tribool::tritrue;
        for (auto &e : x.get_values()) {
            is_zero_ = and_tribool(is_zero_, visitor.apply(*e));
            if (is_false(is_zero_)) {
                return;
            }
        }
    }

    void bvisit(const MatrixAdd &x)
    {
        is_zero_ = tribool::indeterminate;
    }

    tribool apply(const MatrixExpr &s)
    {
        s.accept(*this);
        return is_zero_;
    }
};

tribool is_zero(const MatrixExpr &m, const Assumptions *assumptions)
{
    MatrixZeroVisitor visitor(assumptions);
    return visitor.apply(m);
}

} // namespace SymEngine
