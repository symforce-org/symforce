#include <symengine/basic.h>
#include <symengine/assumptions.h>
#include <symengine/visitor.h>
#include <symengine/test_visitors.h>

namespace SymEngine
{

class MatrixRealVisitor : public BaseVisitor<MatrixRealVisitor>
{
private:
    tribool is_real_;
    const Assumptions *assumptions_;

public:
    MatrixRealVisitor(const Assumptions *assumptions)
        : assumptions_(assumptions)
    {
    }

    void bvisit(const Basic &x){};
    void bvisit(const MatrixExpr &x)
    {
        is_real_ = tribool::indeterminate;
    }

    void bvisit(const IdentityMatrix &x)
    {
        is_real_ = tribool::tritrue;
    }

    void bvisit(const ZeroMatrix &x)
    {
        is_real_ = tribool::tritrue;
    }

    void bvisit(const DiagonalMatrix &x)
    {
        tribool current = tribool::tritrue;
        for (auto &e : x.get_container()) {
            tribool next = is_real(*e, assumptions_);
            if (is_false(next)) {
                is_real_ = next;
                return;
            }
            current = andwk_tribool(current, next);
        }
        is_real_ = current;
    }

    void bvisit(const ImmutableDenseMatrix &x)
    {
        RealVisitor visitor(assumptions_);
        tribool cur = tribool::tritrue;
        for (auto &e : x.get_values()) {
            cur = and_tribool(cur, visitor.apply(*e));
            if (is_false(cur)) {
                is_real_ = cur;
                return;
            }
        }
        is_real_ = cur;
    }

    tribool apply(const MatrixExpr &s)
    {
        s.accept(*this);
        return is_real_;
    }
};

tribool is_real(const MatrixExpr &m, const Assumptions *assumptions)
{
    MatrixRealVisitor visitor(assumptions);
    return visitor.apply(m);
}

} // namespace SymEngine
