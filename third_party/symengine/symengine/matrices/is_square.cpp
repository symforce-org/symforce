#include <symengine/basic.h>
#include <symengine/assumptions.h>
#include <symengine/visitor.h>
#include <symengine/test_visitors.h>

namespace SymEngine
{

class MatrixSquareVisitor : public BaseVisitor<MatrixSquareVisitor>
{
private:
    tribool is_square_;
    const Assumptions *assumptions_;

    void check_vector(const vec_basic &vec)
    {
        for (auto &elt : vec) {
            elt->accept(*this);
            if (not is_indeterminate(is_square_)) {
                return;
            }
        }
    }

public:
    MatrixSquareVisitor(const Assumptions *assumptions)
        : assumptions_(assumptions)
    {
    }

    void bvisit(const Basic &x){};
    void bvisit(const MatrixExpr &x)
    {
        is_square_ = tribool::indeterminate;
    }

    void bvisit(const IdentityMatrix &x)
    {
        is_square_ = tribool::tritrue;
    }

    void bvisit(const ZeroMatrix &x)
    {
        auto diff = sub(x.nrows(), x.ncols());
        is_square_ = is_zero(*diff, assumptions_);
    }

    void bvisit(const DiagonalMatrix &x)
    {
        is_square_ = tribool::tritrue;
    }

    void bvisit(const ImmutableDenseMatrix &x)
    {
        if (x.nrows() == x.ncols()) {
            is_square_ = tribool::tritrue;
        } else {
            is_square_ = tribool::trifalse;
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
        return is_square_;
    }
};

tribool is_square(const MatrixExpr &m, const Assumptions *assumptions)
{
    MatrixSquareVisitor visitor(assumptions);
    return visitor.apply(m);
}

} // namespace SymEngine
