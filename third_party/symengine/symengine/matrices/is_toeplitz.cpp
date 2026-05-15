#include <symengine/basic.h>
#include <symengine/assumptions.h>
#include <symengine/visitor.h>
#include <symengine/test_visitors.h>

namespace SymEngine
{

class MatrixToeplitzVisitor : public BaseVisitor<MatrixToeplitzVisitor>
{
private:
    tribool is_toeplitz_;
    const Assumptions *assumptions_;

public:
    MatrixToeplitzVisitor(const Assumptions *assumptions)
        : assumptions_(assumptions)
    {
    }

    void bvisit(const Basic &x){};
    void bvisit(const MatrixExpr &x)
    {
        is_toeplitz_ = tribool::indeterminate;
    }

    void bvisit(const IdentityMatrix &x)
    {
        is_toeplitz_ = tribool::tritrue;
    }

    void bvisit(const ZeroMatrix &x)
    {
        is_toeplitz_ = tribool::tritrue;
    }

    void bvisit(const DiagonalMatrix &x)
    {
        tribool current = tribool::tritrue;
        auto vec = x.get_container();
        if (vec.size() == 1) {
            is_toeplitz_ = tribool::tritrue;
            return;
        }
        auto first = vec[0];
        for (auto it = vec.begin() + 1; it != vec.end(); ++it) {
            auto diff = sub(first, *it);
            tribool next = is_zero(*diff, assumptions_);
            if (is_false(next)) {
                is_toeplitz_ = next;
                return;
            }
            current = andwk_tribool(current, next);
        }
        is_toeplitz_ = current;
    }

    void bvisit(const ImmutableDenseMatrix &x)
    {
        size_t i_start, j_start, i, j;
        ZeroVisitor visitor(assumptions_);
        is_toeplitz_ = tribool::tritrue;
        // Loop over all diagonals
        for (size_t w = 0; w < std::max(x.nrows(), x.ncols()) - 1; w++) {
            // Loop over diagonals starting from the first row and the first
            // column
            for (size_t k = 0; k < 2; k++) {
                if (k == 0 && w <= x.ncols()) {
                    i_start = 0;
                    j_start = w;
                } else if (k == 1 && w <= x.nrows() && w != 0) {
                    i_start = w;
                    j_start = 0;
                } else {
                    continue;
                }
                auto first = x.get(i_start, j_start);
                // Loop along the diagonal
                for (i = i_start + 1, j = j_start + 1;
                     i < x.nrows() && j < x.ncols(); i++, j++) {
                    is_toeplitz_ = and_tribool(
                        is_toeplitz_, visitor.apply(*sub(first, x.get(i, j))));
                    if (is_false(is_toeplitz_)) {
                        return;
                    }
                }
            }
        }
    }

    tribool apply(const MatrixExpr &s)
    {
        s.accept(*this);
        return is_toeplitz_;
    }
};

tribool is_toeplitz(const MatrixExpr &m, const Assumptions *assumptions)
{
    MatrixToeplitzVisitor visitor(assumptions);
    return visitor.apply(m);
}

} // namespace SymEngine
