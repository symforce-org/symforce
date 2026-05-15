#include <symengine/basic.h>
#include <symengine/matrices/matrix_expr.h>
#include <symengine/matrices/transpose.h>
#include <symengine/visitor.h>

namespace SymEngine
{

hash_t Transpose::__hash__() const
{
    hash_t seed = SYMENGINE_TRANSPOSE;
    hash_combine<Basic>(seed, *arg_);
    return seed;
}

bool Transpose::__eq__(const Basic &o) const
{
    return (is_a<Transpose>(o)
            && arg_->__eq__(*down_cast<const Transpose &>(o).arg_));
}

bool Transpose::is_canonical(const RCP<const MatrixExpr> &arg) const
{
    if (is_a<IdentityMatrix>(*arg) || is_a<ZeroMatrix>(*arg)
        || is_a<DiagonalMatrix>(*arg) || is_a<ImmutableDenseMatrix>(*arg)
        || is_a<Transpose>(*arg) || is_a<MatrixAdd>(*arg)
        || is_a<HadamardProduct>(*arg)) {
        return false;
    }
    return true;
}

int Transpose::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<Transpose>(o));

    return arg_->compare(*down_cast<const Transpose &>(o).arg_);
}

vec_basic Transpose::get_args() const
{
    return {arg_};
}

class TransposeVisitor : public BaseVisitor<TransposeVisitor>
{
private:
    RCP<const MatrixExpr> transpose_;

public:
    TransposeVisitor() {}

    void bvisit(const Basic &x){};

    void bvisit(const MatrixExpr &x)
    {
        auto arg = rcp_static_cast<const MatrixExpr>(x.rcp_from_this());
        transpose_ = make_rcp<const Transpose>(arg);
    }

    void bvisit(const IdentityMatrix &x)
    {
        transpose_ = rcp_static_cast<const MatrixExpr>(x.rcp_from_this());
    }

    void bvisit(const ZeroMatrix &x)
    {
        transpose_ = make_rcp<const ZeroMatrix>(x.ncols(), x.nrows());
    }

    void bvisit(const DiagonalMatrix &x)
    {
        transpose_ = rcp_static_cast<const MatrixExpr>(x.rcp_from_this());
    }

    void bvisit(const ImmutableDenseMatrix &x)
    {
        auto values = x.get_values();
        vec_basic t(values.size());

        for (size_t i = 0; i < x.nrows(); i++)
            for (size_t j = 0; j < x.ncols(); j++)
                t[j * x.ncols() + i] = x.get(i, j);

        transpose_
            = make_rcp<const ImmutableDenseMatrix>(x.ncols(), x.nrows(), t);
    }

    void bvisit(const Transpose &x)
    {
        transpose_ = x.get_arg();
    }

    void bvisit(const MatrixAdd &x)
    {
        vec_basic t;
        for (auto &e : x.get_terms()) {
            e->accept(*this);
            t.push_back(transpose_);
        }
        transpose_ = make_rcp<const MatrixAdd>(t);
    }

    void bvisit(const HadamardProduct &x)
    {
        vec_basic t;
        for (auto &e : x.get_factors()) {
            e->accept(*this);
            t.push_back(transpose_);
        }
        transpose_ = make_rcp<const HadamardProduct>(t);
    }

    RCP<const MatrixExpr> apply(const MatrixExpr &s)
    {
        s.accept(*this);
        return transpose_;
    }
};

RCP<const MatrixExpr> transpose(const RCP<const MatrixExpr> &arg)
{
    TransposeVisitor visitor;
    return visitor.apply(*arg);
}
} // namespace SymEngine
