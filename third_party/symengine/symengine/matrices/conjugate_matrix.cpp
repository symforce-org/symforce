#include <symengine/basic.h>
#include <symengine/matrices/matrix_expr.h>
#include <symengine/matrices/conjugate_matrix.h>
#include <symengine/visitor.h>

namespace SymEngine
{

hash_t ConjugateMatrix::__hash__() const
{
    hash_t seed = SYMENGINE_CONJUGATEMATRIX;
    hash_combine<Basic>(seed, *arg_);
    return seed;
}

bool ConjugateMatrix::__eq__(const Basic &o) const
{
    return (is_a<ConjugateMatrix>(o)
            && arg_->__eq__(*down_cast<const ConjugateMatrix &>(o).arg_));
}

bool ConjugateMatrix::is_canonical(const RCP<const MatrixExpr> &arg) const
{
    // NOTE: For conjugate transpose always have the conjugate operation first
    // i.e. transpose(conjugate(A))
    if (is_a<IdentityMatrix>(*arg) || is_a<ZeroMatrix>(*arg)
        || is_a<DiagonalMatrix>(*arg) || is_a<ImmutableDenseMatrix>(*arg)
        || is_a<ConjugateMatrix>(*arg) || is_a<Transpose>(*arg)
        || is_a<MatrixAdd>(*arg) || is_a<HadamardProduct>(*arg)) {
        return false;
    }
    return true;
}

int ConjugateMatrix::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<ConjugateMatrix>(o));

    return arg_->compare(*down_cast<const ConjugateMatrix &>(o).arg_);
}

vec_basic ConjugateMatrix::get_args() const
{
    return {arg_};
}

class ConjugateMatrixVisitor : public BaseVisitor<ConjugateMatrixVisitor>
{
private:
    RCP<const MatrixExpr> conjugate_;

public:
    ConjugateMatrixVisitor() {}

    void bvisit(const Basic &x){};

    void bvisit(const MatrixExpr &x)
    {
        auto arg = rcp_static_cast<const MatrixExpr>(x.rcp_from_this());
        conjugate_ = make_rcp<const ConjugateMatrix>(arg);
    }

    void bvisit(const IdentityMatrix &x)
    {
        conjugate_ = rcp_static_cast<const MatrixExpr>(x.rcp_from_this());
    }

    void bvisit(const ZeroMatrix &x)
    {
        conjugate_ = rcp_static_cast<const MatrixExpr>(x.rcp_from_this());
    }

    void bvisit(const DiagonalMatrix &x)
    {
        auto diag = x.get_container();
        vec_basic conj(diag.size());
        for (size_t i = 0; i < diag.size(); i++) {
            conj[i] = conjugate(diag[i]);
        }
        conjugate_ = make_rcp<const DiagonalMatrix>(conj);
    }

    void bvisit(const ImmutableDenseMatrix &x)
    {
        auto values = x.get_values();
        vec_basic conj(values.size());
        for (size_t i = 0; i < values.size(); i++) {
            conj[i] = conjugate(values[i]);
        }
        conjugate_
            = make_rcp<const ImmutableDenseMatrix>(x.nrows(), x.ncols(), conj);
    }

    void bvisit(const ConjugateMatrix &x)
    {
        conjugate_ = x.get_arg();
    }

    void bvisit(const Transpose &x)
    {
        // Shift order to transpose(conj(A))
        auto arg = x.get_arg();
        auto conj = make_rcp<const ConjugateMatrix>(arg);
        conjugate_ = make_rcp<const Transpose>(conj);
    }

    void bvisit(const MatrixAdd &x)
    {
        vec_basic conj;
        for (auto &e : x.get_terms()) {
            e->accept(*this);
            conj.push_back(conjugate_);
        }
        conjugate_ = make_rcp<const MatrixAdd>(conj);
    }

    void bvisit(const HadamardProduct &x)
    {
        vec_basic conj;
        for (auto &e : x.get_factors()) {
            e->accept(*this);
            conj.push_back(conjugate_);
        }
        conjugate_ = make_rcp<const HadamardProduct>(conj);
    }

    RCP<const MatrixExpr> apply(const MatrixExpr &s)
    {
        s.accept(*this);
        return conjugate_;
    }
};

RCP<const MatrixExpr> conjugate_matrix(const RCP<const MatrixExpr> &arg)
{
    ConjugateMatrixVisitor visitor;
    return visitor.apply(*arg);
}
} // namespace SymEngine
