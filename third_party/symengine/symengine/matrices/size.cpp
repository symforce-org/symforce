#include <symengine/visitor.h>
#include <symengine/test_visitors.h>
#include <symengine/matrices/size.h>

namespace SymEngine
{

class MatrixSizeVisitor : public BaseVisitor<MatrixSizeVisitor>
{
private:
    RCP<const Basic> nrows_;
    RCP<const Basic> ncols_;

    void all_same_size(const vec_basic &vec)
    {
        vec[0]->accept(*this);
        auto rows = nrows_;
        auto cols = ncols_;
        if (!rows.is_null() && !cols.is_null() && is_a<Integer>(*rows)
            && is_a<Integer>(*cols)) {
            return;
        }
        // Priority order of type of nrows and ncols:
        // 1. integer
        // 2. other expressions
        // 3. nullptr (meaning unknown)
        // Note that all elements must have known same size or indeterminate
        // size diff because of canonicalization
        for (size_t i = 1; i < vec.size(); i++) {
            vec[i]->accept(*this);
            if ((!nrows_.is_null() && is_a<Integer>(*nrows_))
                || (rows.is_null() && !nrows_.is_null())) {
                rows = nrows_;
            }
            if ((!ncols_.is_null() && is_a<Integer>(*ncols_))
                || (cols.is_null() && !ncols_.is_null())) {
                cols = ncols_;
            }
            if (!rows.is_null() && !cols.is_null() && is_a<Integer>(*rows)
                && is_a<Integer>(*cols)) {
                break;
            }
        }
        nrows_ = rows;
        ncols_ = cols;
    }

public:
    MatrixSizeVisitor() {}

    void bvisit(const Basic &x)
    {
        nrows_.reset();
        ncols_.reset();
    }

    void bvisit(const IdentityMatrix &x)
    {
        nrows_ = x.size();
        ncols_ = x.size();
    }

    void bvisit(const ZeroMatrix &x)
    {
        nrows_ = x.nrows();
        ncols_ = x.ncols();
    }

    void bvisit(const MatrixSymbol &x)
    {
        nrows_.reset();
        ncols_.reset();
    }

    void bvisit(const DiagonalMatrix &x)
    {
        nrows_ = integer(x.get_container().size());
        ncols_ = nrows_;
    }

    void bvisit(const ImmutableDenseMatrix &x)
    {
        nrows_ = integer(x.nrows());
        ncols_ = integer(x.ncols());
    }

    void bvisit(const MatrixAdd &x)
    {
        auto vec = x.get_terms();
        all_same_size(vec);
    }

    void bvisit(const HadamardProduct &x)
    {
        auto vec = x.get_factors();
        all_same_size(vec);
    }

    void bvisit(const MatrixMul &x)
    {
        auto vec = x.get_factors();
        vec[0]->accept(*this);
        auto row = nrows_;
        vec.back()->accept(*this);
        nrows_ = row;
    }

    std::pair<RCP<const Basic>, RCP<const Basic>> apply(const MatrixExpr &s)
    {
        s.accept(*this);
        return std::make_pair(nrows_, ncols_);
    }
};

std::pair<RCP<const Basic>, RCP<const Basic>> size(const MatrixExpr &m)
{
    MatrixSizeVisitor visitor;
    return visitor.apply(m);
}

} // namespace SymEngine
