#include <symengine/mul.h>
#include <symengine/add.h>
#include <symengine/constants.h>
#include <symengine/matrices/matrix_mul.h>
#include <symengine/matrices/zero_matrix.h>
#include <symengine/matrices/identity_matrix.h>
#include <symengine/matrices/diagonal_matrix.h>
#include <symengine/matrices/immutable_dense_matrix.h>

namespace SymEngine
{

hash_t MatrixMul::__hash__() const
{
    hash_t seed = SYMENGINE_MATRIXMUL;
    hash_combine<Basic>(seed, *scalar_);
    for (const auto &a : factors_) {
        hash_combine<Basic>(seed, *a);
    }
    return seed;
}

bool MatrixMul::__eq__(const Basic &o) const
{
    if (is_a<MatrixMul>(o)) {
        const MatrixMul &other = down_cast<const MatrixMul &>(o);
        if (!eq(*scalar_, *other.scalar_)) {
            return false;
        }
        return unified_eq(factors_, other.factors_);
    }
    return false;
}

int MatrixMul::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<MatrixMul>(o));
    const MatrixMul &other = down_cast<const MatrixMul &>(o);
    int cmp_scalar = scalar_->compare(*other.scalar_);
    if (cmp_scalar != 0) {
        return cmp_scalar;
    }
    return unified_compare(factors_, other.factors_);
}

bool MatrixMul::is_canonical(const RCP<const Basic> &scalar,
                             const vec_basic &factors) const
{
    if (factors.size() == 0 || (factors.size() == 1 && eq(*scalar, *one))) {
        return false;
    }
    size_t num_diag = 0;
    size_t num_dense = 0;
    for (auto factor : factors) {
        if (is_a<ZeroMatrix>(*factor) || is_a<IdentityMatrix>(*factor)
            || is_a<MatrixMul>(*factor)) {
            return false;
        } else if (is_a<DiagonalMatrix>(*factor)) {
            num_diag++;
        } else if (is_a<ImmutableDenseMatrix>(*factor)) {
            num_dense++;
        } else {
            if (num_diag > 1 || num_dense > 1) {
                return false;
            }
            if (num_diag == 1 && num_dense == 1) {
                return false;
            }
            num_diag = 0;
            num_dense = 0;
        }
    }
    if (num_diag > 1 || num_dense > 1) {
        return false;
    }
    if (num_diag == 1 && num_dense == 1) {
        return false;
    }
    return true;
}

RCP<const DiagonalMatrix> mul_diag_diag(const DiagonalMatrix &A,
                                        const DiagonalMatrix &B)
{
    auto Avec = A.get_container();
    auto Bvec = B.get_container();
    vec_basic product(Avec.size());

    for (size_t i = 0; i < Avec.size(); i++) {
        product[i] = mul(Avec[i], Bvec[i]);
    }

    return make_rcp<const DiagonalMatrix>(product);
}

RCP<const ImmutableDenseMatrix> mul_dense_dense(const ImmutableDenseMatrix &A,
                                                const ImmutableDenseMatrix &B)
{
    size_t nrows = A.nrows();
    size_t ncols = B.ncols();
    auto Avec = A.get_values();
    auto Bvec = B.get_values();
    vec_basic product(nrows * ncols);

    for (size_t i = 0; i < nrows; i++) {
        for (size_t j = 0; j < ncols; j++) {
            product[i * ncols + j] = zero;
            for (size_t k = 0; k < A.ncols(); k++) {
                product[i * ncols + j]
                    = add(product[i * ncols + j],
                          mul(Avec[i * A.ncols() + k], Bvec[k * ncols + j]));
            }
        }
    }
    return make_rcp<const ImmutableDenseMatrix>(nrows, ncols, product);
}

RCP<const ImmutableDenseMatrix> mul_diag_dense(const DiagonalMatrix &A,
                                               const ImmutableDenseMatrix &B)
{
    size_t nrows = B.nrows();
    size_t ncols = B.ncols();

    vec_basic product(B.get_values());

    for (size_t i = 0; i < nrows; i++) {
        auto value = A.get_container()[i];
        for (size_t j = 0; j < ncols; j++) {
            product[i * ncols + j] = mul(product[i * ncols + j], value);
        }
    }
    return make_rcp<const ImmutableDenseMatrix>(nrows, ncols, product);
}

RCP<const ImmutableDenseMatrix> mul_dense_diag(const ImmutableDenseMatrix &A,
                                               const DiagonalMatrix &B)
{
    size_t nrows = A.nrows();
    size_t ncols = A.ncols();

    vec_basic product(A.get_values());

    for (size_t j = 0; j < ncols; j++) {
        auto value = B.get_container()[j];
        for (size_t i = 0; i < nrows; i++) {
            product[i * ncols + j] = mul(product[i * ncols + j], value);
        }
    }
    return make_rcp<const ImmutableDenseMatrix>(nrows, ncols, product);
}

void check_matching_mul_sizes(const vec_basic &vec)
{
    auto first_size = size(down_cast<const MatrixExpr &>(*vec[0]));
    for (size_t i = 1; i < vec.size(); i++) {
        auto second_size = size(down_cast<const MatrixExpr &>(*vec[i]));
        if (first_size.second.is_null() || second_size.first.is_null()) {
            first_size = second_size;
            continue;
        }
        auto diff = sub(first_size.second, second_size.first);
        tribool match = is_zero(*diff);
        if (is_false(match)) {
            throw DomainError("Matrix dimension mismatch");
        }
        first_size = second_size;
    }
}

RCP<const MatrixExpr> matrix_mul(const vec_basic &factors)
{
    if (factors.size() == 0) {
        throw DomainError("Empty product of matrices");
    }
    if (factors.size() == 1) {
        return rcp_static_cast<const MatrixExpr>(factors[0]);
    }

    // extract nested MatrixMul and scalars
    vec_basic expanded;
    RCP<const Basic> scalar = one;
    for (auto &factor : factors) {
        if (is_a<const MatrixMul>(*factor)) {
            auto container
                = down_cast<const MatrixMul &>(*factor).get_factors();
            scalar = mul(scalar,
                         down_cast<const MatrixMul &>(*factor).get_scalar());
            expanded.insert(expanded.end(), container.begin(), container.end());
        } else if (is_a_MatrixExpr(*factor)) {
            expanded.push_back(factor);
        } else {
            scalar = mul(scalar, factor);
        }
    }

    check_matching_mul_sizes(expanded);

    // Handle ZeroMatrix first
    for (auto &factor : factors) {
        if (is_a<ZeroMatrix>(*factor)) {
            return rcp_static_cast<const MatrixExpr>(factor);
        }
    }

    vec_basic keep;
    RCP<const DiagonalMatrix> diag;
    RCP<const ImmutableDenseMatrix> dense;
    RCP<const IdentityMatrix> ident;
    for (auto &factor : expanded) {
        if (is_a<IdentityMatrix>(*factor)) {
            ident = rcp_static_cast<const IdentityMatrix>(factor);
        } else if (is_a<DiagonalMatrix>(*factor)) {
            if (!diag.is_null()) {
                diag = mul_diag_diag(
                    *diag, down_cast<const DiagonalMatrix &>(*factor));
            } else if (!dense.is_null()) {
                dense = mul_dense_diag(
                    *dense, down_cast<const DiagonalMatrix &>(*factor));
            } else {
                diag = rcp_static_cast<const DiagonalMatrix>(factor);
            }
        } else if (is_a<ImmutableDenseMatrix>(*factor)) {
            if (!dense.is_null()) {
                dense = mul_dense_dense(
                    *dense, down_cast<const ImmutableDenseMatrix &>(*factor));
            } else if (!diag.is_null()) {
                dense = mul_diag_dense(
                    *diag, down_cast<const ImmutableDenseMatrix &>(*factor));
                diag.reset();
            } else {
                dense = rcp_static_cast<const ImmutableDenseMatrix>(factor);
            }
        } else {
            if (!diag.is_null()) {
                keep.push_back(diag);
                diag.reset();
            } else if (!dense.is_null()) {
                keep.push_back(dense);
                dense.reset();
            }
            keep.push_back(factor);
        }
    }
    if (!diag.is_null()) {
        keep.push_back(diag);
    } else if (!dense.is_null()) {
        keep.push_back(dense);
    }
    if (keep.size() == 1 && eq(*scalar, *one)) {
        return rcp_static_cast<const MatrixExpr>(keep[0]);
    }
    if (keep.size() == 0 && !ident.is_null()) {
        return ident;
    }
    return make_rcp<const MatrixMul>(scalar, keep);
}

} // namespace SymEngine
