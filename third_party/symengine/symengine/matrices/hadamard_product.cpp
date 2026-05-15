#include <symengine/mul.h>
#include <symengine/matrices/hadamard_product.h>
#include <symengine/matrices/zero_matrix.h>
#include <symengine/matrices/diagonal_matrix.h>
#include <symengine/matrices/immutable_dense_matrix.h>
#include <symengine/matrices/identity_matrix.h>

namespace SymEngine
{

void check_matching_sizes(const vec_basic &vec);

hash_t HadamardProduct::__hash__() const
{
    hash_t seed = SYMENGINE_HADAMARDPRODUCT;
    for (const auto &a : factors_) {
        hash_combine<Basic>(seed, *a);
    }
    return seed;
}

bool HadamardProduct::__eq__(const Basic &o) const
{
    if (is_a<HadamardProduct>(o)) {
        const HadamardProduct &other = down_cast<const HadamardProduct &>(o);
        return unified_eq(factors_, other.factors_);
    }
    return false;
}

int HadamardProduct::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<HadamardProduct>(o));
    const HadamardProduct &other = down_cast<const HadamardProduct &>(o);
    return unified_compare(factors_, other.factors_);
}

bool HadamardProduct::is_canonical(const vec_basic &factors) const
{
    if (factors.size() < 2) {
        return false;
    }
    size_t num_diag = 0;
    size_t num_dense = 0;
    size_t num_ident = 0;
    for (auto factor : factors) {
        if (is_a<ZeroMatrix>(*factor) || is_a<HadamardProduct>(*factor)) {
            return false;
        } else if (is_a<DiagonalMatrix>(*factor)) {
            num_diag++;
        } else if (is_a<ImmutableDenseMatrix>(*factor)) {
            num_dense++;
        } else if (is_a<IdentityMatrix>(*factor)) {
            num_ident++;
        }
    }
    if (num_diag > 1 || num_ident > 1 || num_dense > 1) {
        return false;
    }
    if (num_diag == 1 && num_dense == 1) {
        return false;
    }
    return true;
}

RCP<const MatrixExpr> hadamard_product(const vec_basic &factors)
{
    if (factors.size() == 0) {
        throw DomainError("Empty hadamard product");
    }
    if (factors.size() == 1) {
        return rcp_static_cast<const MatrixExpr>(factors[0]);
    }
    // extract nested HadamardProduct
    vec_basic expanded;
    for (auto &factor : factors) {
        if (is_a<const HadamardProduct>(*factor)) {
            auto container
                = down_cast<const HadamardProduct &>(*factor).get_factors();
            expanded.insert(expanded.end(), container.begin(), container.end());
        } else {
            expanded.push_back(factor);
        }
    }
    check_matching_sizes(expanded);
    vec_basic keep;
    RCP<const DiagonalMatrix> diag;
    RCP<const ImmutableDenseMatrix> dense;
    bool have_identity = false;
    for (auto &factor : expanded) {
        if (is_a<ZeroMatrix>(*factor)) {
            return rcp_static_cast<const MatrixExpr>(factor);
        } else if (is_a<IdentityMatrix>(*factor)) {
            if (!have_identity) {
                have_identity = true;
                keep.push_back(factor);
            }
        } else if (is_a<DiagonalMatrix>(*factor)) {
            if (diag.is_null()) {
                diag = rcp_static_cast<const DiagonalMatrix>(factor);
            } else {
                vec_basic container;
                for (size_t i = 0; i < diag->get_container().size(); i++) {
                    container.push_back(
                        mul(diag->get_container()[i],
                            down_cast<const DiagonalMatrix &>(*factor)
                                .get_container()[i]));
                }
                diag = make_rcp<const DiagonalMatrix>(container);
            }
        } else if (is_a<ImmutableDenseMatrix>(*factor)) {
            if (dense.is_null()) {
                dense = rcp_static_cast<const ImmutableDenseMatrix>(factor);
            } else {
                const vec_basic &vec1
                    = down_cast<const ImmutableDenseMatrix &>(*factor)
                          .get_values();
                const vec_basic &vec2 = dense->get_values();
                vec_basic product(vec1.size());
                for (size_t i = 0; i < vec1.size(); i++) {
                    product[i] = mul(vec1[i], vec2[i]);
                }
                dense = make_rcp<const ImmutableDenseMatrix>(
                    dense->nrows(), dense->ncols(), product);
            }
        } else {
            keep.push_back(factor);
        }
    }
    if (!dense.is_null()) {
        if (!diag.is_null()) {
            // Multiply diagonal with dense matrix
            vec_basic product;
            for (size_t i = 0; i < dense->nrows(); i++) {
                product.push_back(mul(dense->get(i, i), diag->get(i)));
            }
            diag = make_rcp<const DiagonalMatrix>(product);
        } else {
            keep.push_back(dense);
        }
    }
    if (!diag.is_null()) {
        keep.push_back(diag);
    }
    if (keep.size() == 1) {
        return rcp_static_cast<const MatrixExpr>(keep[0]);
    }
    return make_rcp<const HadamardProduct>(keep);
}

} // namespace SymEngine
