#include <symengine/matrices/immutable_dense_matrix.h>
#include <symengine/matrices/zero_matrix.h>
#include <symengine/matrices/identity_matrix.h>
#include <symengine/matrices/diagonal_matrix.h>

namespace SymEngine
{

hash_t ImmutableDenseMatrix::__hash__() const
{
    hash_t seed = SYMENGINE_IMMUTABLEDENSEMATRIX;
    hash_combine(seed, m_);
    hash_combine(seed, n_);
    for (const auto &a : values_) {
        hash_combine<Basic>(seed, *a);
    }
    return seed;
}

bool ImmutableDenseMatrix::__eq__(const Basic &o) const
{
    if (is_a<ImmutableDenseMatrix>(o)) {
        const ImmutableDenseMatrix &other
            = down_cast<const ImmutableDenseMatrix &>(o);
        if (m_ != other.m_ || n_ != other.n_) {
            return false;
        }
        return unified_eq(values_, other.values_);
    }
    return false;
}

int ImmutableDenseMatrix::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<ImmutableDenseMatrix>(o));
    const ImmutableDenseMatrix &other
        = down_cast<const ImmutableDenseMatrix &>(o);
    if (m_ < other.m_) {
        return -1;
    } else if (m_ > other.m_) {
        return 1;
    }
    if (n_ < other.n_) {
        return -1;
    } else if (n_ > other.n_) {
        return 1;
    }
    return unified_compare(values_, other.values_);
}

bool is_identity_dense(size_t n, const vec_basic &container)
{
    size_t i = 0;
    for (size_t row = 0; row < n; row++) {
        for (size_t col = 0; col < n; col++) {
            auto &e = container[i];
            if (col == row) {
                if (!(is_a<Integer>(*e)
                      && down_cast<const Integer &>(*e).is_one())) {
                    return false;
                }
            } else {
                if (!(is_a<Integer>(*e)
                      && down_cast<const Integer &>(*e).is_zero())) {
                    return false;
                }
            }
            i++;
        }
    }
    return true;
}

bool is_diagonal_dense(size_t n, const vec_basic &container)
{
    size_t i = 0;
    for (size_t row = 0; row < n; row++) {
        for (size_t col = 0; col < n; col++) {
            if (col != row) {
                auto &e = container[i];
                if (!(is_a<Integer>(*e)
                      && down_cast<const Integer &>(*e).is_zero())) {
                    return false;
                }
            }
            i++;
        }
    }
    return true;
}

vec_basic extract_diagonal(size_t n, const vec_basic &container)
{
    vec_basic keep;
    size_t i = 0;
    for (size_t row = 0; row < n; row++) {
        for (size_t col = 0; col < n; col++) {
            if (col == row) {
                auto &e = container[i];
                keep.push_back(e);
            }
            i++;
        }
    }
    return keep;
}

bool ImmutableDenseMatrix::is_canonical(size_t m, size_t n,
                                        const vec_basic &values) const
{
    if (m < 1 || n < 1 || values.size() == 0) {
        return false;
    }
    if (m * n != values.size()) {
        return false;
    }
    if (is_zero_vec(values)) {
        return false;
    }
    if (m == n && is_identity_dense(m, values)) {
        return false;
    }
    if (m == n && is_diagonal_dense(m, values)) {
        return false;
    }
    return true;
}

RCP<const MatrixExpr> immutable_dense_matrix(size_t m, size_t n,
                                             const vec_basic &container)
{
    if (is_zero_vec(container)) {
        return make_rcp<const ZeroMatrix>(integer(m), integer(n));
    } else if (m == n && is_identity_dense(m, container)) {
        return make_rcp<const IdentityMatrix>(integer(m));
    } else if (m == n && is_diagonal_dense(m, container)) {
        return make_rcp<const DiagonalMatrix>(extract_diagonal(m, container));
    } else {
        return make_rcp<const ImmutableDenseMatrix>(m, n, container);
    }
}

} // namespace SymEngine
