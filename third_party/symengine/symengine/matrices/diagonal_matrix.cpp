#include <symengine/matrices/diagonal_matrix.h>
#include <symengine/matrices/zero_matrix.h>
#include <symengine/matrices/identity_matrix.h>
#include <symengine/integer.h>

namespace SymEngine
{

hash_t DiagonalMatrix::__hash__() const
{
    hash_t seed = SYMENGINE_DIAGONALMATRIX;
    for (const auto &a : diag_)
        hash_combine<Basic>(seed, *a);
    return seed;
}

bool DiagonalMatrix::__eq__(const Basic &o) const
{
    if (is_a<DiagonalMatrix>(o)) {
        const DiagonalMatrix &other = down_cast<const DiagonalMatrix &>(o);
        return unified_eq(diag_, other.diag_);
    }
    return false;
}

int DiagonalMatrix::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<DiagonalMatrix>(o));
    const DiagonalMatrix &other = down_cast<const DiagonalMatrix &>(o);
    return unified_compare(diag_, other.diag_);
}

bool is_zero_vec(const vec_basic &container)
{
    for (auto &e : container) {
        if (!(is_a<Integer>(*e) && down_cast<const Integer &>(*e).is_zero())) {
            return false;
        }
    }
    return true;
}

bool is_identity_vec(const vec_basic &container)
{
    for (auto &e : container) {
        if (!(is_a<Integer>(*e) && down_cast<const Integer &>(*e).is_one())) {
            return false;
        }
    }
    return true;
}

bool DiagonalMatrix::is_canonical(const vec_basic &container) const
{
    if (container.size() == 0) {
        return false;
    }
    if (is_zero_vec(container)) {
        return false;
    }
    if (is_identity_vec(container)) {
        return false;
    }
    return true;
}

RCP<const MatrixExpr> diagonal_matrix(const vec_basic &container)
{
    if (is_zero_vec(container)) {
        return make_rcp<const ZeroMatrix>(integer(container.size()),
                                          integer(container.size()));
    } else if (is_identity_vec(container)) {
        return make_rcp<const IdentityMatrix>(integer(container.size()));
    } else {
        return make_rcp<const DiagonalMatrix>(container);
    }
}

} // namespace SymEngine
