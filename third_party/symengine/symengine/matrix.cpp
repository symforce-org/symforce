#include <symengine/matrix.h>

namespace SymEngine
{

std::string MatrixBase::__str__() const
{
    std::ostringstream o;

    // Check for uninitialized values.
    if (nrows() > 0 && this->get(0, 0).is_null()) {
        o << "<Matrix(" << nrows() << ", " << ncols() << ") uninitialized>";
        return o.str();
    }

    for (unsigned i = 0; i < nrows(); i++) {
        o << "[";
        for (unsigned j = 0; j < ncols() - 1; j++) {
            o << *this->get(i, j) << ", ";
        }
        o << *this->get(i, ncols() - 1) << "]" << std::endl;
    }

    return o.str();
}

bool MatrixBase::eq(const MatrixBase &other) const
{
    if (this->nrows() != other.nrows() or this->ncols() != other.ncols())
        return false;

    for (unsigned i = 0; i < this->nrows(); i++)
        for (unsigned j = 0; j < this->ncols(); j++)
            if (neq(*this->get(i, j), *(other.get(i, j))))
                return false;

    return true;
}

hash_t MatrixBase::hash() const
{
    // Create a hash as the shape plus contents.
    // This is similar to what is done by sympy ImmutableDenseMatrix.
    // Potentially wasteful for the sparse matrix subclass.
    // TODO(hayk): No typeid, Matrix is not a Basic.
    hash_t seed = 999;
    hash_combine<unsigned>(seed, nrows());
    hash_combine<unsigned>(seed, ncols());
    for (unsigned i = 0; i < this->nrows(); i++) {
        for (unsigned j = 0; j < this->nrows(); j++) {
            hash_combine<Basic>(seed, *get(i, j));
        }
    }
    return seed;
}

} // SymEngine
