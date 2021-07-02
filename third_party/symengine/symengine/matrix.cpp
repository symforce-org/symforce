#include <symengine/matrix.h>

namespace SymEngine
{

std::string MatrixBase::__str__() const
{
    std::ostringstream o;

    for (unsigned i = 0; i < nrows(); i++) {
        o << "[";
        for (unsigned j = 0; j < ncols() - 1; j++)
            o << *this->get(i, j) << ", ";
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

} // SymEngine
