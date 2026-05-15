#include <symengine/matrices/matrix_symbol.h>

namespace SymEngine
{

hash_t MatrixSymbol::__hash__() const
{
    hash_t seed = SYMENGINE_MATRIXSYMBOL;
    hash_combine(seed, name_);
    return seed;
}

bool MatrixSymbol::__eq__(const Basic &o) const
{
    return (is_a<MatrixSymbol>(o)
            && name_ == down_cast<const MatrixSymbol &>(o).name_);
}

int MatrixSymbol::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<MatrixSymbol>(o));

    const MatrixSymbol &s = down_cast<const MatrixSymbol &>(o);
    if (name_ == s.name_)
        return 0;
    return name_ < s.name_ ? -1 : 1;
}

RCP<const MatrixExpr> matrix_symbol(const std::string &name)
{
    return make_rcp<const MatrixSymbol>(name);
}

} // namespace SymEngine
