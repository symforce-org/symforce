#ifndef SYMENGINE_MATRICES_MATRIXEXPR_H
#define SYMENGINE_MATRICES_MATRIXEXPR_H

namespace SymEngine
{

class MatrixExpr : public Basic
{
};

inline bool is_a_MatrixExpr(const Basic &b)
{
    return (b.get_type_code() >= SYMENGINE_IDENTITYMATRIX
            && b.get_type_code() <= SYMENGINE_TRANSPOSE);
}

} // namespace SymEngine

#endif
