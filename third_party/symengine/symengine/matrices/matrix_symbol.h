#ifndef SYMENGINE_MATRICES_MATRIX_SYMBOL_H
#define SYMENGINE_MATRICES_MATRIX_SYMBOL_H

#include <symengine/basic.h>
#include <symengine/matrices/matrix_expr.h>

namespace SymEngine
{

class MatrixSymbol : public MatrixExpr
{
private:
    std::string name_;

public:
    MatrixSymbol(const std::string &name) : name_(name)
    {
        SYMENGINE_ASSIGN_TYPEID();
    }

    IMPLEMENT_TYPEID(SYMENGINE_MATRIXSYMBOL)
    hash_t __hash__() const override;
    bool __eq__(const Basic &o) const override;
    int compare(const Basic &o) const override;

    const std::string &get_name() const
    {
        return name_;
    }

    vec_basic get_args() const override
    {
        return {};
    }
};

RCP<const MatrixExpr> matrix_symbol(const std::string &name);

} // namespace SymEngine

#endif
