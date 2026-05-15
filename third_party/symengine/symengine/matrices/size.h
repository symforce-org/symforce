#ifndef SYMENGINE_MATRICES_SIZE_H
#define SYMENGINE_MATRICES_SIZE_H

#include <symengine/basic.h>
#include <symengine/matrices/matrix_expr.h>

namespace SymEngine
{

std::pair<RCP<const Basic>, RCP<const Basic>> size(const MatrixExpr &m);

}

#endif
