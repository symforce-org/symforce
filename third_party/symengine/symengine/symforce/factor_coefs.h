#ifndef FACTOR_COEFS_H
#define FACTOR_COEFS_H

#include <symengine/basic.h>

namespace SymEngine
{
    RCP<const Basic> factor_coefs(const RCP<const Basic>& b);
}


#endif
