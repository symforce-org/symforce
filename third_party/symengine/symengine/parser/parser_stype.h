#ifndef SYMENGINE_PARSER_STYPE_H
#define SYMENGINE_PARSER_STYPE_H

#include <string>
#include "symengine/basic.h"

namespace SymEngine
{

struct YYSTYPE {
    SymEngine::RCP<const SymEngine::Basic> basic;
    SymEngine::vec_basic basic_vec;
    std::string string;
    // Constructor
    YYSTYPE() = default;
    // Destructor
    ~YYSTYPE() = default;
    // Copy constructor and assignment
    YYSTYPE(const YYSTYPE &) = default;
    YYSTYPE &operator=(const YYSTYPE &) = default;
    // Move constructor and assignment
    YYSTYPE(YYSTYPE &&) = default;
    YYSTYPE &operator=(YYSTYPE &&) = default;
};

} // namespace SymEngine

#endif
