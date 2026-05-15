#ifndef SYMENGINE_SBML_TOKENIZER_H
#define SYMENGINE_SBML_TOKENIZER_H

#include <symengine/parser/tokenizer.h>
#include "sbml_parser.tab.hh"

namespace SymEngine
{

class SbmlTokenizer : public Tokenizer
{
public:
    int lex(sbml::parser::semantic_type *yylval);
};

} // namespace SymEngine

#endif
