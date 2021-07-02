#ifndef SYMENGINE_PARSER_PARSER_H
#define SYMENGINE_PARSER_PARSER_H

#include <fstream>
#include <algorithm>
#include <memory>

#include <symengine/parser/tokenizer.h>
#include <symengine/add.h>
#include <symengine/pow.h>
#include <symengine/logic.h>

namespace SymEngine
{

/*
   To Parse (default) constructor is expensive as it creates all the maps and
   tables. If just one expression needs to be parsed, then calling
   SymEngine::parse() does the job. But if multiple expressions are to be
   parsed, then first initialize SymEngine::Parser and after that call
   SymEngine::Parser::parse() repeatedly.

   Example:

   Parser p;
   auto r = p.parse("x**2");

*/

class Parser
{
    std::string inp;

public:
    Tokenizer m_tokenizer;
    RCP<const Basic> res;

    RCP<const Basic> parse(const std::string &input, bool convert_xor = true);
    int parse();

    RCP<const Basic> functionify(const std::string &name, vec_basic &params);
    RCP<const Basic> parse_numeric(const std::string &expr);
    RCP<const Basic> parse_identifier(const std::string &expr);
    std::tuple<RCP<const Basic>, RCP<const Basic>>
    parse_implicit_mul(const std::string &expr);

private:
};

} // namespace SymEngine

#endif
