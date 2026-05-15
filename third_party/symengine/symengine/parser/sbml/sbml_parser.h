#ifndef SYMENGINE_PARSER_SBML_PARSER_H
#define SYMENGINE_PARSER_SBML_PARSER_H

#include <fstream>
#include <algorithm>
#include <memory>

#include <symengine/parser/parser.h>
#include <symengine/add.h>
#include <symengine/pow.h>
#include <symengine/logic.h>

namespace SymEngine
{

class SbmlTokenizer;

class SbmlParser : public Parser
{
public:
    std::unique_ptr<SbmlTokenizer> m_tokenizer;
    RCP<const Basic> parse(const std::string &input);

    RCP<const Basic> modulo(const RCP<const Basic> &a,
                            const RCP<const Basic> &b);
    RCP<const Basic> functionify(const std::string &name);
    RCP<const Basic> functionify(const std::string &name, vec_basic &params);
    RCP<const Basic> parse_identifier(const std::string &expr);
    explicit SbmlParser(const std::map<const std::string,
                                       const RCP<const Basic>> &parser_constants
                        = {});
    ~SbmlParser();
};

} // namespace SymEngine

#endif
