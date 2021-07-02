#ifndef SYMENGINE_TOKENIZER_H
#define SYMENGINE_TOKENIZER_H

#include <symengine/parser/parser_stype.h>

namespace SymEngine
{

class Tokenizer
{
    unsigned char *cur;
    unsigned char *mar;
    unsigned char *tok;

public:
    // Set the string to tokenize. The caller must ensure `str` will stay valid
    // as long as `lex` is being called.
    void set_string(const std::string &str);

    // Get next token. Token ID is returned as function result, the semantic
    // value is put into `yylval`.
    int lex(YYSTYPE &yylval);

    // Return the current token
    std::string token() const
    {
        return std::string((char *)tok, cur - tok);
    }
};

} // namespace SymEngine

#endif
