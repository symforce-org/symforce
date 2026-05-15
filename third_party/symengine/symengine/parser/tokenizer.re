#include "tokenizer.h"

#include "parser.tab.hh"

namespace SymEngine
{

void Tokenizer::set_string(const std::string &str)
{
    // The input string must be NULL terminated, otherwise the tokenizer will
    // not detect the end of string. After C++11, the std::string is guaranteed
    // to end with \0, but we check this here just in case.
    SYMENGINE_ASSERT(str[str.size()] == '\0');
    cur = (unsigned char *)(&str[0]);
}

int Tokenizer::lex(yy::parser::semantic_type* yylval)
{
    for (;;) {
        tok = cur;
        /*!re2c
            re2c:define:YYCURSOR = cur;
            re2c:define:YYMARKER = mar;
            re2c:yyfill:enable = 0;
            re2c:define:YYCTYPE = "unsigned char";

            end = "\x00";
            whitespace = [ \t\v\n\r]+;
            dig = [0-9];
            char =  [\x80-\xff] | [a-zA-Z_];
            operators = "-"|"+"|"/"|"("|")"|"*"|","|"^"|"~"|"<"|">"|"&"|"|";

            pows = "**"|"@";
            le = "<=";
            ge = ">=";
            ne = "!=";
            eqs = "==";
            ident = char (char | dig)*;
            pwise = "Piecewise";
            numeric = (dig*"."?dig+([eE][-+]?dig+)?) | (dig+".");
            implicitmul = numeric ident;

            * { throw SymEngine::ParseError("Unknown token: '"+token()+"'"); }
            end { return yy::parser::token::yytokentype::END_OF_FILE; }
            whitespace { continue; }

            // FIXME:
            operators { return tok[0]; }
            pows { return yy::parser::token::yytokentype::POW; }
            le   { return yy::parser::token::yytokentype::LE; }
            ge   { return yy::parser::token::yytokentype::GE; }
            ne   { return yy::parser::token::yytokentype::NE; }
            eqs  { return yy::parser::token::yytokentype::EQ; }
            pwise { yylval->emplace<std::string>() = token(); return yy::parser::token::yytokentype::PIECEWISE; }
            ident { yylval->emplace<std::string>() = token(); return yy::parser::token::yytokentype::IDENTIFIER; }
            numeric { yylval->emplace<std::string>() = token(); return yy::parser::token::yytokentype::NUMERIC; }
            implicitmul { yylval->emplace<std::string>() = token(); return yy::parser::token::yytokentype::IMPLICIT_MUL; }
        */
    }
}

} // namespace SymEngine
