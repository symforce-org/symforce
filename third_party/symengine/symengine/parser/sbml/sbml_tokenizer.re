#include "sbml_tokenizer.h"
#include "sbml_parser.tab.hh"

namespace SymEngine
{

int SbmlTokenizer::lex(sbml::parser::semantic_type *yylval)
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
            operators = "-"|"+"|"/"|"("|")"|"*"|","|"^"|"<"|">"|"%"|"!";

            le = "<=";
            ge = ">=";
            ne = "!=";
            eqs = "==";
            and = "&&";
            or = "||";
            ident = char (char | dig)*;
            numeric = (dig*"."?dig+([eE][-+]?dig+)?) | (dig+".");

            * { throw SymEngine::ParseError("Unknown token: '"+token()+"'"); }
            end { return sbml::parser::token::yytokentype::END_OF_FILE; }
            whitespace { continue; }

            // FIXME:
            operators { return tok[0]; }
            le   { return sbml::parser::token::yytokentype::LE; }
            ge   { return sbml::parser::token::yytokentype::GE; }
            ne   { return sbml::parser::token::yytokentype::NE; }
            eqs  { return sbml::parser::token::yytokentype::EQ; }
            and  { return sbml::parser::token::yytokentype::AND; }
            or   { return sbml::parser::token::yytokentype::OR; }
            ident { yylval->emplace<std::string>() = token(); return sbml::parser::token::yytokentype::IDENTIFIER; }
            numeric { yylval->emplace<std::string>() = token(); return sbml::parser::token::yytokentype::NUMERIC; }
        */
    }
}

} // namespace SymEngine
