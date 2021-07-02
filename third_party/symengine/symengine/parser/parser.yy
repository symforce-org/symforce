%require "3.0"
%define api.pure full
%define api.value.type {struct SymEngine::YYSTYPE}
%param {SymEngine::Parser &p}

/*
// Uncomment this to enable parser tracing:
%define parse.trace
%printer { fprintf(yyo, "%s", $$.c_str()); } <string>
%printer { std::cerr << *$$; } <basic>
*/


%code requires // *.h
{

#include "symengine/parser/parser.h"

}

%code // *.cpp
{
#include "symengine/pow.h"
#include "symengine/logic.h"
#include "symengine/parser/parser.h"

using SymEngine::RCP;
using SymEngine::Basic;
using SymEngine::vec_basic;
using SymEngine::rcp_static_cast;
using SymEngine::mul;
using SymEngine::pow;
using SymEngine::add;
using SymEngine::sub;
using SymEngine::Lt;
using SymEngine::Gt;
using SymEngine::Le;
using SymEngine::Ge;
using SymEngine::Eq;
using SymEngine::set_boolean;
using SymEngine::Boolean;
using SymEngine::one;
using SymEngine::vec_boolean;


#include "symengine/parser/tokenizer.h"


int yylex(SymEngine::YYSTYPE *yylval, SymEngine::Parser &p)
{
    return p.m_tokenizer.lex(*yylval);
} // ylex

void yyerror(SymEngine::Parser &p, const std::string &msg)
{
    throw SymEngine::ParseError(msg);
}

// Force YYCOPY to not use memcopy, but rather copy the structs one by one,
// which will cause C++ to call the proper copy constructors.
# define YYCOPY(Dst, Src, Count)              \
    do                                        \
      {                                       \
        YYSIZE_T yyi;                         \
        for (yyi = 0; yyi < (Count); yyi++)   \
          (Dst)[yyi] = (Src)[yyi];            \
      }                                       \
    while (0)

} // code



%token <string> IDENTIFIER
%token <string> NUMERIC
%token <string> IMPLICIT_MUL
%token END_OF_FILE 0

%left '|'
%left '^'
%left '&'
%left EQ
%left '>'
%left '<'
%left LE
%left GE
%left '-' '+'
%left '*' '/'
%right UMINUS
%right POW
%right NOT
%nonassoc '('

%type <basic> st_expr
%type <basic> expr
%type <basic_vec> expr_list
%type <basic> leaf
%type <basic> func

%start st_expr

%%
st_expr :
    expr
    {
        $$ = $1;
        p.res = $$;
    }
;

expr:
        expr '+' expr
        { $$ = add($1, $3); }
|
        expr '-' expr
        { $$ = sub($1, $3); }
|
        expr '*' expr
        { $$ = mul($1, $3); }
|
        expr '/' expr
        { $$ = div($1, $3); }
|
// FIXME: This rule generates:
// parser.yy: warning: 1 shift/reduce conflict [-Wconflicts-sr]
        IMPLICIT_MUL POW expr
        {
          auto tup = p.parse_implicit_mul($1);
          if (neq(*std::get<1>(tup), *one)) {
            $$ = mul(std::get<0>(tup), pow(std::get<1>(tup), $3));
          } else {
            $$ = pow(std::get<0>(tup), $3);
          }
        }
|
        expr POW expr
        { $$ = pow($1, $3); }
|
        expr '<' expr
        { $$ = rcp_static_cast<const Basic>(Lt($1, $3)); }
|
        expr '>' expr
        { $$ = rcp_static_cast<const Basic>(Gt($1, $3)); }
|
        expr LE expr
        { $$ = rcp_static_cast<const Basic>(Le($1, $3)); }
|
        expr GE expr
        { $$ = rcp_static_cast<const Basic>(Ge($1, $3)); }
|
        expr EQ expr
        { $$ = rcp_static_cast<const Basic>(Eq($1, $3)); }
|
        expr '|' expr
        {
            set_boolean s;
            s.insert(rcp_static_cast<const Boolean>($1));
            s.insert(rcp_static_cast<const Boolean>($3));
            $$ = rcp_static_cast<const Basic>(logical_or(s));
        }
|
        expr '&' expr
        {
            set_boolean s;
            s.insert(rcp_static_cast<const Boolean>($1));
            s.insert(rcp_static_cast<const Boolean>($3));
            $$ = rcp_static_cast<const Basic>(logical_and(s));
        }
|
        expr '^' expr
        {
            vec_boolean s;
            s.push_back(rcp_static_cast<const Boolean>($1));
            s.push_back(rcp_static_cast<const Boolean>($3));
            $$ = rcp_static_cast<const Basic>(logical_xor(s));
        }
|
        '(' expr ')'
        { $$ = $2; }
|
        '-' expr %prec UMINUS
        { $$ = neg($2); }
|
        '~' expr %prec NOT
        { $$ = rcp_static_cast<const Basic>(logical_not(rcp_static_cast<const Boolean>($2))); }
|
        leaf
        { $$ = rcp_static_cast<const Basic>($1); }
;

leaf:
    IDENTIFIER
    {
        $$ = p.parse_identifier($1);
    }
|
    IMPLICIT_MUL
    {
        auto tup = p.parse_implicit_mul($1);
        $$ = mul(std::get<0>(tup), std::get<1>(tup));
    }
|
    NUMERIC
    {
        $$ = p.parse_numeric($1);
    }
|
    func
    {
        $$ = $1;
    }
;

func:
    IDENTIFIER '(' expr_list ')'
    {
        $$ = p.functionify($1, $3);
    }
;

expr_list:

    expr_list ',' expr
    {
        $$ = $1; // TODO : should make copy?
        $$ .push_back($3);
    }
|
    expr
    {
        $$ = vec_basic(1, $1);
    }
;
