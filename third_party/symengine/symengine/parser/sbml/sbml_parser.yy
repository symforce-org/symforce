%require "3.2"
%language "c++"
%define api.value.type variant
%define api.namespace {sbml}
%param {SymEngine::SbmlParser &p}

%code requires // *.h
{

#include "symengine/parser/sbml/sbml_parser.h"

}

%code // *.cpp
{
#include "symengine/pow.h"
#include "symengine/logic.h"
#include "symengine/parser/sbml/sbml_parser.h"

using SymEngine::add;
using SymEngine::Basic;
using SymEngine::Boolean;
using SymEngine::Eq;
using SymEngine::Ge;
using SymEngine::Gt;
using SymEngine::Le;
using SymEngine::Lt;
using SymEngine::mul;
using SymEngine::Ne;
using SymEngine::one;
using SymEngine::pow;
using SymEngine::RCP;
using SymEngine::rcp_static_cast;
using SymEngine::set_boolean;
using SymEngine::sub;
using SymEngine::vec_basic;
using SymEngine::vec_boolean;

#include "symengine/parser/sbml/sbml_tokenizer.h"

namespace sbml
{

int yylex(sbml::parser::semantic_type* yylval, SymEngine::SbmlParser & p)
{
    return p.m_tokenizer->lex(yylval);
}

void parser::error(const std::string &msg)
{
    throw SymEngine::ParseError(msg);
}

}

}

%token<std::string> IDENTIFIER
%token<std::string> NUMERIC
%token END_OF_FILE 0

%left AND OR
%left EQ '<' '>' LE GE NE
%left '+' '-'
%left '*' '/' '%'
%right UMINUS UPLUS '!'
%left '^'
%nonassoc '('

%type<SymEngine::RCP<const SymEngine::Basic>> st_expr
%type<SymEngine::RCP<const SymEngine::Basic>> expr
%type<SymEngine::vec_basic> expr_list

%start st_expr

%%
st_expr
    : expr { $$ = $1; p.res = $$; }
    ;

expr
    : expr '+' expr { $$ = add($1, $3); }
    | expr '-' expr { $$ = sub($1, $3); }
    | expr '*' expr { $$ = mul($1, $3); }
    | expr '/' expr { $$ = div($1, $3); }
    | expr '%' expr { $$ = p.modulo($1, $3); }
    | expr '^' expr { $$ = pow($1, $3); }
    | expr '<' expr { $$ = Lt($1, $3); }
    | expr '>' expr { $$ = Gt($1, $3); }
    | expr NE expr { $$ = Ne($1, $3); }
    | expr LE expr { $$ = Le($1, $3); }
    | expr GE expr { $$ = Ge($1, $3); }
    | expr EQ expr { $$ = Eq($1, $3); }
    | expr OR expr {
            set_boolean s;
            s.insert(rcp_static_cast<const Boolean>($1));
            s.insert(rcp_static_cast<const Boolean>($3));
            $$ = logical_or(s); }
    | expr AND expr {
            set_boolean s;
            s.insert(rcp_static_cast<const Boolean>($1));
            s.insert(rcp_static_cast<const Boolean>($3));
            $$ = logical_and(s); }
    | '(' expr ')' { $$ = $2; }
    | '-' expr %prec UMINUS { $$ = neg($2); }
    | '+' expr %prec UPLUS { $$ = $2; }
    | '!' expr {
            $$ = logical_not(rcp_static_cast<const Boolean>($2)); }
    | IDENTIFIER { $$ = p.parse_identifier($1); }
    | NUMERIC { $$ = p.parse_numeric($1); }
    | IDENTIFIER '(' expr_list ')' { $$ = p.functionify($1, $3); }
    | IDENTIFIER '(' ')' { $$ = p.functionify($1); }
    ;

expr_list
    : expr_list ',' expr { $$ = $1; $$.push_back($3); }
    | expr { $$ = vec_basic(1, $1); }
    ;
