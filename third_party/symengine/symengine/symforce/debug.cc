#include <iostream>

#include <symengine/add.h>
#include <symengine/integer.h>
#include <symengine/mul.h>
#include <symengine/symbol.h>
#include <symengine/symforce/factor_coefs.h>

using namespace SymEngine;

int main() {
    const RCP<const Symbol> x = make_rcp<const Symbol>("x");
    const RCP<const Symbol> y = make_rcp<const Symbol>("y");

    const RCP<const Basic> expr = add(mul(integer(2), x), mul(integer(-2), y));
    std::cout << *expr << std::endl;
    const RCP<const Basic> factored = factor_coefs(expr);
    std::cout << *factored << std::endl;
    const RCP<const Basic> expr2 = add(mul(integer(3), expr), mul(integer(-3), x));
    std::cout << *expr2 << std::endl;
    const RCP<const Basic> factored2 = factor_coefs(expr2);
    std::cout << *factored2 << std::endl;

}