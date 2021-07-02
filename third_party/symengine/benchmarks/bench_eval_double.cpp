#define NONIUS_RUNNER
#include "nonius.h++"

#include <symengine/basic.h>
#include <symengine/add.h>
#include <symengine/symbol.h>
#include <symengine/dict.h>
#include <symengine/integer.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/eval_double.h>

using SymEngine::Basic;
using SymEngine::symbol;
using SymEngine::integer;
using SymEngine::RCP;

RCP<const Basic> get_eval_double_expression()
{
    RCP<const Basic> e = sin(integer(1));

    for (int i = 0; i < 10000; i++) {
        e = pow(add(mul(add(e, pow(integer(2), integer(-3))), integer(3)),
                    integer(1)),
                div(integer(2), integer(3)));
    }
    return e;
}

NONIUS_BENCHMARK("eval_double", [](nonius::chronometer meter) {
    auto e = get_eval_double_expression();
    double r;
    meter.measure([&](int i) { r = eval_double(*e); });
})

NONIUS_BENCHMARK("eval_double_visitor_pattern", [](nonius::chronometer meter) {
    auto e = get_eval_double_expression();
    double r;
    meter.measure([&](int i) { r = eval_double_visitor_pattern(*e); });
})

NONIUS_BENCHMARK("eval_double_single_dispatch", [](nonius::chronometer meter) {
    auto e = get_eval_double_expression();
    double r;
    meter.measure([&](int i) { r = eval_double_single_dispatch(*e); });
})
