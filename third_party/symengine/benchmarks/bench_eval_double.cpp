#include <benchmark/benchmark.h>
#include <symengine/basic.h>
#include <symengine/add.h>
#include <symengine/symbol.h>
#include <symengine/dict.h>
#include <symengine/integer.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/eval_double.h>

using SymEngine::Basic;
using SymEngine::integer;
using SymEngine::RCP;
using SymEngine::symbol;

RCP<const Basic> get_eval_double_expression(int n)
{
    RCP<const Basic> e = sin(integer(1));

    for (int i = 0; i < n; i++) {
        e = pow(add(mul(add(e, pow(integer(2), integer(-3))), integer(3)),
                    integer(1)),
                div(integer(2), integer(3)));
    }
    return e;
}

void eval_double(benchmark::State &state)
{
    auto e = get_eval_double_expression(state.range(0));
    double r{0};
    for (auto _ : state) {
        r += eval_double(*e);
    }
    state.SetComplexityN(state.range(0));
}

void eval_double_visitor_pattern(benchmark::State &state)
{
    auto e = get_eval_double_expression(state.range(0));
    double r{0};
    for (auto _ : state) {
        r += eval_double_visitor_pattern(*e);
    }
    state.SetComplexityN(state.range(0));
}

void eval_double_single_dispatch(benchmark::State &state)
{
    auto e = get_eval_double_expression(state.range(0));
    double r{0};
    for (auto _ : state) {
        r += eval_double_single_dispatch(*e);
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(eval_double)->Range(1 << 1, 1 << 14)->Complexity();
BENCHMARK(eval_double_visitor_pattern)->Range(1 << 1, 1 << 14)->Complexity();
BENCHMARK(eval_double_single_dispatch)->Range(1 << 1, 1 << 14)->Complexity();

BENCHMARK_MAIN();
