#include <benchmark/benchmark.h>
#include <symengine/basic.h>
#include <symengine/add.h>
#include <symengine/symbol.h>
#include <symengine/dict.h>
#include <symengine/integer.h>
#include <symengine/mul.h>
#include <symengine/pow.h>

using SymEngine::Basic;
using SymEngine::integer;
using SymEngine::RCP;
using SymEngine::symbol;

void expand1(benchmark::State &state)
{
    auto x = symbol("x"), y = symbol("y"), z = symbol("z"), w = symbol("w");
    auto i = integer(state.range(0));
    RCP<const Basic> e, r;
    e = pow(add(add(add(x, y), z), w), i);
    for (auto _ : state) {
        r = expand(e);
    }
    state.SetComplexityN(state.range(0));
}

void expand2(benchmark::State &state)
{
    auto x = symbol("x"), y = symbol("y"), z = symbol("z"), w = symbol("w");
    auto i = integer(state.range(0));
    RCP<const Basic> e, r;
    e = pow(add(add(add(x, y), z), w), i);
    e = mul(e, add(e, w));
    for (auto _ : state) {
        r = expand(e);
    }
    state.SetComplexityN(state.range(0));
}

void expand3(benchmark::State &state)
{
    auto x = symbol("x"), y = symbol("y"), z = symbol("z");
    auto i = integer(state.range(0));
    RCP<const Basic> e, r;
    e = pow(add(add(pow(x, y), pow(y, x)), pow(z, x)), i);
    for (auto _ : state) {
        r = expand(e);
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(expand1)->RangeMultiplier(2)->Range(2, 64)->Complexity();
BENCHMARK(expand2)->RangeMultiplier(2)->Range(2, 16)->Complexity();
BENCHMARK(expand3)->RangeMultiplier(2)->Range(2, 128)->Complexity();

BENCHMARK_MAIN();
