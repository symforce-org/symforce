#include <benchmark/benchmark.h>
#include <symengine/parser.h>
#include <symengine/parser/parser.h>

using SymEngine::Basic;
using SymEngine::parse;
using SymEngine::RCP;

static void parse_0(benchmark::State &state)
{
    RCP<const Basic> a;
    for (auto _ : state) {
        a = parse("0");
    }
}

static void parse_long_expr1(benchmark::State &state)
{
    std::string text = "1";
    std::string t0 = " * (x + y - sin(x)/(z**2-4) - x**(y**z))";
    for (int i = 0; i < state.range(0); i++) {
        text.append(t0);
    }
    RCP<const Basic> a;
    for (auto _ : state) {
        a = parse(text);
    }
    state.SetComplexityN(state.range(0));
}

static void parse_long_expr2(benchmark::State &state)
{
    std::string text = "1";
    std::string t0 = " * (cos(x) + arcsinh(y - sin(x))/(z**2-4) - x**(y**z))";
    for (int i = 0; i < state.range(0); i++) {
        text.append(t0);
    }
    RCP<const Basic> a;
    for (auto _ : state) {
        a = parse(text);
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(parse_0);
BENCHMARK(parse_long_expr1)->Range(2, 4096)->Complexity();
BENCHMARK(parse_long_expr2)->Range(2, 4096)->Complexity();

BENCHMARK_MAIN();
