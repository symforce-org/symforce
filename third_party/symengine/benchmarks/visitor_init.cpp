#include <benchmark/benchmark.h>
#include "visitor_expressions.h"

template <typename Visitor, typename Expr, typename Real>
static void Init(benchmark::State &state)
{
    bool cse{static_cast<bool>(state.range(0))};
    unsigned opt_level{static_cast<unsigned>(state.range(1))};
    Expr e;
    vec_basic inputs{e.vec};
    vec_basic outputs{e.expr()};
    Visitor v;
    for (auto _ : state) {
        init(v, inputs, outputs, cse, opt_level);
    }
    state.SetLabel(to_label(cse, opt_level));
}

SYMENGINE_BENCHMARK_VISITORS(Init);

BENCHMARK_MAIN();
