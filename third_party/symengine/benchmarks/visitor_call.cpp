#include <benchmark/benchmark.h>
#include "visitor_expressions.h"

struct CompiledExpr1 {
    void call(double *d, const double *v)
    {
        double r1
            = std::sin(v[0] + std::cos((v[1] * v[2]) + std::pow(v[0], 2)));
        double r2 = (3 + r1) * (2 + r1);
        *d = std::pow((5 + r2), (r2 - 2));
    }
    void call(float *d, const float *v)
    {
        float r1 = sinf(v[0] + cosf((v[1] * v[2]) + powf(v[0], 2)));
        float r2 = (3 + r1) * (2 + r1);
        *d = powf((5 + r2), (r2 - 2));
    }
};

struct CompiledExpr2 {
    template <typename Real>
    void call(Real *d, const Real *v)
    {
        d[0] = static_cast<Real>(2.0) * (v[0] + v[0] + (v[1] * v[2]));
        d[1] = v[0] + v[0] + (v[2] * v[1]);
        d[2] = -static_cast<Real>(2.0) * (v[0] + v[0] + (v[1] * v[2]));
    }
};

void init(CompiledExpr1 &v, const vec_basic &args, const vec_basic &expr,
          bool cse, unsigned opt_level){};

void init(CompiledExpr2 &v, const vec_basic &args, const vec_basic &expr,
          bool cse, unsigned opt_level){};

template <typename Visitor, typename Expr, typename Real>
static void Call(benchmark::State &state)
{
    Expr e;
    vec_basic inputs{e.vec};
    vec_basic outputs{e.expr()};
    const std::size_t n_inputs{inputs.size()};
    const std::size_t n_outputs{outputs.size()};
    std::vector<Real> s(n_outputs, 0.0);
    std::vector<Real> d(n_outputs, 0.0);
    std::vector<Real> x(n_inputs, 0.0);
    for (std::size_t i = 0; i < n_inputs; ++i) {
        x[i] = static_cast<Real>(1.732 * i);
    }
    Visitor v;
    bool cse{static_cast<bool>(state.range(0))};
    unsigned opt_level{static_cast<unsigned>(state.range(1))};
    init(v, inputs, outputs, cse, opt_level);
    for (auto _ : state) {
        for (std::size_t i = 0; i < n_inputs; ++i) {
            x[i] += static_cast<Real>(0.1);
        }
        v.call(d.data(), x.data());
        benchmark::ClobberMemory();
        for (std::size_t i = 0; i < n_outputs; ++i) {
            s[i] += d[i];
        }
    }
    state.SetLabel(to_label(cse, opt_level));
}

SYMENGINE_BENCHMARK_VISITORS(Call);

// repeat benchmarks with natively compiled version of expressions
BENCHMARK_TEMPLATE(Call, CompiledExpr1, Expr1, double)->Args({0, 0});
BENCHMARK_TEMPLATE(Call, CompiledExpr1, Expr1, float)->Args({0, 0});
BENCHMARK_TEMPLATE(Call, CompiledExpr2, Expr2, double)->Args({0, 0});
BENCHMARK_TEMPLATE(Call, CompiledExpr2, Expr2, float)->Args({0, 0});

BENCHMARK_MAIN();
