#include <benchmark/benchmark.h>
#include <math.h>
#include <symengine/add.h>
#include <symengine/integer.h>
#include <symengine/lambda_double.h>
#include <symengine/llvm_double.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/symbol.h>

using SymEngine::add;
using SymEngine::Basic;
using SymEngine::cos;
using SymEngine::integer;
using SymEngine::LambdaRealDoubleVisitor;
using SymEngine::LLVMDoubleVisitor;
using SymEngine::LLVMFloatVisitor;
using SymEngine::mul;
using SymEngine::pow;
using SymEngine::RCP;
using SymEngine::sin;
using SymEngine::symbol;
using SymEngine::vec_basic;

struct Expr1 {
    vec_basic vec{{symbol("x"), symbol("y"), symbol("z")}};
    vec_basic expr()
    {
        vec_basic r{symbol("r0")};
        r[0] = sin(add(vec[0],
                       cos(add(mul(vec[1], vec[2]), pow(vec[0], integer(2))))));
        r[0] = mul(add(integer(3), r[0]), add(integer(2), r[0]));
        r[0] = pow(add(integer(5), r[0]), add(integer(-2), r[0]));
        return r;
    }
    static void compiled_expr(double *d, const double *v)
    {
        double r1
            = std::sin(v[0] + std::cos((v[1] * v[2]) + std::pow(v[0], 2)));
        double r2 = (3 + r1) * (2 + r1);
        *d = std::pow((5 + r2), (r2 - 2));
    }
    static void compiled_expr(float *d, const float *v)
    {
        float r1 = sinf(v[0] + cosf((v[1] * v[2]) + powf(v[0], 2)));
        float r2 = (3 + r1) * (2 + r1);
        *d = powf((5 + r2), (r2 - 2));
    }
};

struct Expr2 {
    vec_basic vec{{symbol("x"), symbol("y"), symbol("z")}};
    vec_basic expr()
    {
        vec_basic r{symbol("r0"), symbol("r1"), symbol("r2")};
        r[0] = mul(integer(2), add(vec[0], add(vec[0], mul(vec[1], vec[2]))));
        r[1] = add(vec[0], add(vec[0], mul(vec[2], vec[1])));
        r[2] = mul(integer(-2), add(vec[0], add(vec[0], mul(vec[1], vec[2]))));
        return r;
    }
    template <typename Real>
    static void compiled_expr(Real *d, const Real *v)
    {
        d[0] = 2.0 * (v[0] + v[0] + (v[1] * v[2]));
        d[1] = v[0] + v[0] + (v[2] * v[1]);
        d[2] = -2.0 * (v[0] + v[0] + (v[1] * v[2]));
    }
};

template <typename Real>
struct NativeVisitor {
    void (*f)(Real *, const Real *){nullptr};
    void call(Real *d, const Real *v)
    {
        f(d, v);
    }
};

template <typename Expr, typename Real>
std::string init(NativeVisitor<Real> &v, bool cse, unsigned opt_level)
{
    v.f = &Expr::compiled_expr;
    return {};
}

template <typename Expr>
std::string init(LambdaRealDoubleVisitor &v, bool cse, unsigned opt_level)
{
    std::string label;
    Expr e;
    v.init(e.vec, e.expr(), cse);
    if (cse) {
        label = "cse";
    }
    return label;
}

template <typename Expr>
std::string init(LLVMDoubleVisitor &v, bool cse, unsigned opt_level)
{
    std::string label;
    Expr e;
    v.init(e.vec, e.expr(), cse, opt_level);
    if (cse) {
        label.append("cse_");
    }
    label.append("O");
    label.append(std::to_string(opt_level));
    return label;
}

template <typename Expr>
std::string init(LLVMFloatVisitor &v, bool cse, unsigned opt_level)
{
    std::string label;
    Expr e;
    v.init(e.vec, e.expr(), cse, opt_level);
    if (cse) {
        label.append("cse_");
    }
    label.append("O");
    label.append(std::to_string(opt_level));
    return label;
}

template <typename Visitor, typename Expr, typename Real>
static void Call(benchmark::State &state)
{
    std::vector<Real> s{0.0, 0.0, 0.0};
    std::vector<Real> d{0.0, 0.0, 0.0};
    std::vector<Real> x{1.0, 4.4365, 12.8};
    Visitor v;
    bool cse{static_cast<bool>(state.range(0))};
    unsigned opt_level{static_cast<unsigned>(state.range(1))};
    auto label = init<Expr>(v, cse, opt_level);
    for (auto _ : state) {
        x[0] += 0.1;
        x[1] += 0.2;
        x[2] += 0.3;
        v.call(d.data(), x.data());
        benchmark::ClobberMemory();
        s[0] += d[0];
        s[1] += d[1];
        s[2] += d[2];
    }
    state.SetLabel(label);
}

// BENCHMARK_TEMPLATE(BenchmarkName, VisitorClass,
// ExpressionClass, RealType)->ArgsProduct({{cse values}, {opt_level values}});

static std::vector<int64_t> opt_code_values{0, 1, 2, 3};
static std::vector<int64_t> cse_values{0, 1};

BENCHMARK_TEMPLATE(Call, LambdaRealDoubleVisitor, Expr1, double)
    ->ArgsProduct({cse_values, {0}});
BENCHMARK_TEMPLATE(Call, LLVMDoubleVisitor, Expr1, double)
    ->ArgsProduct({cse_values, opt_code_values});
BENCHMARK_TEMPLATE(Call, NativeVisitor<double>, Expr1, double)->Args({0, 0});
BENCHMARK_TEMPLATE(Call, LLVMFloatVisitor, Expr1, float)
    ->ArgsProduct({cse_values, opt_code_values});
BENCHMARK_TEMPLATE(Call, NativeVisitor<float>, Expr1, float)->Args({0, 0});

BENCHMARK_TEMPLATE(Call, LambdaRealDoubleVisitor, Expr2, double)
    ->ArgsProduct({cse_values, {0}});
BENCHMARK_TEMPLATE(Call, LLVMDoubleVisitor, Expr2, double)
    ->ArgsProduct({cse_values, opt_code_values});
BENCHMARK_TEMPLATE(Call, NativeVisitor<double>, Expr2, double)->Args({0, 0});
BENCHMARK_TEMPLATE(Call, LLVMFloatVisitor, Expr2, float)
    ->ArgsProduct({cse_values, opt_code_values});
BENCHMARK_TEMPLATE(Call, NativeVisitor<float>, Expr2, float)->Args({0, 0});

BENCHMARK_MAIN();
