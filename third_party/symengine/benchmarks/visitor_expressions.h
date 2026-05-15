#ifndef SYMENGINE_BENCHMARKS_VISITOR_EXPRESSIONS_H
#define SYMENGINE_BENCHMARKS_VISITOR_EXPRESSIONS_H

#include <benchmark/benchmark.h>
#include <symengine/lambda_double.h>
#include <symengine/llvm_double.h>
#include <math.h>
#include <symengine/add.h>
#include <symengine/integer.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/symbol.h>
#include <symengine/functions.h>
#include <symengine/matrix.h>
#include <array>

using SymEngine::add;
using SymEngine::Basic;
using SymEngine::cos;
using SymEngine::DenseMatrix;
using SymEngine::E;
using SymEngine::integer;
using SymEngine::LambdaRealDoubleVisitor;
using SymEngine::LLVMDoubleVisitor;
using SymEngine::LLVMFloatVisitor;
using SymEngine::log;
using SymEngine::mul;
using SymEngine::pi;
using SymEngine::pow;
using SymEngine::RCP;
using SymEngine::sin;
using SymEngine::sqrt;
using SymEngine::symbol;
using SymEngine::vec_basic;

static const std::vector<int64_t> opt_code_values{0, 1, 2, 3};
static const std::vector<int64_t> cse_values{0, 1};

std::string to_label(bool cse, unsigned opt_level)
{
    std::string label;
    if (cse) {
        label.append("cse_");
    }
    label.append("O");
    label.append(std::to_string(opt_level));
    return label;
}

#define SYMENGINE_BENCHMARK_VISITORS_EXPR(func, expr)                          \
    BENCHMARK_TEMPLATE(func, LambdaRealDoubleVisitor, expr, double)            \
        ->ArgsProduct({cse_values, {0}});                                      \
    BENCHMARK_TEMPLATE(func, LLVMDoubleVisitor, expr, double)                  \
        ->ArgsProduct({cse_values, opt_code_values});                          \
    BENCHMARK_TEMPLATE(func, LLVMFloatVisitor, expr, float)                    \
        ->ArgsProduct({cse_values, opt_code_values})

// to add an expression to these benchmarks:
//  - add a line to the macro below with your expression ExprN
//  - define a struct ExprN that implements vec and expr()

#define SYMENGINE_BENCHMARK_VISITORS(func)                                     \
    SYMENGINE_BENCHMARK_VISITORS_EXPR(func, Expr1);                            \
    SYMENGINE_BENCHMARK_VISITORS_EXPR(func, Expr2);                            \
    SYMENGINE_BENCHMARK_VISITORS_EXPR(func, Expr3);                            \
    SYMENGINE_BENCHMARK_VISITORS_EXPR(func, Expr4);                            \
    SYMENGINE_BENCHMARK_VISITORS_EXPR(func, Expr5)

void init(LLVMFloatVisitor &v, const vec_basic &args, const vec_basic &expr,
          bool cse, unsigned opt_level)
{
    v.init(args, expr, cse, opt_level);
}

void init(LLVMDoubleVisitor &v, const vec_basic &args, const vec_basic &expr,
          bool cse, unsigned opt_level)
{
    v.init(args, expr, cse, opt_level);
}

void init(LambdaRealDoubleVisitor &v, const vec_basic &args,
          const vec_basic &expr, bool cse, unsigned opt_level)
{
    v.init(args, expr, cse);
}

// each expression struct implements
//   - vec: a vector of inputs
//   - expr(): a vector of outputs that can depend on the inputs

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
};

// copied from llvm_double test case
struct Expr3 {
    vec_basic vec{{symbol("x"), symbol("y"), symbol("z")}};
    vec_basic expr()
    {
        vec_basic r{symbol("r0")};
        auto &x = vec[0];
        auto &y = vec[1];
        auto &z = vec[2];
        vec_basic v = {log(x),
                       abs(x),
                       tan(x),
                       sinh(x),
                       cosh(x),
                       tanh(x),
                       asinh(y),
                       acosh(y),
                       atanh(x),
                       asin(x),
                       acos(x),
                       atan(x),
                       gamma(x),
                       loggamma(x),
                       erf(x),
                       erfc(x),
                       add(pi, div(integer(1), integer(3)))};
        r[0] = mul(add(sin(x), add(mul(pow(y, integer(4)), mul(z, integer(2))),
                                   pow(sin(x), integer(2)))),
                   add(v));
        for (int i = 0; i < 4; ++i) {
            r[0] = mul(
                add(pow(integer(2), E), add(r[0], pow(x, pow(E, cos(x))))),
                r[0]);
        }
        return r;
    }
};

// large cse-friendly expression based on
// https://github.com/symengine/symengine/pull/1612
struct Expr4 {
    vec_basic vec{symbol("a"), symbol("b"), symbol("c"),
                  symbol("d"), symbol("e"), symbol("f")};
    vec_basic expr()
    {
        RCP<const Basic> e = integer(23);
        const std::size_t n{vec.size()};
        for (std::size_t i = 0; i < n; ++i) {
            e = pow(e,
                    add(cos(sqrt(log(sin(pow(vec[n - i - 1], vec[i]))))), e));
        }
        e = expand(e);
        DenseMatrix M(1, 1, {e});
        DenseMatrix S(n, 1, vec);
        DenseMatrix J(1, n);
        jacobian(M, S, J);
        vec_basic expression;
        for (std::size_t i = 0; i < n; ++i) {
            expression.push_back(J.get(0, i));
        }
        return expression;
    }
};

// large cse-friendly expression based on
// https://github.com/symengine/symengine/pull/1612
struct Expr5 {
    vec_basic vec{symbol("a"), symbol("b"), symbol("c"),
                  symbol("d"), symbol("e"), symbol("f")};
    vec_basic expr()
    {
        RCP<const Basic> e = integer(23);
        const std::size_t n{vec.size()};
        for (std::size_t i = 0; i < n; ++i) {
            e = pow(e, cos(sqrt(log(sin(pow(vec[n - i - 1], vec[i]))))));
        }
        e = expand(e);
        DenseMatrix M(1, 1, {e});
        DenseMatrix S(n, 1, vec);
        DenseMatrix J(1, n);
        jacobian(M, S, J);
        vec_basic expression;
        for (std::size_t i = 0; i < n; ++i) {
            expression.push_back(J.get(0, i));
        }
        return expression;
    }
};

#endif