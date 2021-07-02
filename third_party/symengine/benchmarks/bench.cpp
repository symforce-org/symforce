#define NONIUS_RUNNER
#include "nonius.h++"

#include <symengine/basic.h>
#include <symengine/add.h>
#include <symengine/symbol.h>
#include <symengine/dict.h>
#include <symengine/integer.h>
#include <symengine/mul.h>
#include <symengine/pow.h>

using SymEngine::Basic;
using SymEngine::symbol;
using SymEngine::integer;
using SymEngine::RCP;

NONIUS_BENCHMARK("expand1", [](nonius::chronometer meter) {
    auto x = symbol("x"), y = symbol("y"), z = symbol("z"), w = symbol("w");
    auto i60 = integer(60);
    std::vector<RCP<const Basic>> e(meter.runs()), r(meter.runs());
    for (auto &v : e) {
        v = pow(add(add(add(x, y), z), w), i60);
    }
    meter.measure([&](int i) { r[i] = expand(e[i]); });
})

NONIUS_BENCHMARK("expand2", [](nonius::chronometer meter) {
    auto x = symbol("x"), y = symbol("y"), z = symbol("z"), w = symbol("w");
    auto i15 = integer(15);
    std::vector<RCP<const Basic>> f(meter.runs()), r(meter.runs());
    for (auto &v : f) {
        auto e = pow(add(add(add(x, y), z), w), i15);
        v = mul(e, add(e, w));
    }
    meter.measure([&](int i) { r[i] = expand(f[i]); });
})

NONIUS_BENCHMARK("expand3", [](nonius::chronometer meter) {
    auto x = symbol("x"), y = symbol("y"), z = symbol("z");
    auto i100 = integer(100);
    std::vector<RCP<const Basic>> e(meter.runs()), r(meter.runs());
    for (auto &v : e) {
        v = pow(add(add(pow(x, y), pow(y, x)), pow(z, x)), i100);
    }
    meter.measure([&](int i) { r[i] = expand(e[i]); });
})
