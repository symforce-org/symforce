#include <symengine/complex.h>
#include <symengine/add.h>
#include <symengine/infinity.h>
#include <symengine/pow.h>
#include <symengine/nan.h>

namespace SymEngine
{

Constant::Constant(const std::string &name)
    : name_{name} {SYMENGINE_ASSIGN_TYPEID()}

      hash_t Constant::__hash__() const
{
    hash_t seed = SYMENGINE_CONSTANT;
    hash_combine<std::string>(seed, name_);
    return seed;
}

bool Constant::__eq__(const Basic &o) const
{
    if (is_a<Constant>(o))
        return name_ == down_cast<const Constant &>(o).name_;
    return false;
}

int Constant::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<Constant>(o))
    const Constant &s = down_cast<const Constant &>(o);
    if (name_ == s.name_)
        return 0;
    return name_ < s.name_ ? -1 : 1;
}

#define DEFINE_CONSTANT(t, n, d)                                               \
    RCP<const t> n = []() {                                                    \
        static const RCP<const t> c = d;                                       \
        return c;                                                              \
    }()

DEFINE_CONSTANT(Integer, zero, integer(0));
DEFINE_CONSTANT(Integer, one, integer(1));
DEFINE_CONSTANT(Integer, minus_one, integer(-1));
DEFINE_CONSTANT(Integer, two, integer(2));

DEFINE_CONSTANT(Number, I, Complex::from_two_nums(*zero, *one));

DEFINE_CONSTANT(Constant, pi, constant("pi"));
DEFINE_CONSTANT(Constant, E, constant("E"));
DEFINE_CONSTANT(Constant, EulerGamma, constant("EulerGamma"));
DEFINE_CONSTANT(Constant, Catalan, constant("Catalan"));
DEFINE_CONSTANT(Constant, GoldenRatio, constant("GoldenRatio"));

DEFINE_CONSTANT(Infty, Inf, Infty::from_int(1));
DEFINE_CONSTANT(Infty, NegInf, Infty::from_int(-1));
DEFINE_CONSTANT(Infty, ComplexInf, Infty::from_int(0));

DEFINE_CONSTANT(NaN, Nan, make_rcp<NaN>());

// Global variables declared in functions.cpp
// Look over https://github.com/sympy/symengine/issues/272
// for further details
DEFINE_CONSTANT(Basic, i2, integer(2));

namespace
{
RCP<const Basic> sqrt_(const RCP<const Basic> &arg)
{
    return pow(arg, div(one, i2));
}
} // namespace

DEFINE_CONSTANT(Basic, i3, integer(3));
DEFINE_CONSTANT(Basic, i5, integer(5));
DEFINE_CONSTANT(Basic, im2, integer(-2));
DEFINE_CONSTANT(Basic, im3, integer(-3));
DEFINE_CONSTANT(Basic, im5, integer(-5));

DEFINE_CONSTANT(Basic, sq3, sqrt_(i3));
DEFINE_CONSTANT(Basic, sq2, sqrt_(i2));
DEFINE_CONSTANT(Basic, sq5, sqrt_(i5));

DEFINE_CONSTANT(Basic, C0, div(sub(sq3, one), mul(i2, sq2)));
DEFINE_CONSTANT(Basic, C1, div(one, i2));
DEFINE_CONSTANT(Basic, C2, div(sq2, i2));
DEFINE_CONSTANT(Basic, C3, div(sq3, i2));
DEFINE_CONSTANT(Basic, C4, div(add(sq3, one), mul(i2, sq2)));
DEFINE_CONSTANT(Basic, C5, div(sqrt_(sub(i5, sqrt_(i5))), integer(8)));
DEFINE_CONSTANT(Basic, C6, div(sub(sqrt_(i5), one), integer(4)));

DEFINE_CONSTANT(Basic, mC0, mul(minus_one, C0));
DEFINE_CONSTANT(Basic, mC1, mul(minus_one, C1));
DEFINE_CONSTANT(Basic, mC2, mul(minus_one, C2));
DEFINE_CONSTANT(Basic, mC3, mul(minus_one, C3));
DEFINE_CONSTANT(Basic, mC4, mul(minus_one, C4));
DEFINE_CONSTANT(Basic, mC5, mul(minus_one, C5));
DEFINE_CONSTANT(Basic, mC6, mul(minus_one, C6));

#undef DEFINE_CONSTANT

} // namespace SymEngine
