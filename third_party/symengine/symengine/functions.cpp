#include <symengine/visitor.h>
#include <symengine/symengine_exception.h>

namespace SymEngine
{

extern RCP<const Basic> i2;
extern RCP<const Basic> i3;
extern RCP<const Basic> i5;
extern RCP<const Basic> im2;
extern RCP<const Basic> im3;
extern RCP<const Basic> im5;

RCP<const Basic> sqrt(RCP<const Basic> &arg)
{
    return pow(arg, div(one, i2));
}
RCP<const Basic> cbrt(RCP<const Basic> &arg)
{
    return pow(arg, div(one, i3));
}

extern RCP<const Basic> sq3;
extern RCP<const Basic> sq2;
extern RCP<const Basic> sq5;

extern RCP<const Basic> C0;
extern RCP<const Basic> C1;
extern RCP<const Basic> C2;
extern RCP<const Basic> C3;
extern RCP<const Basic> C4;
extern RCP<const Basic> C5;
extern RCP<const Basic> C6;

extern RCP<const Basic> mC0;
extern RCP<const Basic> mC1;
extern RCP<const Basic> mC2;
extern RCP<const Basic> mC3;
extern RCP<const Basic> mC4;
extern RCP<const Basic> mC5;
extern RCP<const Basic> mC6;

extern RCP<const Basic> sin_table[];

extern umap_basic_basic inverse_cst;

extern umap_basic_basic inverse_tct;

Conjugate::Conjugate(const RCP<const Basic> &arg) : OneArgFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool Conjugate::is_canonical(const RCP<const Basic> &arg) const
{
    if (is_a_Number(*arg)) {
        if (eq(*arg, *ComplexInf)) {
            return true;
        }
        return false;
    }
    if (is_a<Constant>(*arg)) {
        return false;
    }
    if (is_a<Mul>(*arg)) {
        return false;
    }
    if (is_a<Pow>(*arg)) {
        if (is_a<Integer>(*down_cast<const Pow &>(*arg).get_exp())) {
            return false;
        }
    }
    // OneArgFunction classes
    if (is_a<Sign>(*arg) or is_a<Conjugate>(*arg) or is_a<Erf>(*arg)
        or is_a<Erfc>(*arg) or is_a<Gamma>(*arg) or is_a<LogGamma>(*arg)
        or is_a<Abs>(*arg)) {
        return false;
    }
    if (is_a<Sin>(*arg) or is_a<Cos>(*arg) or is_a<Tan>(*arg) or is_a<Cot>(*arg)
        or is_a<Sec>(*arg) or is_a<Csc>(*arg)) {
        return false;
    }
    if (is_a<Sinh>(*arg) or is_a<Cosh>(*arg) or is_a<Tanh>(*arg)
        or is_a<Coth>(*arg) or is_a<Sech>(*arg) or is_a<Csch>(*arg)) {
        return false;
    }
    // TwoArgFunction classes
    if (is_a<KroneckerDelta>(*arg) or is_a<ATan2>(*arg)
        or is_a<LowerGamma>(*arg) or is_a<UpperGamma>(*arg)
        or is_a<Beta>(*arg)) {
        return false;
    }
    // MultiArgFunction class
    if (is_a<LeviCivita>(*arg)) {
        return false;
    }
    return true;
}

RCP<const Basic> Conjugate::create(const RCP<const Basic> &arg) const
{
    return conjugate(arg);
}

RCP<const Basic> conjugate(const RCP<const Basic> &arg)
{
    if (is_a_Number(*arg)) {
        return down_cast<const Number &>(*arg).conjugate();
    }
    if (is_a<Constant>(*arg) or is_a<Abs>(*arg) or is_a<KroneckerDelta>(*arg)
        or is_a<LeviCivita>(*arg)) {
        return arg;
    }
    if (is_a<Mul>(*arg)) {
        const map_basic_basic &dict = down_cast<const Mul &>(*arg).get_dict();
        map_basic_basic new_dict;
        RCP<const Number> coef = rcp_static_cast<const Number>(
            conjugate(down_cast<const Mul &>(*arg).get_coef()));
        for (const auto &p : dict) {
            if (is_a<Integer>(*p.second)) {
                Mul::dict_add_term_new(outArg(coef), new_dict, p.second,
                                       conjugate(p.first));
            } else {
                Mul::dict_add_term_new(
                    outArg(coef), new_dict, one,
                    conjugate(Mul::from_dict(one, {{p.first, p.second}})));
            }
        }
        return Mul::from_dict(coef, std::move(new_dict));
    }
    if (is_a<Pow>(*arg)) {
        RCP<const Basic> base = down_cast<const Pow &>(*arg).get_base();
        RCP<const Basic> exp = down_cast<const Pow &>(*arg).get_exp();
        if (is_a<Integer>(*exp)) {
            return pow(conjugate(base), exp);
        }
    }
    if (is_a<Conjugate>(*arg)) {
        return down_cast<const Conjugate &>(*arg).get_arg();
    }
    if (is_a<Sign>(*arg) or is_a<Erf>(*arg) or is_a<Erfc>(*arg)
        or is_a<Gamma>(*arg) or is_a<LogGamma>(*arg) or is_a<Sin>(*arg)
        or is_a<Cos>(*arg) or is_a<Tan>(*arg) or is_a<Cot>(*arg)
        or is_a<Sec>(*arg) or is_a<Csc>(*arg) or is_a<Sinh>(*arg)
        or is_a<Cosh>(*arg) or is_a<Tanh>(*arg) or is_a<Coth>(*arg)
        or is_a<Sech>(*arg) or is_a<Csch>(*arg)) {
        const OneArgFunction &func = down_cast<const OneArgFunction &>(*arg);
        return func.create(conjugate(func.get_arg()));
    }
    if (is_a<ATan2>(*arg) or is_a<LowerGamma>(*arg) or is_a<UpperGamma>(*arg)
        or is_a<Beta>(*arg)) {
        const TwoArgFunction &func = down_cast<const TwoArgFunction &>(*arg);
        return func.create(conjugate(func.get_arg1()),
                           conjugate(func.get_arg2()));
    }
    return make_rcp<const Conjugate>(arg);
}

bool get_pi_shift(const RCP<const Basic> &arg, const Ptr<RCP<const Number>> &n,
                  const Ptr<RCP<const Basic>> &x)
{
    if (is_a<Add>(*arg)) {
        const Add &s = down_cast<const Add &>(*arg);
        RCP<const Basic> coef = s.get_coef();
        auto size = s.get_dict().size();
        if (size > 1) {
            // arg should be of form `x + n*pi`
            // `n` is an integer
            // `x` is an `Expression`
            bool check_pi = false;
            RCP<const Basic> temp;
            *x = coef;
            for (const auto &p : s.get_dict()) {
                if (eq(*p.first, *pi) and (is_a<Integer>(*p.second)
                                           or is_a<Rational>(*p.second))) {
                    check_pi = true;
                    *n = p.second;
                } else {
                    *x = add(mul(p.first, p.second), *x);
                }
            }
            if (check_pi)
                return true;
            else // No term with `pi` found
                return false;
        } else if (size == 1) {
            // arg should be of form `a + n*pi`
            // where `a` is a `Number`.
            auto p = s.get_dict().begin();
            if (eq(*p->first, *pi)
                and (is_a<Integer>(*p->second) or is_a<Rational>(*p->second))) {
                *n = p->second;
                *x = coef;
                return true;
            } else {
                return false;
            }
        } else { // Should never reach here though!
            // Dict of size < 1
            return false;
        }
    } else if (is_a<Mul>(*arg)) {
        // `arg` is of the form `k*pi/12`
        const Mul &s = down_cast<const Mul &>(*arg);
        auto p = s.get_dict().begin();
        // dict should contain symbol `pi` only
        if (s.get_dict().size() == 1 and eq(*p->first, *pi)
            and eq(*p->second, *one) and (is_a<Integer>(*s.get_coef())
                                          or is_a<Rational>(*s.get_coef()))) {
            *n = s.get_coef();
            *x = zero;
            return true;
        } else {
            return false;
        }
    } else if (eq(*arg, *pi)) {
        *n = one;
        *x = zero;
        return true;
    } else if (eq(*arg, *zero)) {
        *n = zero;
        *x = zero;
        return true;
    } else {
        return false;
    }
}

// Return true if arg is of form a+b*pi, with b integer or rational
// with denominator 2. The a may be zero or any expression.
bool trig_has_basic_shift(const RCP<const Basic> &arg)
{
    if (is_a<Add>(*arg)) {
        const Add &s = down_cast<const Add &>(*arg);
        for (const auto &p : s.get_dict()) {
            const auto &temp = mul(p.second, integer(2));
            if (eq(*p.first, *pi)) {
                if (is_a<Integer>(*temp)) {
                    return true;
                }
                if (is_a<Rational>(*temp)) {
                    auto m = down_cast<const Rational &>(*temp)
                                 .as_rational_class();
                    return (m < 0) or (m > 1);
                }
                return false;
            }
        }
        return false;
    } else if (is_a<Mul>(*arg)) {
        // is `arg` of the form `k*pi/2`?
        // dict should contain symbol `pi` only
        // and `k` should be a rational s.t. 0 < k < 1
        const Mul &s = down_cast<const Mul &>(*arg);
        RCP<const Basic> coef = mul(s.get_coef(), integer(2));
        auto p = s.get_dict().begin();
        if (s.get_dict().size() == 1 and eq(*p->first, *pi)
            and eq(*p->second, *one)) {
            if (is_a<Integer>(*coef)) {
                return true;
            }
            if (is_a<Rational>(*coef)) {
                auto m = down_cast<const Rational &>(*coef).as_rational_class();
                return (m < 0) or (m > 1);
            }
            return false;
        } else {
            return false;
        }
    } else if (eq(*arg, *pi)) {
        return true;
    } else if (eq(*arg, *zero)) {
        return true;
    } else {
        return false;
    }
}

bool could_extract_minus(const Basic &arg)
{
    if (is_a_Number(arg)) {
        if (down_cast<const Number &>(arg).is_negative()) {
            return true;
        } else if (is_a_Complex(arg)) {
            const ComplexBase &c = down_cast<const ComplexBase &>(arg);
            RCP<const Number> real_part = c.real_part();
            return (real_part->is_negative())
                   or (eq(*real_part, *zero)
                       and c.imaginary_part()->is_negative());
        } else {
            return false;
        }
    } else if (is_a<Mul>(arg)) {
        const Mul &s = down_cast<const Mul &>(arg);
        return could_extract_minus(*s.get_coef());
    } else if (is_a<Add>(arg)) {
        const Add &s = down_cast<const Add &>(arg);
        if (s.get_coef()->is_zero()) {
            map_basic_num d(s.get_dict().begin(), s.get_dict().end());
            return could_extract_minus(*d.begin()->second);
        } else {
            return could_extract_minus(*s.get_coef());
        }
    } else {
        return false;
    }
}

bool handle_minus(const RCP<const Basic> &arg,
                  const Ptr<RCP<const Basic>> &rarg)
{
    if (is_a<Mul>(*arg)) {
        const Mul &s = down_cast<const Mul &>(*arg);
        // Check for -Add instances to transform -(-x + 2*y) to (x - 2*y)
        if (s.get_coef()->is_minus_one() && s.get_dict().size() == 1
            && eq(*s.get_dict().begin()->second, *one)) {
            return not handle_minus(mul(minus_one, arg), rarg);
        } else if (could_extract_minus(*s.get_coef())) {
            *rarg = mul(minus_one, arg);
            return true;
        }
    } else if (is_a<Add>(*arg)) {
        if (could_extract_minus(*arg)) {
            const Add &s = down_cast<const Add &>(*arg);
            umap_basic_num d = s.get_dict();
            for (auto &p : d) {
                p.second = p.second->mul(*minus_one);
            }
            *rarg = Add::from_dict(s.get_coef()->mul(*minus_one), std::move(d));
            return true;
        }
    } else if (could_extract_minus(*arg)) {
        *rarg = mul(minus_one, arg);
        return true;
    }
    *rarg = arg;
    return false;
}

// \return true if conjugate has to be returned finally else false
bool trig_simplify(const RCP<const Basic> &arg, unsigned period, bool odd,
                   bool conj_odd, // input
                   const Ptr<RCP<const Basic>> &rarg, int &index,
                   int &sign) // output
{
    bool check;
    RCP<const Number> n;
    RCP<const Basic> r;
    RCP<const Basic> ret_arg;
    check = get_pi_shift(arg, outArg(n), outArg(r));
    if (check) {
        RCP<const Number> t = mulnum(n, integer(12));
        sign = 1;
        if (is_a<Integer>(*t)) {
            int m = numeric_cast<int>(
                mod_f(down_cast<const Integer &>(*t), *integer(12 * period))
                    ->as_int());
            if (eq(*r, *zero)) {
                index = m;
                *rarg = zero;
                return false;
            } else if (m == 0) {
                index = 0;
                bool b = handle_minus(r, outArg(ret_arg));
                *rarg = ret_arg;
                if (odd and b)
                    sign = -1;
                return false;
            }
        }

        rational_class m;
        if (is_a<Integer>(*n)) {
            // 2*pi periodic => f(r + pi * n) = f(r - pi * n)
            m = mp_abs(down_cast<const Integer &>(*n).as_integer_class());
            m /= period;
        } else {
            SYMENGINE_ASSERT(is_a<Rational>(*n));
            m = down_cast<const Rational &>(*n).as_rational_class() / period;
            integer_class t;
#if SYMENGINE_INTEGER_CLASS != SYMENGINE_BOOSTMP
            mp_fdiv_r(t, get_num(m), get_den(m));
            get_num(m) = t;
#else
            integer_class quo;
            mp_fdiv_qr(quo, t, get_num(m), get_den(m));
            m -= rational_class(quo);
#endif
            // m = a / b => m = (a % b / b)
        }
        // Now, arg = r + 2 * pi * m  where 0 <= m < 1
        m *= 2 * period;
        // Now, arg = r + pi * m / 2  where 0 <= m < 4
        if (m >= 2 and m < 3) {
            sign = -1;
            r = add(r, mul(pi, Rational::from_mpq((m - 2) / 2)));
            bool b = handle_minus(r, outArg(ret_arg));
            *rarg = ret_arg;
            if (odd and b)
                sign = -1 * sign;
            return false;
        } else if (m >= 1) {
            if (m < 2) {
                // 1 <= m < 2
                sign = 1;
                r = add(r, mul(pi, Rational::from_mpq((m - 1) / 2)));
            } else {
                // 3 <= m < 4
                sign = -1;
                r = add(r, mul(pi, Rational::from_mpq((m - 3) / 2)));
            }
            bool b = handle_minus(r, outArg(ret_arg));
            *rarg = ret_arg;
            if (not b and conj_odd)
                sign = -sign;
            return true;
        } else {
            *rarg = add(r, mul(pi, Rational::from_mpq(m / 2)));
            index = -1;
            return false;
        }
    } else {
        bool b = handle_minus(arg, outArg(ret_arg));
        *rarg = ret_arg;
        index = -1;
        if (odd and b)
            sign = -1;
        else
            sign = 1;
        return false;
    }
}

bool inverse_lookup(umap_basic_basic &d, const RCP<const Basic> &t,
                    const Ptr<RCP<const Basic>> &index)
{
    auto it = d.find(t);
    if (it == d.end()) {
        // Not found in lookup
        return false;
    } else {
        *index = (it->second);
        return true;
    }
}

Sign::Sign(const RCP<const Basic> &arg) : OneArgFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool Sign::is_canonical(const RCP<const Basic> &arg) const
{
    if (is_a_Number(*arg)) {
        if (eq(*arg, *ComplexInf)) {
            return true;
        }
        return false;
    }
    if (is_a<Constant>(*arg)) {
        return false;
    }
    if (is_a<Sign>(*arg)) {
        return false;
    }
    if (is_a<Mul>(*arg)) {
        if (neq(*down_cast<const Mul &>(*arg).get_coef(), *one)
            and neq(*down_cast<const Mul &>(*arg).get_coef(), *minus_one)) {
            return false;
        }
    }
    return true;
}

RCP<const Basic> Sign::create(const RCP<const Basic> &arg) const
{
    return sign(arg);
}

RCP<const Basic> sign(const RCP<const Basic> &arg)
{
    if (is_a_Number(*arg)) {
        if (is_a<NaN>(*arg)) {
            return Nan;
        }
        if (down_cast<const Number &>(*arg).is_zero()) {
            return zero;
        }
        if (down_cast<const Number &>(*arg).is_positive()) {
            return one;
        }
        if (down_cast<const Number &>(*arg).is_negative()) {
            return minus_one;
        }
        if (is_a_Complex(*arg)
            and down_cast<const ComplexBase &>(*arg).is_re_zero()) {
            RCP<const Number> r
                = down_cast<const ComplexBase &>(*arg).imaginary_part();
            if (down_cast<const Number &>(*r).is_positive()) {
                return I;
            }
            if (down_cast<const Number &>(*r).is_negative()) {
                return mul(minus_one, I);
            }
        }
    }
    if (is_a<Constant>(*arg)) {
        if (eq(*arg, *pi) or eq(*arg, *E) or eq(*arg, *EulerGamma)
            or eq(*arg, *Catalan) or eq(*arg, *GoldenRatio))
            return one;
    }
    if (is_a<Sign>(*arg)) {
        return arg;
    }
    if (is_a<Mul>(*arg)) {
        RCP<const Basic> s = sign(down_cast<const Mul &>(*arg).get_coef());
        map_basic_basic dict = down_cast<const Mul &>(*arg).get_dict();
        return mul(s,
                   make_rcp<const Sign>(Mul::from_dict(one, std::move(dict))));
    }
    return make_rcp<const Sign>(arg);
}

Floor::Floor(const RCP<const Basic> &arg) : OneArgFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool Floor::is_canonical(const RCP<const Basic> &arg) const
{
    if (is_a_Number(*arg)) {
        return false;
    }
    if (is_a<Constant>(*arg)) {
        return false;
    }
    if (is_a<Floor>(*arg)) {
        return false;
    }
    if (is_a<Ceiling>(*arg)) {
        return false;
    }
    if (is_a<Truncate>(*arg)) {
        return false;
    }
    if (is_a<BooleanAtom>(*arg) or is_a_Relational(*arg)) {
        return false;
    }
    if (is_a<Add>(*arg)) {
        RCP<const Number> s = down_cast<const Add &>(*arg).get_coef();
        if (neq(*zero, *s) and is_a<Integer>(*s)) {
            return false;
        }
    }
    return true;
}

RCP<const Basic> Floor::create(const RCP<const Basic> &arg) const
{
    return floor(arg);
}

RCP<const Basic> floor(const RCP<const Basic> &arg)
{
    if (is_a_Number(*arg)) {
        if (down_cast<const Number &>(*arg).is_exact()) {
            if (is_a<Rational>(*arg)) {
                const Rational &s = down_cast<const Rational &>(*arg);
                integer_class quotient;
                mp_fdiv_q(quotient, SymEngine::get_num(s.as_rational_class()),
                          SymEngine::get_den(s.as_rational_class()));
                return integer(std::move(quotient));
            }
            return arg;
        }
        RCP<const Number> _arg = rcp_static_cast<const Number>(arg);
        return _arg->get_eval().floor(*_arg);
    }
    if (is_a<Constant>(*arg)) {
        if (eq(*arg, *pi)) {
            return integer(3);
        }
        if (eq(*arg, *E)) {
            return integer(2);
        }
        if (eq(*arg, *GoldenRatio)) {
            return integer(1);
        }
        if (eq(*arg, *Catalan) or eq(*arg, *EulerGamma)) {
            return integer(0);
        }
    }
    if (is_a<Floor>(*arg)) {
        return arg;
    }
    if (is_a<Ceiling>(*arg)) {
        return arg;
    }
    if (is_a<Truncate>(*arg)) {
        return arg;
    }
    if (is_a<BooleanAtom>(*arg) or is_a_Relational(*arg)) {
        throw SymEngineException(
            "Boolean objects not allowed in this context.");
    }
    if (is_a<Add>(*arg)) {
        RCP<const Number> s = down_cast<const Add &>(*arg).get_coef();
        umap_basic_num d = down_cast<const Add &>(*arg).get_dict();
        if (is_a<Integer>(*s)) {
            return add(
                s, make_rcp<const Floor>(Add::from_dict(zero, std::move(d))));
        }
    }
    return make_rcp<const Floor>(arg);
}

Ceiling::Ceiling(const RCP<const Basic> &arg) : OneArgFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool Ceiling::is_canonical(const RCP<const Basic> &arg) const
{
    if (is_a_Number(*arg)) {
        return false;
    }
    if (is_a<Constant>(*arg)) {
        return false;
    }
    if (is_a<Floor>(*arg)) {
        return false;
    }
    if (is_a<Ceiling>(*arg)) {
        return false;
    }
    if (is_a<Truncate>(*arg)) {
        return false;
    }
    if (is_a<BooleanAtom>(*arg) or is_a_Relational(*arg)) {
        return false;
    }
    if (is_a<Add>(*arg)) {
        RCP<const Number> s = down_cast<const Add &>(*arg).get_coef();
        if (neq(*zero, *s) and is_a<Integer>(*s)) {
            return false;
        }
    }
    return true;
}

RCP<const Basic> Ceiling::create(const RCP<const Basic> &arg) const
{
    return ceiling(arg);
}

RCP<const Basic> ceiling(const RCP<const Basic> &arg)
{
    if (is_a_Number(*arg)) {
        if (down_cast<const Number &>(*arg).is_exact()) {
            if (is_a<Rational>(*arg)) {
                const Rational &s = down_cast<const Rational &>(*arg);
                integer_class quotient;
                mp_cdiv_q(quotient, SymEngine::get_num(s.as_rational_class()),
                          SymEngine::get_den(s.as_rational_class()));
                return integer(std::move(quotient));
            }
            return arg;
        }
        RCP<const Number> _arg = rcp_static_cast<const Number>(arg);
        return _arg->get_eval().ceiling(*_arg);
    }
    if (is_a<Constant>(*arg)) {
        if (eq(*arg, *pi)) {
            return integer(4);
        }
        if (eq(*arg, *E)) {
            return integer(3);
        }
        if (eq(*arg, *GoldenRatio)) {
            return integer(2);
        }
        if (eq(*arg, *Catalan) or eq(*arg, *EulerGamma)) {
            return integer(1);
        }
    }
    if (is_a<Floor>(*arg)) {
        return arg;
    }
    if (is_a<Ceiling>(*arg)) {
        return arg;
    }
    if (is_a<Truncate>(*arg)) {
        return arg;
    }
    if (is_a<BooleanAtom>(*arg) or is_a_Relational(*arg)) {
        throw SymEngineException(
            "Boolean objects not allowed in this context.");
    }
    if (is_a<Add>(*arg)) {
        RCP<const Number> s = down_cast<const Add &>(*arg).get_coef();
        umap_basic_num d = down_cast<const Add &>(*arg).get_dict();
        if (is_a<Integer>(*s)) {
            return add(
                s, make_rcp<const Ceiling>(Add::from_dict(zero, std::move(d))));
        }
    }
    return make_rcp<const Ceiling>(arg);
}

Truncate::Truncate(const RCP<const Basic> &arg) : OneArgFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool Truncate::is_canonical(const RCP<const Basic> &arg) const
{
    if (is_a_Number(*arg)) {
        return false;
    }
    if (is_a<Constant>(*arg)) {
        return false;
    }
    if (is_a<Floor>(*arg)) {
        return false;
    }
    if (is_a<Ceiling>(*arg)) {
        return false;
    }
    if (is_a<Truncate>(*arg)) {
        return false;
    }
    if (is_a<BooleanAtom>(*arg) or is_a_Relational(*arg)) {
        return false;
    }
    if (is_a<Add>(*arg)) {
        RCP<const Number> s = down_cast<const Add &>(*arg).get_coef();
        if (neq(*zero, *s) and is_a<Integer>(*s)) {
            return false;
        }
    }
    return true;
}

RCP<const Basic> Truncate::create(const RCP<const Basic> &arg) const
{
    return truncate(arg);
}

RCP<const Basic> truncate(const RCP<const Basic> &arg)
{
    if (is_a_Number(*arg)) {
        if (down_cast<const Number &>(*arg).is_exact()) {
            if (is_a<Rational>(*arg)) {
                const Rational &s = down_cast<const Rational &>(*arg);
                integer_class quotient;
                mp_tdiv_q(quotient, SymEngine::get_num(s.as_rational_class()),
                          SymEngine::get_den(s.as_rational_class()));
                return integer(std::move(quotient));
            }
            return arg;
        }
        RCP<const Number> _arg = rcp_static_cast<const Number>(arg);
        return _arg->get_eval().truncate(*_arg);
    }
    if (is_a<Constant>(*arg)) {
        if (eq(*arg, *pi)) {
            return integer(3);
        }
        if (eq(*arg, *E)) {
            return integer(2);
        }
        if (eq(*arg, *GoldenRatio)) {
            return integer(1);
        }
        if (eq(*arg, *Catalan) or eq(*arg, *EulerGamma)) {
            return integer(0);
        }
    }
    if (is_a<Floor>(*arg)) {
        return arg;
    }
    if (is_a<Ceiling>(*arg)) {
        return arg;
    }
    if (is_a<Truncate>(*arg)) {
        return arg;
    }
    if (is_a<BooleanAtom>(*arg) or is_a_Relational(*arg)) {
        throw SymEngineException(
            "Boolean objects not allowed in this context.");
    }
    if (is_a<Add>(*arg)) {
        RCP<const Number> s = down_cast<const Add &>(*arg).get_coef();
        umap_basic_num d = down_cast<const Add &>(*arg).get_dict();
        if (is_a<Integer>(*s)) {
            return add(s, make_rcp<const Truncate>(
                              Add::from_dict(zero, std::move(d))));
        }
    }
    return make_rcp<const Truncate>(arg);
}

Sin::Sin(const RCP<const Basic> &arg) : TrigFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool Sin::is_canonical(const RCP<const Basic> &arg) const
{
    // e.g. sin(0)
    if (is_a<Integer>(*arg) and down_cast<const Integer &>(*arg).is_zero())
        return false;
    // e.g sin(7*pi/2+y)
    if (trig_has_basic_shift(arg)) {
        return false;
    }
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return false;
    }
    return true;
}

RCP<const Basic> sin(const RCP<const Basic> &arg)
{
    if (eq(*arg, *zero))
        return zero;
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return down_cast<const Number &>(*arg).get_eval().sin(*arg);
    }

    if (is_a<ASin>(*arg)) {
        return down_cast<const ASin &>(*arg).get_arg();
    } else if (is_a<ACsc>(*arg)) {
        return div(one, down_cast<const ACsc &>(*arg).get_arg());
    }

    RCP<const Basic> ret_arg;
    int index, sign;
    bool conjugate = trig_simplify(arg, 2, true, false,           // input
                                   outArg(ret_arg), index, sign); // output

    if (conjugate) {
        // cos has to be returned
        if (sign == 1) {
            return cos(ret_arg);
        } else {
            return mul(minus_one, cos(ret_arg));
        }
    } else {
        if (eq(*ret_arg, *zero)) {
            return mul(integer(sign), sin_table[index]);
        } else {
            // If ret_arg is the same as arg, a `Sin` instance is returned
            // Or else `sin` is called again.
            if (sign == 1) {
                if (neq(*ret_arg, *arg)) {
                    return sin(ret_arg);
                } else {
                    return make_rcp<const Sin>(arg);
                }
            } else {
                return mul(minus_one, sin(ret_arg));
            }
        }
    }
}

/* ---------------------------- */

Cos::Cos(const RCP<const Basic> &arg) : TrigFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool Cos::is_canonical(const RCP<const Basic> &arg) const
{
    // e.g. cos(0)
    if (is_a<Integer>(*arg) and down_cast<const Integer &>(*arg).is_zero())
        return false;
    // e.g cos(k*pi/2)
    if (trig_has_basic_shift(arg)) {
        return false;
    }
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return false;
    }
    return true;
}

RCP<const Basic> cos(const RCP<const Basic> &arg)
{
    if (eq(*arg, *zero))
        return one;
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return down_cast<const Number &>(*arg).get_eval().cos(*arg);
    }

    if (is_a<ACos>(*arg)) {
        return down_cast<const ACos &>(*arg).get_arg();
    } else if (is_a<ASec>(*arg)) {
        return div(one, down_cast<const ASec &>(*arg).get_arg());
    }

    RCP<const Basic> ret_arg;
    int index, sign;
    bool conjugate = trig_simplify(arg, 2, false, true,           // input
                                   outArg(ret_arg), index, sign); // output

    if (conjugate) {
        // sin has to be returned
        if (sign == 1) {
            return sin(ret_arg);
        } else {
            return mul(minus_one, sin(ret_arg));
        }
    } else {
        if (eq(*ret_arg, *zero)) {
            return mul(integer(sign), sin_table[(index + 6) % 24]);
        } else {
            if (sign == 1) {
                if (neq(*ret_arg, *arg)) {
                    return cos(ret_arg);
                } else {
                    return make_rcp<const Cos>(ret_arg);
                }
            } else {
                return mul(minus_one, cos(ret_arg));
            }
        }
    }
}

/* ---------------------------- */

Tan::Tan(const RCP<const Basic> &arg) : TrigFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool Tan::is_canonical(const RCP<const Basic> &arg) const
{
    if (is_a<Integer>(*arg) and down_cast<const Integer &>(*arg).is_zero())
        return false;
    // e.g tan(k*pi/2)
    if (trig_has_basic_shift(arg)) {
        return false;
    }
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return false;
    }
    return true;
}

RCP<const Basic> tan(const RCP<const Basic> &arg)
{
    if (eq(*arg, *zero))
        return zero;
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return down_cast<const Number &>(*arg).get_eval().tan(*arg);
    }

    if (is_a<ATan>(*arg)) {
        return down_cast<const ATan &>(*arg).get_arg();
    } else if (is_a<ACot>(*arg)) {
        return div(one, down_cast<const ACot &>(*arg).get_arg());
    }

    RCP<const Basic> ret_arg;
    int index, sign;
    bool conjugate = trig_simplify(arg, 1, true, true,            // input
                                   outArg(ret_arg), index, sign); // output

    if (conjugate) {
        // cot has to be returned
        if (sign == 1) {
            return cot(ret_arg);
        } else {
            return mul(minus_one, cot(ret_arg));
        }
    } else {
        if (eq(*ret_arg, *zero)) {
            return mul(integer(sign),
                       div(sin_table[index], sin_table[(index + 6) % 24]));
        } else {
            if (sign == 1) {
                if (neq(*ret_arg, *arg)) {
                    return tan(ret_arg);
                } else {
                    return make_rcp<const Tan>(ret_arg);
                }
            } else {
                return mul(minus_one, tan(ret_arg));
            }
        }
    }
}

/* ---------------------------- */

Cot::Cot(const RCP<const Basic> &arg) : TrigFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool Cot::is_canonical(const RCP<const Basic> &arg) const
{
    if (is_a<Integer>(*arg) and down_cast<const Integer &>(*arg).is_zero())
        return false;
    // e.g cot(k*pi/2)
    if (trig_has_basic_shift(arg)) {
        return false;
    }
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return false;
    }
    return true;
}

RCP<const Basic> cot(const RCP<const Basic> &arg)
{
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return down_cast<const Number &>(*arg).get_eval().cot(*arg);
    }

    if (is_a<ACot>(*arg)) {
        return down_cast<const ACot &>(*arg).get_arg();
    } else if (is_a<ATan>(*arg)) {
        return div(one, down_cast<const ATan &>(*arg).get_arg());
    }

    RCP<const Basic> ret_arg;
    int index, sign;
    bool conjugate = trig_simplify(arg, 1, true, true,            // input
                                   outArg(ret_arg), index, sign); // output

    if (conjugate) {
        // tan has to be returned
        if (sign == 1) {
            return tan(ret_arg);
        } else {
            return mul(minus_one, tan(ret_arg));
        }
    } else {
        if (eq(*ret_arg, *zero)) {
            return mul(integer(sign),
                       div(sin_table[(index + 6) % 24], sin_table[index]));
        } else {
            if (sign == 1) {
                if (neq(*ret_arg, *arg)) {
                    return cot(ret_arg);
                } else {
                    return make_rcp<const Cot>(ret_arg);
                }
            } else {
                return mul(minus_one, cot(ret_arg));
            }
        }
    }
}

/* ---------------------------- */

Csc::Csc(const RCP<const Basic> &arg) : TrigFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool Csc::is_canonical(const RCP<const Basic> &arg) const
{
    // e.g. Csc(0)
    if (is_a<Integer>(*arg) and down_cast<const Integer &>(*arg).is_zero())
        return false;
    // e.g csc(k*pi/2)
    if (trig_has_basic_shift(arg)) {
        return false;
    }
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return false;
    }
    return true;
}

RCP<const Basic> csc(const RCP<const Basic> &arg)
{
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return down_cast<const Number &>(*arg).get_eval().csc(*arg);
    }

    if (is_a<ACsc>(*arg)) {
        return down_cast<const ACsc &>(*arg).get_arg();
    } else if (is_a<ASin>(*arg)) {
        return div(one, down_cast<const ASin &>(*arg).get_arg());
    }

    RCP<const Basic> ret_arg;
    int index, sign;
    bool conjugate = trig_simplify(arg, 2, true, false,           // input
                                   outArg(ret_arg), index, sign); // output

    if (conjugate) {
        // cos has to be returned
        if (sign == 1) {
            return sec(ret_arg);
        } else {
            return mul(minus_one, sec(ret_arg));
        }
    } else {
        if (eq(*ret_arg, *zero)) {
            return mul(integer(sign), div(one, sin_table[index]));
        } else {
            if (sign == 1) {
                if (neq(*ret_arg, *arg)) {
                    return csc(ret_arg);
                } else {
                    return make_rcp<const Csc>(ret_arg);
                }
            } else {
                return mul(minus_one, csc(ret_arg));
            }
        }
    }
}

/* ---------------------------- */

Sec::Sec(const RCP<const Basic> &arg) : TrigFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool Sec::is_canonical(const RCP<const Basic> &arg) const
{
    // e.g. Sec(0)
    if (is_a<Integer>(*arg) and down_cast<const Integer &>(*arg).is_zero())
        return false;
    // e.g sec(k*pi/2)
    if (trig_has_basic_shift(arg)) {
        return false;
    }
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return false;
    }
    return true;
}

RCP<const Basic> sec(const RCP<const Basic> &arg)
{
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return down_cast<const Number &>(*arg).get_eval().sec(*arg);
    }

    if (is_a<ASec>(*arg)) {
        return down_cast<const ASec &>(*arg).get_arg();
    } else if (is_a<ACos>(*arg)) {
        return div(one, down_cast<const ACos &>(*arg).get_arg());
    }

    RCP<const Basic> ret_arg;
    int index, sign;
    bool conjugate = trig_simplify(arg, 2, false, true,           // input
                                   outArg(ret_arg), index, sign); // output

    if (conjugate) {
        // csc has to be returned
        if (sign == 1) {
            return csc(ret_arg);
        } else {
            return mul(minus_one, csc(ret_arg));
        }
    } else {
        if (eq(*ret_arg, *zero)) {
            return mul(integer(sign), div(one, sin_table[(index + 6) % 24]));
        } else {
            if (sign == 1) {
                if (neq(*ret_arg, *arg)) {
                    return sec(ret_arg);
                } else {
                    return make_rcp<const Sec>(ret_arg);
                }
            } else {
                return mul(minus_one, sec(ret_arg));
            }
        }
    }
}
/* ---------------------------- */

// simplifies trigonometric functions wherever possible
// currently deals with simplifications of type sin(acos())
RCP<const Basic> trig_to_sqrt(const RCP<const Basic> &arg)
{
    RCP<const Basic> i_arg;

    if (is_a<Sin>(*arg)) {
        if (is_a<ACos>(*arg->get_args()[0])) {
            i_arg = down_cast<const ACos &>(*(arg->get_args()[0])).get_arg();
            return sqrt(sub(one, pow(i_arg, i2)));
        } else if (is_a<ATan>(*arg->get_args()[0])) {
            i_arg = down_cast<const ATan &>(*(arg->get_args()[0])).get_arg();
            return div(i_arg, sqrt(add(one, pow(i_arg, i2))));
        } else if (is_a<ASec>(*arg->get_args()[0])) {
            i_arg = down_cast<const ASec &>(*(arg->get_args()[0])).get_arg();
            return sqrt(sub(one, pow(i_arg, im2)));
        } else if (is_a<ACot>(*arg->get_args()[0])) {
            i_arg = down_cast<const ACot &>(*(arg->get_args()[0])).get_arg();
            return div(one, mul(i_arg, sqrt(add(one, pow(i_arg, im2)))));
        }
    } else if (is_a<Cos>(*arg)) {
        if (is_a<ASin>(*arg->get_args()[0])) {
            i_arg = down_cast<const ASin &>(*(arg->get_args()[0])).get_arg();
            return sqrt(sub(one, pow(i_arg, i2)));
        } else if (is_a<ATan>(*arg->get_args()[0])) {
            i_arg = down_cast<const ATan &>(*(arg->get_args()[0])).get_arg();
            return div(one, sqrt(add(one, pow(i_arg, i2))));
        } else if (is_a<ACsc>(*arg->get_args()[0])) {
            i_arg = down_cast<const ACsc &>(*(arg->get_args()[0])).get_arg();
            return sqrt(sub(one, pow(i_arg, im2)));
        } else if (is_a<ACot>(*arg->get_args()[0])) {
            i_arg = down_cast<const ACot &>(*(arg->get_args()[0])).get_arg();
            return div(one, sqrt(add(one, pow(i_arg, im2))));
        }
    } else if (is_a<Tan>(*arg)) {
        if (is_a<ASin>(*arg->get_args()[0])) {
            i_arg = down_cast<const ASin &>(*(arg->get_args()[0])).get_arg();
            return div(i_arg, sqrt(sub(one, pow(i_arg, i2))));
        } else if (is_a<ACos>(*arg->get_args()[0])) {
            i_arg = down_cast<const ACos &>(*(arg->get_args()[0])).get_arg();
            return div(sqrt(sub(one, pow(i_arg, i2))), i_arg);
        } else if (is_a<ACsc>(*arg->get_args()[0])) {
            i_arg = down_cast<const ACsc &>(*(arg->get_args()[0])).get_arg();
            return div(one, mul(i_arg, sqrt(sub(one, pow(i_arg, im2)))));
        } else if (is_a<ASec>(*arg->get_args()[0])) {
            i_arg = down_cast<const ASec &>(*(arg->get_args()[0])).get_arg();
            return mul(i_arg, sqrt(sub(one, pow(i_arg, im2))));
        }
    } else if (is_a<Csc>(*arg)) {
        if (is_a<ACos>(*arg->get_args()[0])) {
            i_arg = down_cast<const ACos &>(*(arg->get_args()[0])).get_arg();
            return div(one, sqrt(sub(one, pow(i_arg, i2))));
        } else if (is_a<ATan>(*arg->get_args()[0])) {
            i_arg = down_cast<const ATan &>(*(arg->get_args()[0])).get_arg();
            return div(sqrt(add(one, pow(i_arg, i2))), i_arg);
        } else if (is_a<ASec>(*arg->get_args()[0])) {
            i_arg = down_cast<const ASec &>(*(arg->get_args()[0])).get_arg();
            return div(one, sqrt(sub(one, pow(i_arg, im2))));
        } else if (is_a<ACot>(*arg->get_args()[0])) {
            i_arg = down_cast<const ACot &>(*(arg->get_args()[0])).get_arg();
            return mul(i_arg, sqrt(add(one, pow(i_arg, im2))));
        }
    } else if (is_a<Sec>(*arg)) {
        if (is_a<ASin>(*arg->get_args()[0])) {
            i_arg = down_cast<const ASin &>(*(arg->get_args()[0])).get_arg();
            return div(one, sqrt(sub(one, pow(i_arg, i2))));
        } else if (is_a<ATan>(*arg->get_args()[0])) {
            i_arg = down_cast<const ATan &>(*(arg->get_args()[0])).get_arg();
            return sqrt(add(one, pow(i_arg, i2)));
        } else if (is_a<ACsc>(*arg->get_args()[0])) {
            i_arg = down_cast<const ACsc &>(*(arg->get_args()[0])).get_arg();
            return div(one, sqrt(sub(one, pow(i_arg, im2))));
        } else if (is_a<ACot>(*arg->get_args()[0])) {
            i_arg = down_cast<const ACot &>(*(arg->get_args()[0])).get_arg();
            return sqrt(add(one, pow(i_arg, im2)));
        }
    } else if (is_a<Cot>(*arg)) {
        if (is_a<ASin>(*arg->get_args()[0])) {
            i_arg = down_cast<const ASin &>(*(arg->get_args()[0])).get_arg();
            return div(sqrt(sub(one, pow(i_arg, i2))), i_arg);
        } else if (is_a<ACos>(*arg->get_args()[0])) {
            i_arg = down_cast<const ACos &>(*(arg->get_args()[0])).get_arg();
            return div(i_arg, sqrt(sub(one, pow(i_arg, i2))));
        } else if (is_a<ACsc>(*arg->get_args()[0])) {
            i_arg = down_cast<const ACsc &>(*(arg->get_args()[0])).get_arg();
            return mul(i_arg, sqrt(sub(one, pow(i_arg, im2))));
        } else if (is_a<ASec>(*arg->get_args()[0])) {
            i_arg = down_cast<const ASec &>(*(arg->get_args()[0])).get_arg();
            return div(one, mul(i_arg, sqrt(sub(one, pow(i_arg, im2)))));
        }
    }

    return arg;
}

/* ---------------------------- */
ASin::ASin(const RCP<const Basic> &arg) : InverseTrigFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool ASin::is_canonical(const RCP<const Basic> &arg) const
{
    if (eq(*arg, *zero) or eq(*arg, *one) or eq(*arg, *minus_one))
        return false;
    RCP<const Basic> index;
    if (inverse_lookup(inverse_cst, get_arg(), outArg(index))) {
        return false;
    }
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return false;
    }
    return true;
}

RCP<const Basic> asin(const RCP<const Basic> &arg)
{
    if (eq(*arg, *zero))
        return zero;
    else if (eq(*arg, *one))
        return div(pi, i2);
    else if (eq(*arg, *minus_one))
        return mul(minus_one, div(pi, i2));
    else if (is_a_Number(*arg)
             and not down_cast<const Number &>(*arg).is_exact()) {
        return down_cast<const Number &>(*arg).get_eval().asin(*arg);
    }

    RCP<const Basic> index;
    bool b = inverse_lookup(inverse_cst, arg, outArg(index));
    if (b) {
        return div(pi, index);
    } else {
        return make_rcp<const ASin>(arg);
    }
}

ACos::ACos(const RCP<const Basic> &arg) : InverseTrigFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool ACos::is_canonical(const RCP<const Basic> &arg) const
{
    if (eq(*arg, *zero) or eq(*arg, *one) or eq(*arg, *minus_one))
        return false;
    RCP<const Basic> index;
    if (inverse_lookup(inverse_cst, get_arg(), outArg(index))) {
        return false;
    }
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return false;
    }
    return true;
}

RCP<const Basic> acos(const RCP<const Basic> &arg)
{
    if (eq(*arg, *zero))
        return div(pi, i2);
    else if (eq(*arg, *one))
        return zero;
    else if (eq(*arg, *minus_one))
        return pi;
    else if (is_a_Number(*arg)
             and not down_cast<const Number &>(*arg).is_exact()) {
        return down_cast<const Number &>(*arg).get_eval().acos(*arg);
    }

    RCP<const Basic> index;
    bool b = inverse_lookup(inverse_cst, arg, outArg(index));
    if (b) {
        return sub(div(pi, i2), div(pi, index));
    } else {
        return make_rcp<const ACos>(arg);
    }
}

ASec::ASec(const RCP<const Basic> &arg) : InverseTrigFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool ASec::is_canonical(const RCP<const Basic> &arg) const
{
    if (eq(*arg, *one) or eq(*arg, *minus_one))
        return false;
    RCP<const Basic> index;
    if (inverse_lookup(inverse_cst, div(one, get_arg()), outArg(index))) {
        return false;
    } else if (is_a_Number(*arg)
               and not down_cast<const Number &>(*arg).is_exact()) {
        return false;
    }
    return true;
}

RCP<const Basic> asec(const RCP<const Basic> &arg)
{
    if (eq(*arg, *one))
        return zero;
    else if (eq(*arg, *minus_one))
        return pi;
    else if (is_a_Number(*arg)
             and not down_cast<const Number &>(*arg).is_exact()) {
        return down_cast<const Number &>(*arg).get_eval().asec(*arg);
    }

    RCP<const Basic> index;
    bool b = inverse_lookup(inverse_cst, div(one, arg), outArg(index));
    if (b) {
        return sub(div(pi, i2), div(pi, index));
    } else {
        return make_rcp<const ASec>(arg);
    }
}

ACsc::ACsc(const RCP<const Basic> &arg) : InverseTrigFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool ACsc::is_canonical(const RCP<const Basic> &arg) const
{
    if (eq(*arg, *one) or eq(*arg, *minus_one))
        return false;
    RCP<const Basic> index;
    if (inverse_lookup(inverse_cst, div(one, arg), outArg(index))) {
        return false;
    }
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return false;
    }
    return true;
}

RCP<const Basic> acsc(const RCP<const Basic> &arg)
{
    if (eq(*arg, *one))
        return div(pi, i2);
    else if (eq(*arg, *minus_one))
        return div(pi, im2);
    else if (is_a_Number(*arg)
             and not down_cast<const Number &>(*arg).is_exact()) {
        return down_cast<const Number &>(*arg).get_eval().acsc(*arg);
    }

    RCP<const Basic> index;
    bool b = inverse_lookup(inverse_cst, div(one, arg), outArg(index));
    if (b) {
        return div(pi, index);
    } else {
        return make_rcp<const ACsc>(arg);
    }
}

ATan::ATan(const RCP<const Basic> &arg) : InverseTrigFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool ATan::is_canonical(const RCP<const Basic> &arg) const
{
    if (eq(*arg, *zero) or eq(*arg, *one) or eq(*arg, *minus_one))
        return false;
    RCP<const Basic> index;
    if (inverse_lookup(inverse_tct, get_arg(), outArg(index))) {
        return false;
    }
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return false;
    }
    return true;
}

RCP<const Basic> atan(const RCP<const Basic> &arg)
{
    if (eq(*arg, *zero))
        return zero;
    else if (eq(*arg, *one))
        return div(pi, mul(i2, i2));
    else if (eq(*arg, *minus_one))
        return mul(minus_one, div(pi, mul(i2, i2)));
    else if (is_a_Number(*arg)
             and not down_cast<const Number &>(*arg).is_exact()) {
        return down_cast<const Number &>(*arg).get_eval().atan(*arg);
    }

    RCP<const Basic> index;
    bool b = inverse_lookup(inverse_tct, arg, outArg(index));
    if (b) {
        return div(pi, index);
    } else {
        return make_rcp<const ATan>(arg);
    }
}

ACot::ACot(const RCP<const Basic> &arg) : InverseTrigFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool ACot::is_canonical(const RCP<const Basic> &arg) const
{
    if (eq(*arg, *zero) or eq(*arg, *one) or eq(*arg, *minus_one))
        return false;
    RCP<const Basic> index;
    if (inverse_lookup(inverse_tct, arg, outArg(index))) {
        return false;
    }
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return false;
    }
    return true;
}

RCP<const Basic> acot(const RCP<const Basic> &arg)
{
    if (eq(*arg, *zero))
        return div(pi, i2);
    else if (eq(*arg, *one))
        return div(pi, mul(i2, i2));
    else if (eq(*arg, *minus_one))
        return mul(i3, div(pi, mul(i2, i2)));
    else if (is_a_Number(*arg)
             and not down_cast<const Number &>(*arg).is_exact()) {
        return down_cast<const Number &>(*arg).get_eval().acot(*arg);
    }

    RCP<const Basic> index;
    bool b = inverse_lookup(inverse_tct, arg, outArg(index));
    if (b) {
        return sub(div(pi, i2), div(pi, index));
    } else {
        return make_rcp<const ACot>(arg);
    }
}

ATan2::ATan2(const RCP<const Basic> &num, const RCP<const Basic> &den)
    : TwoArgFunction(num, den)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(num, den))
}

bool ATan2::is_canonical(const RCP<const Basic> &num,
                         const RCP<const Basic> &den) const
{
    if (eq(*num, *zero) or eq(*num, *den) or eq(*num, *mul(minus_one, den)))
        return false;
    RCP<const Basic> index;
    bool b = inverse_lookup(inverse_tct, div(num, den), outArg(index));
    if (b)
        return false;
    else
        return true;
}

RCP<const Basic> ATan2::create(const RCP<const Basic> &a,
                               const RCP<const Basic> &b) const
{
    return atan2(a, b);
}

RCP<const Basic> atan2(const RCP<const Basic> &num, const RCP<const Basic> &den)
{
    if (eq(*num, *zero)) {
        if (is_a_Number(*den)) {
            RCP<const Number> den_new = rcp_static_cast<const Number>(den);
            if (den_new->is_negative())
                return pi;
            else if (den_new->is_positive())
                return zero;
            else {
                return Nan;
            }
        }
    } else if (eq(*den, *zero)) {
        if (is_a_Number(*num)) {
            RCP<const Number> num_new = rcp_static_cast<const Number>(num);
            if (num_new->is_negative())
                return div(pi, im2);
            else
                return div(pi, i2);
        }
    }
    RCP<const Basic> index;
    bool b = inverse_lookup(inverse_tct, div(num, den), outArg(index));
    if (b) {
        // Ideally the answer should depend on the signs of `num` and `den`
        // Currently is_positive() and is_negative() is not implemented for
        // types other than `Number`
        // Hence this will give exact answers in case when num and den are
        // numbers in SymEngine sense and when num and den are positive.
        // for the remaining cases in which we just return the value from
        // the lookup table.
        // TODO: update once is_positive() and is_negative() is implemented
        // in `Basic`
        if (is_a_Number(*den) and is_a_Number(*num)) {
            RCP<const Number> den_new = rcp_static_cast<const Number>(den);
            RCP<const Number> num_new = rcp_static_cast<const Number>(num);

            if (den_new->is_positive()) {
                return div(pi, index);
            } else if (den_new->is_negative()) {
                if (num_new->is_negative()) {
                    return sub(div(pi, index), pi);
                } else {
                    return add(div(pi, index), pi);
                }
            } else {
                return div(pi, index);
            }
        } else {
            return div(pi, index);
        }
    } else {
        return make_rcp<const ATan2>(num, den);
    }
}

/* ---------------------------- */

RCP<const Basic> Sin::create(const RCP<const Basic> &arg) const
{
    return sin(arg);
}

RCP<const Basic> Cos::create(const RCP<const Basic> &arg) const
{
    return cos(arg);
}

RCP<const Basic> Tan::create(const RCP<const Basic> &arg) const
{
    return tan(arg);
}

RCP<const Basic> Cot::create(const RCP<const Basic> &arg) const
{
    return cot(arg);
}

RCP<const Basic> Sec::create(const RCP<const Basic> &arg) const
{
    return sec(arg);
}

RCP<const Basic> Csc::create(const RCP<const Basic> &arg) const
{
    return csc(arg);
}

RCP<const Basic> ASin::create(const RCP<const Basic> &arg) const
{
    return asin(arg);
}

RCP<const Basic> ACos::create(const RCP<const Basic> &arg) const
{
    return acos(arg);
}

RCP<const Basic> ATan::create(const RCP<const Basic> &arg) const
{
    return atan(arg);
}

RCP<const Basic> ACot::create(const RCP<const Basic> &arg) const
{
    return acot(arg);
}

RCP<const Basic> ASec::create(const RCP<const Basic> &arg) const
{
    return asec(arg);
}

RCP<const Basic> ACsc::create(const RCP<const Basic> &arg) const
{
    return acsc(arg);
}

/* ---------------------------- */

Log::Log(const RCP<const Basic> &arg) : OneArgFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool Log::is_canonical(const RCP<const Basic> &arg) const
{
    //  log(0)
    if (is_a<Integer>(*arg) and down_cast<const Integer &>(*arg).is_zero())
        return false;
    //  log(1)
    if (is_a<Integer>(*arg) and down_cast<const Integer &>(*arg).is_one())
        return false;
    // log(E)
    if (eq(*arg, *E))
        return false;

    if (is_a_Number(*arg) and down_cast<const Number &>(*arg).is_negative())
        return false;

    // log(Inf) is also handled here.
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact())
        return false;

    // log(3I) should be expanded to log(3) + I*pi/2
    if (is_a<Complex>(*arg) and down_cast<const Complex &>(*arg).is_re_zero())
        return false;
    // log(num/den) = log(num) - log(den)
    if (is_a<Rational>(*arg))
        return false;
    return true;
}

RCP<const Basic> Log::create(const RCP<const Basic> &a) const
{
    return log(a);
}

RCP<const Basic> log(const RCP<const Basic> &arg)
{
    if (eq(*arg, *zero))
        return ComplexInf;
    if (eq(*arg, *one))
        return zero;
    if (eq(*arg, *E))
        return one;

    if (is_a_Number(*arg)) {
        RCP<const Number> _arg = rcp_static_cast<const Number>(arg);
        if (not _arg->is_exact()) {
            return _arg->get_eval().log(*_arg);
        } else if (_arg->is_negative()) {
            return add(log(mul(minus_one, _arg)), mul(pi, I));
        }
    }

    if (is_a<Rational>(*arg)) {
        RCP<const Integer> num, den;
        get_num_den(down_cast<const Rational &>(*arg), outArg(num),
                    outArg(den));
        return sub(log(num), log(den));
    }

    if (is_a<Complex>(*arg)) {
        RCP<const Complex> _arg = rcp_static_cast<const Complex>(arg);
        if (_arg->is_re_zero()) {
            RCP<const Number> arg_img = _arg->imaginary_part();
            if (arg_img->is_negative()) {
                return sub(log(mul(minus_one, arg_img)),
                           mul(I, div(pi, integer(2))));
            } else if (arg_img->is_zero()) {
                return ComplexInf;
            } else if (arg_img->is_positive()) {
                return add(log(arg_img), mul(I, div(pi, integer(2))));
            }
        }
    }

    return make_rcp<const Log>(arg);
}

RCP<const Basic> log(const RCP<const Basic> &arg, const RCP<const Basic> &base)
{
    return div(log(arg), log(base));
}

LambertW::LambertW(const RCP<const Basic> &arg) : OneArgFunction{arg}
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool LambertW::is_canonical(const RCP<const Basic> &arg) const
{
    if (eq(*arg, *zero))
        return false;
    if (eq(*arg, *E))
        return false;
    if (eq(*arg, *div(neg(one), E)))
        return false;
    if (eq(*arg, *div(log(i2), im2)))
        return false;
    return true;
}

RCP<const Basic> LambertW::create(const RCP<const Basic> &arg) const
{
    return lambertw(arg);
}

RCP<const Basic> lambertw(const RCP<const Basic> &arg)
{
    if (eq(*arg, *zero))
        return zero;
    if (eq(*arg, *E))
        return one;
    if (eq(*arg, *div(neg(one), E)))
        return minus_one;
    if (eq(*arg, *div(log(i2), im2)))
        return mul(minus_one, log(i2));
    return make_rcp<const LambertW>(arg);
}

FunctionSymbol::FunctionSymbol(std::string name, const RCP<const Basic> &arg)
    : MultiArgFunction({arg}), name_{name}
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(get_vec()))
}

FunctionSymbol::FunctionSymbol(std::string name, const vec_basic &arg)
    : MultiArgFunction(arg), name_{name}
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(get_vec()))
}

bool FunctionSymbol::is_canonical(const vec_basic &arg) const
{
    return true;
}

hash_t FunctionSymbol::__hash__() const
{
    hash_t seed = SYMENGINE_FUNCTIONSYMBOL;
    for (const auto &a : get_vec())
        hash_combine<Basic>(seed, *a);
    hash_combine<std::string>(seed, name_);
    return seed;
}

bool FunctionSymbol::__eq__(const Basic &o) const
{
    if (is_a<FunctionSymbol>(o)
        and name_ == down_cast<const FunctionSymbol &>(o).name_
        and unified_eq(get_vec(),
                       down_cast<const FunctionSymbol &>(o).get_vec()))
        return true;
    return false;
}

int FunctionSymbol::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<FunctionSymbol>(o))
    const FunctionSymbol &s = down_cast<const FunctionSymbol &>(o);
    if (name_ == s.name_)
        return unified_compare(get_vec(), s.get_vec());
    else
        return name_ < s.name_ ? -1 : 1;
}

RCP<const Basic> FunctionSymbol::create(const vec_basic &x) const
{
    return make_rcp<const FunctionSymbol>(name_, x);
}

RCP<const Basic> function_symbol(std::string name, const vec_basic &arg)
{
    return make_rcp<const FunctionSymbol>(name, arg);
}

RCP<const Basic> function_symbol(std::string name, const RCP<const Basic> &arg)
{
    return make_rcp<const FunctionSymbol>(name, arg);
}

FunctionWrapper::FunctionWrapper(std::string name, const RCP<const Basic> &arg)
    : FunctionSymbol(name, arg)
{
    SYMENGINE_ASSIGN_TYPEID()
}

FunctionWrapper::FunctionWrapper(std::string name, const vec_basic &vec)
    : FunctionSymbol(name, vec)
{
    SYMENGINE_ASSIGN_TYPEID()
}

/* ---------------------------- */

Derivative::Derivative(const RCP<const Basic> &arg, const multiset_basic &x)
    : arg_{arg}, x_{x}
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg, x))
}

bool Derivative::is_canonical(const RCP<const Basic> &arg,
                              const multiset_basic &x) const
{
    // Check that 'x' are Symbols:
    for (const auto &a : x)
        if (not is_a<Symbol>(*a))
            return false;
    if (is_a<FunctionSymbol>(*arg) or is_a<LeviCivita>(*arg)) {
        for (auto &p : x) {
            RCP<const Symbol> s = rcp_static_cast<const Symbol>(p);
            RCP<const MultiArgFunction> f
                = rcp_static_cast<const MultiArgFunction>(arg);
            bool found_s = false;
            // 's' should be one of the args of the function
            // and should not appear anywhere else.
            for (const auto &a : f->get_args()) {
                if (eq(*a, *s)) {
                    if (found_s) {
                        return false;
                    } else {
                        found_s = true;
                    }
                } else if (neq(*a->diff(s), *zero)) {
                    return false;
                }
            }
            if (!found_s) {
                return false;
            }
        }
        return true;
    } else if (is_a<Abs>(*arg)) {
        return true;
    } else if (is_a<FunctionWrapper>(*arg)) {
        return true;
    } else if (is_a<PolyGamma>(*arg) or is_a<Zeta>(*arg)
               or is_a<UpperGamma>(*arg) or is_a<LowerGamma>(*arg)
               or is_a<Dirichlet_eta>(*arg)) {
        bool found = false;
        auto v = arg->get_args();
        for (auto &p : x) {
            if (has_symbol(*v[0], *rcp_static_cast<const Symbol>(p))) {
                found = true;
                break;
            }
        }
        return found;
    } else if (is_a<KroneckerDelta>(*arg)) {
        bool found = false;
        auto v = arg->get_args();
        for (auto &p : x) {
            if (has_symbol(*v[0], *rcp_static_cast<const Symbol>(p))
                or has_symbol(*v[1], *rcp_static_cast<const Symbol>(p))) {
                found = true;
                break;
            }
        }
        return found;
    }
    return false;
}

hash_t Derivative::__hash__() const
{
    hash_t seed = SYMENGINE_DERIVATIVE;
    hash_combine<Basic>(seed, *arg_);
    for (auto &p : x_) {
        hash_combine<Basic>(seed, *p);
    }
    return seed;
}

bool Derivative::__eq__(const Basic &o) const
{
    if (is_a<Derivative>(o)
        and eq(*arg_, *(down_cast<const Derivative &>(o).arg_))
        and unified_eq(x_, down_cast<const Derivative &>(o).x_))
        return true;
    return false;
}

int Derivative::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<Derivative>(o))
    const Derivative &s = down_cast<const Derivative &>(o);
    int cmp = arg_->__cmp__(*(s.arg_));
    if (cmp != 0)
        return cmp;
    cmp = unified_compare(x_, s.x_);
    return cmp;
}

// Subs class
Subs::Subs(const RCP<const Basic> &arg, const map_basic_basic &dict)
    : arg_{arg}, dict_{dict}
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg, dict))
}

bool Subs::is_canonical(const RCP<const Basic> &arg,
                        const map_basic_basic &dict) const
{
    if (is_a<Derivative>(*arg)) {
        return true;
    }
    return false;
}

hash_t Subs::__hash__() const
{
    hash_t seed = SYMENGINE_SUBS;
    hash_combine<Basic>(seed, *arg_);
    for (const auto &p : dict_) {
        hash_combine<Basic>(seed, *p.first);
        hash_combine<Basic>(seed, *p.second);
    }
    return seed;
}

bool Subs::__eq__(const Basic &o) const
{
    if (is_a<Subs>(o) and eq(*arg_, *(down_cast<const Subs &>(o).arg_))
        and unified_eq(dict_, down_cast<const Subs &>(o).dict_))
        return true;
    return false;
}

int Subs::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<Subs>(o))
    const Subs &s = down_cast<const Subs &>(o);
    int cmp = arg_->__cmp__(*(s.arg_));
    if (cmp != 0)
        return cmp;
    cmp = unified_compare(dict_, s.dict_);
    return cmp;
}

vec_basic Subs::get_variables() const
{
    vec_basic v;
    for (const auto &p : dict_) {
        v.push_back(p.first);
    }
    return v;
}

vec_basic Subs::get_point() const
{
    vec_basic v;
    for (const auto &p : dict_) {
        v.push_back(p.second);
    }
    return v;
}

vec_basic Subs::get_args() const
{
    vec_basic v = {arg_};
    for (const auto &p : dict_) {
        v.push_back(p.first);
    }
    for (const auto &p : dict_) {
        v.push_back(p.second);
    }
    return v;
}

Sinh::Sinh(const RCP<const Basic> &arg) : HyperbolicFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool Sinh::is_canonical(const RCP<const Basic> &arg) const
{
    if (eq(*arg, *zero))
        return false;
    if (is_a_Number(*arg)) {
        if (down_cast<const Number &>(*arg).is_negative()) {
            return false;
        } else if (not down_cast<const Number &>(*arg).is_exact()) {
            return false;
        }
    }
    if (could_extract_minus(*arg))
        return false;
    return true;
}

RCP<const Basic> sinh(const RCP<const Basic> &arg)
{
    if (eq(*arg, *zero))
        return zero;
    if (is_a_Number(*arg)) {
        RCP<const Number> _arg = rcp_static_cast<const Number>(arg);
        if (not _arg->is_exact()) {
            return _arg->get_eval().sinh(*_arg);
        } else if (_arg->is_negative()) {
            return neg(sinh(zero->sub(*_arg)));
        }
    }
    RCP<const Basic> d;
    bool b = handle_minus(arg, outArg(d));
    if (b) {
        return neg(sinh(d));
    }
    return make_rcp<const Sinh>(d);
}

Csch::Csch(const RCP<const Basic> &arg) : HyperbolicFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool Csch::is_canonical(const RCP<const Basic> &arg) const
{
    if (eq(*arg, *zero))
        return false;
    if (is_a_Number(*arg)) {
        if (down_cast<const Number &>(*arg).is_negative()) {
            return false;
        } else if (not down_cast<const Number &>(*arg).is_exact()) {
            return false;
        }
    }
    if (could_extract_minus(*arg))
        return false;
    return true;
}

RCP<const Basic> csch(const RCP<const Basic> &arg)
{
    if (eq(*arg, *zero)) {
        return ComplexInf;
    }
    if (is_a_Number(*arg)) {
        RCP<const Number> _arg = rcp_static_cast<const Number>(arg);
        if (not _arg->is_exact()) {
            return _arg->get_eval().csch(*_arg);
        } else if (_arg->is_negative()) {
            return neg(csch(zero->sub(*_arg)));
        }
    }
    RCP<const Basic> d;
    bool b = handle_minus(arg, outArg(d));
    if (b) {
        return neg(csch(d));
    }
    return make_rcp<const Csch>(d);
}

Cosh::Cosh(const RCP<const Basic> &arg) : HyperbolicFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool Cosh::is_canonical(const RCP<const Basic> &arg) const
{
    if (eq(*arg, *zero))
        return false;
    if (is_a_Number(*arg)) {
        if (down_cast<const Number &>(*arg).is_negative()) {
            return false;
        } else if (not down_cast<const Number &>(*arg).is_exact()) {
            return false;
        }
    }
    if (could_extract_minus(*arg))
        return false;
    return true;
}

RCP<const Basic> cosh(const RCP<const Basic> &arg)
{
    if (eq(*arg, *zero))
        return one;
    if (is_a_Number(*arg)) {
        RCP<const Number> _arg = rcp_static_cast<const Number>(arg);
        if (not _arg->is_exact()) {
            return _arg->get_eval().cosh(*_arg);
        } else if (_arg->is_negative()) {
            return cosh(zero->sub(*_arg));
        }
    }
    RCP<const Basic> d;
    handle_minus(arg, outArg(d));
    return make_rcp<const Cosh>(d);
}

Sech::Sech(const RCP<const Basic> &arg) : HyperbolicFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool Sech::is_canonical(const RCP<const Basic> &arg) const
{
    if (eq(*arg, *zero))
        return false;
    if (is_a_Number(*arg)) {
        if (down_cast<const Number &>(*arg).is_negative()) {
            return false;
        } else if (not down_cast<const Number &>(*arg).is_exact()) {
            return false;
        }
    }
    if (could_extract_minus(*arg))
        return false;
    return true;
}

RCP<const Basic> sech(const RCP<const Basic> &arg)
{
    if (eq(*arg, *zero))
        return one;
    if (is_a_Number(*arg)) {
        RCP<const Number> _arg = rcp_static_cast<const Number>(arg);
        if (not _arg->is_exact()) {
            return _arg->get_eval().sech(*_arg);
        } else if (_arg->is_negative()) {
            return sech(zero->sub(*_arg));
        }
    }
    RCP<const Basic> d;
    handle_minus(arg, outArg(d));
    return make_rcp<const Sech>(d);
}

Tanh::Tanh(const RCP<const Basic> &arg) : HyperbolicFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool Tanh::is_canonical(const RCP<const Basic> &arg) const
{
    if (eq(*arg, *zero))
        return false;
    if (is_a_Number(*arg)) {
        if (down_cast<const Number &>(*arg).is_negative()) {
            return false;
        } else if (not down_cast<const Number &>(*arg).is_exact()) {
            return false;
        }
    }
    if (could_extract_minus(*arg))
        return false;
    return true;
}

RCP<const Basic> tanh(const RCP<const Basic> &arg)
{
    if (eq(*arg, *zero))
        return zero;
    if (is_a_Number(*arg)) {
        RCP<const Number> _arg = rcp_static_cast<const Number>(arg);
        if (not _arg->is_exact()) {
            return _arg->get_eval().tanh(*_arg);
        } else if (_arg->is_negative()) {
            return neg(tanh(zero->sub(*_arg)));
        }
    }

    RCP<const Basic> d;
    bool b = handle_minus(arg, outArg(d));
    if (b) {
        return neg(tanh(d));
    }
    return make_rcp<const Tanh>(d);
}

Coth::Coth(const RCP<const Basic> &arg) : HyperbolicFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool Coth::is_canonical(const RCP<const Basic> &arg) const
{
    if (eq(*arg, *zero))
        return false;
    if (is_a_Number(*arg)) {
        if (down_cast<const Number &>(*arg).is_negative()) {
            return false;
        } else if (not down_cast<const Number &>(*arg).is_exact()) {
            return false;
        }
    }
    if (could_extract_minus(*arg))
        return false;
    return true;
}

RCP<const Basic> coth(const RCP<const Basic> &arg)
{
    if (eq(*arg, *zero)) {
        return ComplexInf;
    }
    if (is_a_Number(*arg)) {
        RCP<const Number> _arg = rcp_static_cast<const Number>(arg);
        if (not _arg->is_exact()) {
            return _arg->get_eval().coth(*_arg);
        } else if (_arg->is_negative()) {
            return neg(coth(zero->sub(*_arg)));
        }
    }
    RCP<const Basic> d;
    bool b = handle_minus(arg, outArg(d));
    if (b) {
        return neg(coth(d));
    }
    return make_rcp<const Coth>(d);
}

ASinh::ASinh(const RCP<const Basic> &arg) : InverseHyperbolicFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool ASinh::is_canonical(const RCP<const Basic> &arg) const
{
    if (eq(*arg, *zero) or eq(*arg, *one) or eq(*arg, *minus_one))
        return false;
    if (is_a_Number(*arg)) {
        if (down_cast<const Number &>(*arg).is_negative()) {
            return false;
        } else if (not down_cast<const Number &>(*arg).is_exact()) {
            return false;
        }
    }
    if (could_extract_minus(*arg))
        return false;
    return true;
}

RCP<const Basic> asinh(const RCP<const Basic> &arg)
{
    if (eq(*arg, *zero))
        return zero;
    if (eq(*arg, *one))
        return log(add(one, sq2));
    if (eq(*arg, *minus_one))
        return log(sub(sq2, one));
    if (is_a_Number(*arg)) {
        RCP<const Number> _arg = rcp_static_cast<const Number>(arg);
        if (not _arg->is_exact()) {
            return _arg->get_eval().asinh(*_arg);
        } else if (_arg->is_negative()) {
            return neg(asinh(zero->sub(*_arg)));
        }
    }
    RCP<const Basic> d;
    bool b = handle_minus(arg, outArg(d));
    if (b) {
        return neg(asinh(d));
    }
    return make_rcp<const ASinh>(d);
}

ACsch::ACsch(const RCP<const Basic> &arg) : InverseHyperbolicFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool ACsch::is_canonical(const RCP<const Basic> &arg) const
{
    if (eq(*arg, *one) or eq(*arg, *minus_one))
        return false;
    if (is_a_Number(*arg)) {
        if (down_cast<const Number &>(*arg).is_negative()) {
            return false;
        } else if (not down_cast<const Number &>(*arg).is_exact()) {
            return false;
        }
    }
    if (could_extract_minus(*arg))
        return false;
    return true;
}

RCP<const Basic> acsch(const RCP<const Basic> &arg)
{
    if (eq(*arg, *one))
        return log(add(one, sq2));
    if (eq(*arg, *minus_one))
        return log(sub(sq2, one));

    if (is_a_Number(*arg)) {
        RCP<const Number> _arg = rcp_static_cast<const Number>(arg);
        if (not _arg->is_exact()) {
            return _arg->get_eval().acsch(*_arg);
        }
    }

    RCP<const Basic> d;
    bool b = handle_minus(arg, outArg(d));
    if (b) {
        return neg(acsch(d));
    }
    return make_rcp<const ACsch>(d);
}

ACosh::ACosh(const RCP<const Basic> &arg) : InverseHyperbolicFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool ACosh::is_canonical(const RCP<const Basic> &arg) const
{
    // TODO: Lookup into a cst table once complex is implemented
    if (eq(*arg, *one))
        return false;
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return false;
    }
    return true;
}

RCP<const Basic> acosh(const RCP<const Basic> &arg)
{
    // TODO: Lookup into a cst table once complex is implemented
    if (eq(*arg, *one))
        return zero;
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return down_cast<const Number &>(*arg).get_eval().acosh(*arg);
    }
    return make_rcp<const ACosh>(arg);
}

ATanh::ATanh(const RCP<const Basic> &arg) : InverseHyperbolicFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool ATanh::is_canonical(const RCP<const Basic> &arg) const
{
    if (eq(*arg, *zero))
        return false;
    if (is_a_Number(*arg)) {
        if (down_cast<const Number &>(*arg).is_negative()) {
            return false;
        } else if (not down_cast<const Number &>(*arg).is_exact()) {
            return false;
        }
    }
    if (could_extract_minus(*arg))
        return false;
    return true;
}

RCP<const Basic> atanh(const RCP<const Basic> &arg)
{
    if (eq(*arg, *zero))
        return zero;
    if (is_a_Number(*arg)) {
        RCP<const Number> _arg = rcp_static_cast<const Number>(arg);
        if (not _arg->is_exact()) {
            return _arg->get_eval().atanh(*_arg);
        } else if (_arg->is_negative()) {
            return neg(atanh(zero->sub(*_arg)));
        }
    }
    RCP<const Basic> d;
    bool b = handle_minus(arg, outArg(d));
    if (b) {
        return neg(atanh(d));
    }
    return make_rcp<const ATanh>(d);
}

ACoth::ACoth(const RCP<const Basic> &arg) : InverseHyperbolicFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool ACoth::is_canonical(const RCP<const Basic> &arg) const
{
    if (is_a_Number(*arg)) {
        if (down_cast<const Number &>(*arg).is_negative()) {
            return false;
        } else if (not down_cast<const Number &>(*arg).is_exact()) {
            return false;
        }
    }
    if (could_extract_minus(*arg))
        return false;
    return true;
}

RCP<const Basic> acoth(const RCP<const Basic> &arg)
{
    if (is_a_Number(*arg)) {
        RCP<const Number> _arg = rcp_static_cast<const Number>(arg);
        if (not _arg->is_exact()) {
            return _arg->get_eval().acoth(*_arg);
        } else if (_arg->is_negative()) {
            return neg(acoth(zero->sub(*_arg)));
        }
    }
    RCP<const Basic> d;
    bool b = handle_minus(arg, outArg(d));
    if (b) {
        return neg(acoth(d));
    }
    return make_rcp<const ACoth>(d);
}

ASech::ASech(const RCP<const Basic> &arg) : InverseHyperbolicFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool ASech::is_canonical(const RCP<const Basic> &arg) const
{
    // TODO: Lookup into a cst table once complex is implemented
    if (eq(*arg, *one))
        return false;
    if (eq(*arg, *zero))
        return false;
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return false;
    }
    return true;
}

RCP<const Basic> asech(const RCP<const Basic> &arg)
{
    // TODO: Lookup into a cst table once complex is implemented
    if (eq(*arg, *one))
        return zero;
    if (eq(*arg, *zero))
        return Inf;
    if (is_a_Number(*arg)) {
        RCP<const Number> _arg = rcp_static_cast<const Number>(arg);
        if (not _arg->is_exact()) {
            return _arg->get_eval().asech(*_arg);
        }
    }
    return make_rcp<const ASech>(arg);
}

RCP<const Basic> Sinh::create(const RCP<const Basic> &arg) const
{
    return sinh(arg);
}

RCP<const Basic> Csch::create(const RCP<const Basic> &arg) const
{
    return csch(arg);
}

RCP<const Basic> Cosh::create(const RCP<const Basic> &arg) const
{
    return cosh(arg);
}

RCP<const Basic> Sech::create(const RCP<const Basic> &arg) const
{
    return sech(arg);
}

RCP<const Basic> Tanh::create(const RCP<const Basic> &arg) const
{
    return tanh(arg);
}

RCP<const Basic> Coth::create(const RCP<const Basic> &arg) const
{
    return coth(arg);
}

RCP<const Basic> ASinh::create(const RCP<const Basic> &arg) const
{
    return asinh(arg);
}

RCP<const Basic> ACsch::create(const RCP<const Basic> &arg) const
{
    return acsch(arg);
}

RCP<const Basic> ACosh::create(const RCP<const Basic> &arg) const
{
    return acosh(arg);
}

RCP<const Basic> ATanh::create(const RCP<const Basic> &arg) const
{
    return atanh(arg);
}

RCP<const Basic> ACoth::create(const RCP<const Basic> &arg) const
{
    return acoth(arg);
}

RCP<const Basic> ASech::create(const RCP<const Basic> &arg) const
{
    return asech(arg);
}

KroneckerDelta::KroneckerDelta(const RCP<const Basic> &i,
                               const RCP<const Basic> &j)
    : TwoArgFunction(i, j)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(i, j))
}

bool KroneckerDelta::is_canonical(const RCP<const Basic> &i,
                                  const RCP<const Basic> &j) const
{
    RCP<const Basic> diff = expand(sub(i, j));
    if (eq(*diff, *zero)) {
        return false;
    } else if (is_a_Number(*diff)) {
        return false;
    } else {
        // TODO: SymPy uses default key sorting to return in order
        return true;
    }
}

RCP<const Basic> KroneckerDelta::create(const RCP<const Basic> &a,
                                        const RCP<const Basic> &b) const
{
    return kronecker_delta(a, b);
}

RCP<const Basic> kronecker_delta(const RCP<const Basic> &i,
                                 const RCP<const Basic> &j)
{
    // Expand is needed to simplify things like `i-(i+1)` to `-1`
    RCP<const Basic> diff = expand(sub(i, j));
    if (eq(*diff, *zero)) {
        return one;
    } else if (is_a_Number(*diff)) {
        return zero;
    } else {
        // SymPy uses default key sorting to return in order
        return make_rcp<const KroneckerDelta>(i, j);
    }
}

bool has_dup(const vec_basic &arg)
{
    map_basic_basic d;
    auto it = d.end();
    for (const auto &p : arg) {
        it = d.find(p);
        if (it == d.end()) {
            insert(d, p, one);
        } else {
            return true;
        }
    }
    return false;
}

LeviCivita::LeviCivita(const vec_basic &&arg) : MultiArgFunction(std::move(arg))
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(get_vec()))
}

bool LeviCivita::is_canonical(const vec_basic &arg) const
{
    bool are_int = true;
    for (const auto &p : arg) {
        if (not(is_a_Number(*p))) {
            are_int = false;
            break;
        }
    }
    if (are_int) {
        return false;
    } else if (has_dup(arg)) {
        return false;
    } else {
        return true;
    }
}

RCP<const Basic> LeviCivita::create(const vec_basic &a) const
{
    return levi_civita(a);
}

RCP<const Basic> eval_levicivita(const vec_basic &arg, int len)
{
    int i, j;
    RCP<const Basic> res = one;
    for (i = 0; i < len; i++) {
        for (j = i + 1; j < len; j++) {
            res = mul(sub(arg[j], arg[i]), res);
        }
        res = div(res, factorial(i));
    }
    return res;
}

RCP<const Basic> levi_civita(const vec_basic &arg)
{
    bool are_int = true;
    int len = 0;
    for (const auto &p : arg) {
        if (not(is_a_Number(*p))) {
            are_int = false;
            break;
        } else {
            len++;
        }
    }
    if (are_int) {
        return eval_levicivita(arg, len);
    } else if (has_dup(arg)) {
        return zero;
    } else {
        return make_rcp<const LeviCivita>(std::move(arg));
    }
}

Zeta::Zeta(const RCP<const Basic> &s, const RCP<const Basic> &a)
    : TwoArgFunction(s, a)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(s, a))
}

Zeta::Zeta(const RCP<const Basic> &s) : TwoArgFunction(s, one)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(s, one))
}

bool Zeta::is_canonical(const RCP<const Basic> &s,
                        const RCP<const Basic> &a) const
{
    if (eq(*s, *zero))
        return false;
    if (eq(*s, *one))
        return false;
    if (is_a<Integer>(*s) and is_a<Integer>(*a)) {
        auto s_ = down_cast<const Integer &>(*s).as_int();
        if (s_ < 0 || s_ % 2 == 0)
            return false;
    }
    return true;
}

RCP<const Basic> Zeta::create(const RCP<const Basic> &a,
                              const RCP<const Basic> &b) const
{
    return zeta(a, b);
}

RCP<const Basic> zeta(const RCP<const Basic> &s, const RCP<const Basic> &a)
{
    if (is_a_Number(*s)) {
        if (down_cast<const Number &>(*s).is_zero()) {
            return sub(div(one, i2), a);
        } else if (down_cast<const Number &>(*s).is_one()) {
            return infty(0);
        } else if (is_a<Integer>(*s) and is_a<Integer>(*a)) {
            auto s_ = down_cast<const Integer &>(*s).as_int();
            auto a_ = down_cast<const Integer &>(*a).as_int();
            RCP<const Basic> zeta;
            if (s_ < 0) {
                RCP<const Number> res = (s_ % 2 == 0) ? one : minus_one;
                zeta
                    = mulnum(res, divnum(bernoulli(-s_ + 1), integer(-s_ + 1)));
            } else if (s_ % 2 == 0) {
                RCP<const Number> b = bernoulli(s_);
                RCP<const Number> f = factorial(s_);
                zeta = divnum(pownum(integer(2), integer(s_ - 1)), f);
                zeta = mul(zeta, mul(pow(pi, s), abs(b)));
            } else {
                return make_rcp<const Zeta>(s, a);
            }
            if (a_ < 0)
                return add(zeta, harmonic(-a_, s_));
            return sub(zeta, harmonic(a_ - 1, s_));
        }
    }
    return make_rcp<const Zeta>(s, a);
}

RCP<const Basic> zeta(const RCP<const Basic> &s)
{
    return zeta(s, one);
}

Dirichlet_eta::Dirichlet_eta(const RCP<const Basic> &s) : OneArgFunction(s)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(s))
}

bool Dirichlet_eta::is_canonical(const RCP<const Basic> &s) const
{
    if (eq(*s, *one))
        return false;
    if (not(is_a<Zeta>(*zeta(s))))
        return false;
    return true;
}

RCP<const Basic> Dirichlet_eta::rewrite_as_zeta() const
{
    return mul(sub(one, pow(i2, sub(one, get_arg()))), zeta(get_arg()));
}

RCP<const Basic> Dirichlet_eta::create(const RCP<const Basic> &arg) const
{
    return dirichlet_eta(arg);
}

RCP<const Basic> dirichlet_eta(const RCP<const Basic> &s)
{
    if (is_a_Number(*s) and down_cast<const Number &>(*s).is_one()) {
        return log(i2);
    }
    RCP<const Basic> z = zeta(s);
    if (is_a<Zeta>(*z)) {
        return make_rcp<const Dirichlet_eta>(s);
    } else {
        return mul(sub(one, pow(i2, sub(one, s))), z);
    }
}

bool Erf::is_canonical(const RCP<const Basic> &arg) const
{
    if (is_a<Integer>(*arg) and down_cast<const Integer &>(*arg).is_zero())
        return false;
    if (could_extract_minus(*arg))
        return false;
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return false;
    }
    return true;
}

RCP<const Basic> Erf::create(const RCP<const Basic> &arg) const
{
    return erf(arg);
}

RCP<const Basic> erf(const RCP<const Basic> &arg)
{
    if (is_a<Integer>(*arg) and down_cast<const Integer &>(*arg).is_zero()) {
        return zero;
    }
    if (is_a_Number(*arg)) {
        RCP<const Number> _arg = rcp_static_cast<const Number>(arg);
        if (not _arg->is_exact()) {
            return _arg->get_eval().erf(*_arg);
        }
    }
    RCP<const Basic> d;
    bool b = handle_minus(arg, outArg(d));
    if (b) {
        return neg(erf(d));
    }
    return make_rcp<const Erf>(d);
}

bool Erfc::is_canonical(const RCP<const Basic> &arg) const
{
    if (is_a<Integer>(*arg) and down_cast<const Integer &>(*arg).is_zero())
        return false;
    if (could_extract_minus(*arg))
        return false;
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return false;
    }
    return true;
}

RCP<const Basic> Erfc::create(const RCP<const Basic> &arg) const
{
    return erfc(arg);
}

RCP<const Basic> erfc(const RCP<const Basic> &arg)
{
    if (is_a<Integer>(*arg) and down_cast<const Integer &>(*arg).is_zero()) {
        return one;
    }
    if (is_a_Number(*arg)) {
        RCP<const Number> _arg = rcp_static_cast<const Number>(arg);
        if (not _arg->is_exact()) {
            return _arg->get_eval().erfc(*_arg);
        }
    }

    RCP<const Basic> d;
    bool b = handle_minus(arg, outArg(d));
    if (b) {
        return add(integer(2), neg(erfc(d)));
    }
    return make_rcp<const Erfc>(d);
}

Gamma::Gamma(const RCP<const Basic> &arg) : OneArgFunction{arg}
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool Gamma::is_canonical(const RCP<const Basic> &arg) const
{
    if (is_a<Integer>(*arg))
        return false;
    if (is_a<Rational>(*arg)
        and (get_den(down_cast<const Rational &>(*arg).as_rational_class()))
                == 2) {
        return false;
    }
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return false;
    }
    return true;
}

RCP<const Basic> Gamma::create(const RCP<const Basic> &arg) const
{
    return gamma(arg);
}

RCP<const Basic> gamma_positive_int(const RCP<const Basic> &arg)
{
    SYMENGINE_ASSERT(is_a<Integer>(*arg))
    RCP<const Integer> arg_ = rcp_static_cast<const Integer>(arg);
    SYMENGINE_ASSERT(arg_->is_positive())
    return factorial((arg_->subint(*one))->as_int());
}

RCP<const Basic> gamma_multiple_2(const RCP<const Basic> &arg)
{
    SYMENGINE_ASSERT(is_a<Rational>(*arg))
    RCP<const Rational> arg_ = rcp_static_cast<const Rational>(arg);
    SYMENGINE_ASSERT(get_den(arg_->as_rational_class()) == 2)
    RCP<const Integer> n, k;
    RCP<const Number> coeff;
    n = quotient_f(*(integer(mp_abs(get_num(arg_->as_rational_class())))),
                   *(integer(get_den(arg_->as_rational_class()))));
    if (arg_->is_positive()) {
        k = n;
        coeff = one;
    } else {
        n = n->addint(*one);
        k = n;
        if ((n->as_int() & 1) == 0) {
            coeff = one;
        } else {
            coeff = minus_one;
        }
    }
    int j = 1;
    for (int i = 3; i < 2 * k->as_int(); i = i + 2) {
        j = j * i;
    }
    coeff = mulnum(coeff, integer(j));
    if (arg_->is_positive()) {
        return div(mul(coeff, sqrt(pi)), pow(i2, n));
    } else {
        return div(mul(pow(i2, n), sqrt(pi)), coeff);
    }
}

RCP<const Basic> gamma(const RCP<const Basic> &arg)
{
    if (is_a<Integer>(*arg)) {
        RCP<const Integer> arg_ = rcp_static_cast<const Integer>(arg);
        if (arg_->is_positive()) {
            return gamma_positive_int(arg);
        } else {
            return ComplexInf;
        }
    } else if (is_a<Rational>(*arg)) {
        RCP<const Rational> arg_ = rcp_static_cast<const Rational>(arg);
        if ((get_den(arg_->as_rational_class())) == 2) {
            return gamma_multiple_2(arg);
        } else {
            return make_rcp<const Gamma>(arg);
        }
    } else if (is_a_Number(*arg)
               and not down_cast<const Number &>(*arg).is_exact()) {
        return down_cast<const Number &>(*arg).get_eval().gamma(*arg);
    }
    return make_rcp<const Gamma>(arg);
}

LowerGamma::LowerGamma(const RCP<const Basic> &s, const RCP<const Basic> &x)
    : TwoArgFunction(s, x)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(s, x))
}

bool LowerGamma::is_canonical(const RCP<const Basic> &s,
                              const RCP<const Basic> &x) const
{
    // Only special values are evaluated
    if (eq(*s, *one))
        return false;
    if (is_a<Integer>(*s)
        and down_cast<const Integer &>(*s).as_integer_class() > 1)
        return false;
    if (is_a<Integer>(*mul(i2, s)))
        return false;
#ifdef HAVE_SYMENGINE_MPFR
#if MPFR_VERSION_MAJOR > 3
    if (is_a<RealMPFR>(*s) && is_a<RealMPFR>(*x))
        return false;
#endif
#endif
    return true;
}

RCP<const Basic> LowerGamma::create(const RCP<const Basic> &a,
                                    const RCP<const Basic> &b) const
{
    return lowergamma(a, b);
}

RCP<const Basic> lowergamma(const RCP<const Basic> &s,
                            const RCP<const Basic> &x)
{
    // Only special values are being evaluated
    if (is_a<Integer>(*s)) {
        RCP<const Integer> s_int = rcp_static_cast<const Integer>(s);
        if (s_int->is_one()) {
            return sub(one, exp(mul(minus_one, x)));
        } else if (s_int->as_integer_class() > 1) {
            s_int = s_int->subint(*one);
            return sub(mul(s_int, lowergamma(s_int, x)),
                       mul(pow(x, s_int), exp(mul(minus_one, x))));
        } else {
            return make_rcp<const LowerGamma>(s, x);
        }
    } else if (is_a<Integer>(*(mul(i2, s)))) {
        RCP<const Number> s_num = rcp_static_cast<const Number>(s);
        s_num = subnum(s_num, one);
        if (eq(*s, *div(one, integer(2)))) {
            return mul(sqrt(pi),
                       erf(sqrt(x))); // base case for s of the form n/2
        } else if (s_num->is_positive()) {
            return sub(mul(s_num, lowergamma(s_num, x)),
                       mul(pow(x, s_num), exp(mul(minus_one, x))));
        } else {
            return div(add(lowergamma(add(s, one), x),
                           mul(pow(x, s), exp(mul(minus_one, x)))),
                       s);
        }
#ifdef HAVE_SYMENGINE_MPFR
#if MPFR_VERSION_MAJOR > 3
    } else if (is_a<RealMPFR>(*s) && is_a<RealMPFR>(*x)) {
        const auto &s_ = down_cast<const RealMPFR &>(*s).i.get_mpfr_t();
        const auto &x_ = down_cast<const RealMPFR &>(*x).i.get_mpfr_t();
        if (mpfr_cmp_si(x_, 0) >= 0) {
            mpfr_class t(std::max(mpfr_get_prec(s_), mpfr_get_prec(x_)));
            mpfr_class u(std::max(mpfr_get_prec(s_), mpfr_get_prec(x_)));
            mpfr_gamma_inc(t.get_mpfr_t(), s_, x_, MPFR_RNDN);
            mpfr_gamma(u.get_mpfr_t(), s_, MPFR_RNDN);
            mpfr_sub(t.get_mpfr_t(), u.get_mpfr_t(), t.get_mpfr_t(), MPFR_RNDN);
            return real_mpfr(std::move(t));
        } else {
            throw NotImplementedError("Not implemented.");
        }
#endif
#endif
    }
    return make_rcp<const LowerGamma>(s, x);
}

UpperGamma::UpperGamma(const RCP<const Basic> &s, const RCP<const Basic> &x)
    : TwoArgFunction(s, x)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(s, x))
}

bool UpperGamma::is_canonical(const RCP<const Basic> &s,
                              const RCP<const Basic> &x) const
{
    // Only special values are evaluated
    if (eq(*s, *one))
        return false;
    if (is_a<Integer>(*s)
        and down_cast<const Integer &>(*s).as_integer_class() > 1)
        return false;
    if (is_a<Integer>(*mul(i2, s)))
        return false;
#ifdef HAVE_SYMENGINE_MPFR
#if MPFR_VERSION_MAJOR > 3
    if (is_a<RealMPFR>(*s) && is_a<RealMPFR>(*x))
        return false;
#endif
#endif
    return true;
}

RCP<const Basic> UpperGamma::create(const RCP<const Basic> &a,
                                    const RCP<const Basic> &b) const
{
    return uppergamma(a, b);
}

RCP<const Basic> uppergamma(const RCP<const Basic> &s,
                            const RCP<const Basic> &x)
{
    // Only special values are being evaluated
    if (is_a<Integer>(*s)) {
        RCP<const Integer> s_int = rcp_static_cast<const Integer>(s);
        if (s_int->is_one()) {
            return exp(mul(minus_one, x));
        } else if (s_int->as_integer_class() > 1) {
            s_int = s_int->subint(*one);
            return add(mul(s_int, uppergamma(s_int, x)),
                       mul(pow(x, s_int), exp(mul(minus_one, x))));
        } else {
            // TODO: implement unpolarfy to handle this case
            return make_rcp<const LowerGamma>(s, x);
        }
    } else if (is_a<Integer>(*(mul(i2, s)))) {
        RCP<const Number> s_num = rcp_static_cast<const Number>(s);
        s_num = subnum(s_num, one);
        if (eq(*s, *div(one, integer(2)))) {
            return mul(sqrt(pi),
                       erfc(sqrt(x))); // base case for s of the form n/2
        } else if (s_num->is_positive()) {
            return add(mul(s_num, uppergamma(s_num, x)),
                       mul(pow(x, s_num), exp(mul(minus_one, x))));
        } else {
            return div(sub(uppergamma(add(s, one), x),
                           mul(pow(x, s), exp(mul(minus_one, x)))),
                       s);
        }
#ifdef HAVE_SYMENGINE_MPFR
#if MPFR_VERSION_MAJOR > 3
    } else if (is_a<RealMPFR>(*s) && is_a<RealMPFR>(*x)) {
        const auto &s_ = down_cast<const RealMPFR &>(*s).i.get_mpfr_t();
        const auto &x_ = down_cast<const RealMPFR &>(*x).i.get_mpfr_t();
        if (mpfr_cmp_si(x_, 0) >= 0) {
            mpfr_class t(std::max(mpfr_get_prec(s_), mpfr_get_prec(x_)));
            mpfr_gamma_inc(t.get_mpfr_t(), s_, x_, MPFR_RNDN);
            return real_mpfr(std::move(t));
        } else {
            throw NotImplementedError("Not implemented.");
        }
#endif
#endif
    }
    return make_rcp<const UpperGamma>(s, x);
}

bool LogGamma::is_canonical(const RCP<const Basic> &arg) const
{
    if (is_a<Integer>(*arg)) {
        RCP<const Integer> arg_int = rcp_static_cast<const Integer>(arg);
        if (not arg_int->is_positive()) {
            return false;
        }
        if (eq(*integer(1), *arg_int) or eq(*integer(2), *arg_int)
            or eq(*integer(3), *arg_int)) {
            return false;
        }
    }
    return true;
}

RCP<const Basic> LogGamma::rewrite_as_gamma() const
{
    return log(gamma(get_arg()));
}

RCP<const Basic> LogGamma::create(const RCP<const Basic> &arg) const
{
    return loggamma(arg);
}

RCP<const Basic> loggamma(const RCP<const Basic> &arg)
{
    if (is_a<Integer>(*arg)) {
        RCP<const Integer> arg_int = rcp_static_cast<const Integer>(arg);
        if (not arg_int->is_positive()) {
            return Inf;
        }
        if (eq(*integer(1), *arg_int) or eq(*integer(2), *arg_int)) {
            return zero;
        } else if (eq(*integer(3), *arg_int)) {
            return log(integer(2));
        }
    }
    return make_rcp<const LogGamma>(arg);
}

RCP<const Beta> Beta::from_two_basic(const RCP<const Basic> &x,
                                     const RCP<const Basic> &y)
{
    if (x->__cmp__(*y) == -1) {
        return make_rcp<const Beta>(y, x);
    }
    return make_rcp<const Beta>(x, y);
}

bool Beta::is_canonical(const RCP<const Basic> &x, const RCP<const Basic> &y)
{
    if (x->__cmp__(*y) == -1) {
        return false;
    }
    if (is_a<Integer>(*x)
        or (is_a<Rational>(*x)
            and (get_den(down_cast<const Rational &>(*x).as_rational_class()))
                    == 2)) {
        if (is_a<Integer>(*y)
            or (is_a<Rational>(*y)
                and (get_den(
                        down_cast<const Rational &>(*y).as_rational_class()))
                        == 2)) {
            return false;
        }
    }
    return true;
}

RCP<const Basic> Beta::rewrite_as_gamma() const
{
    return div(mul(gamma(get_arg1()), gamma(get_arg2())),
               gamma(add(get_arg1(), get_arg2())));
}

RCP<const Basic> Beta::create(const RCP<const Basic> &a,
                              const RCP<const Basic> &b) const
{
    return beta(a, b);
}

RCP<const Basic> beta(const RCP<const Basic> &x, const RCP<const Basic> &y)
{
    // Only special values are being evaluated
    if (eq(*add(x, y), *one)) {
        return ComplexInf;
    }

    if (is_a<Integer>(*x)) {
        RCP<const Integer> x_int = rcp_static_cast<const Integer>(x);
        if (x_int->is_positive()) {
            if (is_a<Integer>(*y)) {
                RCP<const Integer> y_int = rcp_static_cast<const Integer>(y);
                if (y_int->is_positive()) {
                    return div(
                        mul(gamma_positive_int(x), gamma_positive_int(y)),
                        gamma_positive_int(add(x, y)));
                } else {
                    return ComplexInf;
                }
            } else if (is_a<Rational>(*y)) {
                RCP<const Rational> y_ = rcp_static_cast<const Rational>(y);
                if (get_den(y_->as_rational_class()) == 2) {
                    return div(mul(gamma_positive_int(x), gamma_multiple_2(y)),
                               gamma_multiple_2(add(x, y)));
                } else {
                    return Beta::from_two_basic(x, y);
                }
            }
        } else {
            return ComplexInf;
        }
    }

    if (is_a<Integer>(*y)) {
        RCP<const Integer> y_int = rcp_static_cast<const Integer>(y);
        if (y_int->is_positive()) {
            if (is_a<Rational>(*x)) {
                RCP<const Rational> x_ = rcp_static_cast<const Rational>(x);
                if (get_den(x_->as_rational_class()) == 2) {
                    return div(mul(gamma_positive_int(y), gamma_multiple_2(x)),
                               gamma_multiple_2(add(x, y)));
                } else {
                    return Beta::from_two_basic(x, y);
                }
            }
        } else {
            return ComplexInf;
        }
    }

    if (is_a<const Rational>(*x)
        and get_den(down_cast<const Rational &>(*x).as_rational_class()) == 2) {
        if (is_a<Integer>(*y)) {
            RCP<const Integer> y_int = rcp_static_cast<const Integer>(y);
            if (y_int->is_positive()) {
                return div(mul(gamma_multiple_2(x), gamma_positive_int(y)),
                           gamma_multiple_2(add(x, y)));
            } else {
                return ComplexInf;
            }
        }
        if (is_a<const Rational>(*y)
            and get_den((down_cast<const Rational &>(*y)).as_rational_class())
                    == 2) {
            return div(mul(gamma_multiple_2(x), gamma_multiple_2(y)),
                       gamma_positive_int(add(x, y)));
        }
    }
    return Beta::from_two_basic(x, y);
}

bool PolyGamma::is_canonical(const RCP<const Basic> &n,
                             const RCP<const Basic> &x)
{
    if (is_a_Number(*x) and not(down_cast<const Number &>(*x)).is_positive()) {
        return false;
    }
    if (eq(*n, *zero)) {
        if (eq(*x, *one)) {
            return false;
        }
        if (is_a<Rational>(*x)) {
            auto x_ = rcp_static_cast<const Rational>(x);
            auto den = get_den(x_->as_rational_class());
            if (den == 2 or den == 3 or den == 4) {
                return false;
            }
        }
    }
    return true;
}

RCP<const Basic> PolyGamma::rewrite_as_zeta() const
{
    if (not is_a<Integer>(*get_arg1())) {
        return rcp_from_this();
    }
    RCP<const Integer> n = rcp_static_cast<const Integer>(get_arg1());
    if (not(n->is_positive())) {
        return rcp_from_this();
    }
    if ((n->as_int() & 1) == 0) {
        return neg(mul(factorial(n->as_int()), zeta(add(n, one), get_arg2())));
    } else {
        return mul(factorial(n->as_int()), zeta(add(n, one), get_arg2()));
    }
}

RCP<const Basic> PolyGamma::create(const RCP<const Basic> &a,
                                   const RCP<const Basic> &b) const
{
    return polygamma(a, b);
}

RCP<const Basic> polygamma(const RCP<const Basic> &n_,
                           const RCP<const Basic> &x_)
{
    // Only special values are being evaluated
    if (is_a_Number(*x_)
        and not(down_cast<const Number &>(*x_)).is_positive()) {
        return ComplexInf;
    }
    if (is_a<Integer>(*n_) and is_a<Integer>(*x_)) {
        auto n = down_cast<const Integer &>(*n_).as_int();
        auto x = down_cast<const Integer &>(*x_).as_int();
        if (n == 0) {
            return sub(harmonic(x - 1, 1), EulerGamma);
        } else if (n % 2 == 1) {
            return mul(factorial(n), zeta(add(n_, one), x_));
        }
    }
    if (eq(*n_, *zero)) {
        if (eq(*x_, *one)) {
            return neg(EulerGamma);
        }
        if (is_a<Rational>(*x_)) {
            RCP<const Rational> x = rcp_static_cast<const Rational>(x_);
            const auto den = get_den(x->as_rational_class());
            const auto num = get_num(x->as_rational_class());
            const integer_class r = num % den;
            RCP<const Basic> res;
            if (den == 2) {
                res = sub(mul(im2, log(i2)), EulerGamma);
            } else if (den == 3) {
                if (num == 1) {
                    res = add(neg(div(div(pi, i2), sqrt(i3))),
                              sub(div(mul(im3, log(i3)), i2), EulerGamma));
                } else {
                    res = add(div(div(pi, i2), sqrt(i3)),
                              sub(div(mul(im3, log(i3)), i2), EulerGamma));
                }
            } else if (den == 4) {
                if (num == 1) {
                    res = add(div(pi, im2), sub(mul(im3, log(i2)), EulerGamma));
                } else {
                    res = add(div(pi, i2), sub(mul(im3, log(i2)), EulerGamma));
                }
            } else {
                return make_rcp<const PolyGamma>(n_, x_);
            }
            rational_class a(0), f(r, den);
            for (unsigned long i = 0; i < (num - r) / den; ++i) {
                a += 1 / (f + i);
            }
            return add(Rational::from_mpq(a), res);
        }
    }
    return make_rcp<const PolyGamma>(n_, x_);
}

RCP<const Basic> digamma(const RCP<const Basic> &x)
{
    return polygamma(zero, x);
}

RCP<const Basic> trigamma(const RCP<const Basic> &x)
{
    return polygamma(one, x);
}

Abs::Abs(const RCP<const Basic> &arg) : OneArgFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool Abs::is_canonical(const RCP<const Basic> &arg) const
{
    if (is_a<Integer>(*arg) or is_a<Rational>(*arg) or is_a<Complex>(*arg))
        return false;
    if (is_a_Number(*arg) and not down_cast<const Number &>(*arg).is_exact()) {
        return false;
    }
    if (is_a<Abs>(*arg)) {
        return false;
    }

    if (could_extract_minus(*arg)) {
        return false;
    }

    return true;
}

RCP<const Basic> Abs::create(const RCP<const Basic> &arg) const
{
    return abs(arg);
}

RCP<const Basic> abs(const RCP<const Basic> &arg)
{
    if (is_a<Integer>(*arg)) {
        RCP<const Integer> arg_ = rcp_static_cast<const Integer>(arg);
        if (arg_->is_negative()) {
            return arg_->neg();
        } else {
            return arg_;
        }
    } else if (is_a<Rational>(*arg)) {
        RCP<const Rational> arg_ = rcp_static_cast<const Rational>(arg);
        if (arg_->is_negative()) {
            return arg_->neg();
        } else {
            return arg_;
        }
    } else if (is_a<Complex>(*arg)) {
        RCP<const Complex> arg_ = rcp_static_cast<const Complex>(arg);
        return sqrt(Rational::from_mpq(arg_->real_ * arg_->real_
                                       + arg_->imaginary_ * arg_->imaginary_));
    } else if (is_a_Number(*arg)
               and not down_cast<const Number &>(*arg).is_exact()) {
        return down_cast<const Number &>(*arg).get_eval().abs(*arg);
    }
    if (is_a<Abs>(*arg)) {
        return arg;
    }

    RCP<const Basic> d;
    handle_minus(arg, outArg(d));
    return make_rcp<const Abs>(d);
}

Max::Max(const vec_basic &&arg) : MultiArgFunction(std::move(arg))
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(get_vec()))
}

bool Max::is_canonical(const vec_basic &arg) const
{
    if (arg.size() < 2)
        return false;

    bool non_number_exists = false;

    for (const auto &p : arg) {
        if (is_a<Complex>(*p) or is_a<Max>(*p))
            return false;
        if (not is_a_Number(*p))
            non_number_exists = true;
    }
    if (not std::is_sorted(arg.begin(), arg.end(), RCPBasicKeyLess()))
        return false;

    return non_number_exists; // all arguments cant be numbers
}

RCP<const Basic> Max::create(const vec_basic &a) const
{
    return max(a);
}

RCP<const Basic> max(const vec_basic &arg)
{
    bool number_set = false;
    RCP<const Number> max_number, difference;
    set_basic new_args;

    for (const auto &p : arg) {
        if (is_a<Complex>(*p))
            throw SymEngineException("Complex can't be passed to max!");

        if (is_a_Number(*p)) {
            if (not number_set) {
                max_number = rcp_static_cast<const Number>(p);

            } else {
                if (eq(*p, *Inf)) {
                    return Inf;
                } else if (eq(*p, *NegInf)) {
                    continue;
                }
                difference = down_cast<const Number &>(*p).sub(*max_number);

                if (difference->is_zero() and not difference->is_exact()) {
                    if (max_number->is_exact())
                        max_number = rcp_static_cast<const Number>(p);
                } else if (difference->is_positive()) {
                    max_number = rcp_static_cast<const Number>(p);
                }
            }
            number_set = true;

        } else if (is_a<Max>(*p)) {
            for (const auto &l : down_cast<const Max &>(*p).get_args()) {
                if (is_a_Number(*l)) {
                    if (not number_set) {
                        max_number = rcp_static_cast<const Number>(l);

                    } else {
                        difference = rcp_static_cast<const Number>(l)->sub(
                            *max_number);

                        if (difference->is_zero()
                            and not difference->is_exact()) {
                            if (max_number->is_exact())
                                max_number = rcp_static_cast<const Number>(l);
                        } else if (difference->is_positive()) {
                            max_number = rcp_static_cast<const Number>(l);
                        }
                    }
                    number_set = true;
                } else {
                    new_args.insert(l);
                }
            }
        } else {
            new_args.insert(p);
        }
    }

    if (number_set)
        new_args.insert(max_number);

    vec_basic final_args(new_args.size());
    std::copy(new_args.begin(), new_args.end(), final_args.begin());

    if (final_args.size() > 1) {
        return make_rcp<const Max>(std::move(final_args));
    } else if (final_args.size() == 1) {
        return final_args[0];
    } else {
        throw SymEngineException("Empty vec_basic passed to max!");
    }
}

Min::Min(const vec_basic &&arg) : MultiArgFunction(std::move(arg))
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(get_vec()))
}

bool Min::is_canonical(const vec_basic &arg) const
{
    if (arg.size() < 2)
        return false;

    bool non_number_exists = false;

    for (const auto &p : arg) {
        if (is_a<Complex>(*p) or is_a<Min>(*p))
            return false;
        if (not is_a_Number(*p))
            non_number_exists = true;
    }
    if (not std::is_sorted(arg.begin(), arg.end(), RCPBasicKeyLess()))
        return false;

    return non_number_exists; // all arguments cant be numbers
}

RCP<const Basic> Min::create(const vec_basic &a) const
{
    return min(a);
}

RCP<const Basic> min(const vec_basic &arg)
{
    bool number_set = false;
    RCP<const Number> min_number, difference;
    set_basic new_args;

    for (const auto &p : arg) {
        if (is_a<Complex>(*p))
            throw SymEngineException("Complex can't be passed to min!");

        if (is_a_Number(*p)) {
            if (not number_set) {
                min_number = rcp_static_cast<const Number>(p);

            } else {
                if (eq(*p, *Inf)) {
                    continue;
                } else if (eq(*p, *NegInf)) {
                    return NegInf;
                }
                difference = min_number->sub(*rcp_static_cast<const Number>(p));

                if (difference->is_zero() and not difference->is_exact()) {
                    if (min_number->is_exact())
                        min_number = rcp_static_cast<const Number>(p);
                } else if (difference->is_positive()) {
                    min_number = rcp_static_cast<const Number>(p);
                }
            }
            number_set = true;

        } else if (is_a<Min>(*p)) {
            for (const auto &l : down_cast<const Min &>(*p).get_args()) {
                if (is_a_Number(*l)) {
                    if (not number_set) {
                        min_number = rcp_static_cast<const Number>(l);

                    } else {
                        difference = min_number->sub(
                            *rcp_static_cast<const Number>(l));

                        if (difference->is_zero()
                            and not difference->is_exact()) {
                            if (min_number->is_exact())
                                min_number = rcp_static_cast<const Number>(l);
                        } else if (difference->is_positive()) {
                            min_number = rcp_static_cast<const Number>(l);
                        }
                    }
                    number_set = true;
                } else {
                    new_args.insert(l);
                }
            }
        } else {
            new_args.insert(p);
        }
    }

    if (number_set)
        new_args.insert(min_number);

    vec_basic final_args(new_args.size());
    std::copy(new_args.begin(), new_args.end(), final_args.begin());

    if (final_args.size() > 1) {
        return make_rcp<const Min>(std::move(final_args));
    } else if (final_args.size() == 1) {
        return final_args[0];
    } else {
        throw SymEngineException("Empty vec_basic passed to min!");
    }
}

UnevaluatedExpr::UnevaluatedExpr(const RCP<const Basic> &arg)
    : OneArgFunction(arg)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(arg))
}

bool UnevaluatedExpr::is_canonical(const RCP<const Basic> &arg) const
{
    return true;
}

RCP<const Basic> UnevaluatedExpr::create(const RCP<const Basic> &arg) const
{
    return make_rcp<const UnevaluatedExpr>(arg);
}

RCP<const Basic> unevaluated_expr(const RCP<const Basic> &arg)
{
    return make_rcp<const UnevaluatedExpr>(arg);
}

} // SymEngine
