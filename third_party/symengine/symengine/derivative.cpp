#include <symengine/visitor.h>
#include <symengine/subs.h>
#include <symengine/symengine_casts.h>
#include <symengine/derivative.h>

namespace SymEngine
{

extern RCP<const Basic> i2;

// Needs create(vec_basic) method to be used.
template <typename T>
static inline RCP<const Basic> fdiff(const T &self, RCP<const Symbol> x,
                                     DiffVisitor &visitor)
{
    RCP<const Basic> diff = zero;
    RCP<const Basic> ret;
    bool know_deriv;

    vec_basic v = self.get_args();
    vec_basic vdiff(v.size());

    unsigned count = 0;
    for (unsigned i = 0; i < v.size(); i++) {
        vdiff[i] = visitor.apply(v[i]);
        if (neq(*vdiff[i], *zero)) {
            count++;
        }
    }

    if (count == 0) {
        return diff;
    }

    for (unsigned i = 0; i < v.size(); i++) {
        if (eq(*vdiff[i], *zero))
            continue;
        know_deriv = fdiff(outArg(ret), self, i);
        if (know_deriv) {
            diff = add(diff, mul(ret, vdiff[i]));
        } else {
            if (count == 1 and eq(*v[i], *x)) {
                return Derivative::create(self.rcp_from_this(), {x});
            }
            vec_basic new_args = v;
            std::ostringstream stm;
            stm << (i + 1);
            new_args[i] = get_dummy(self, "xi_" + stm.str());
            map_basic_basic m;
            insert(m, new_args[i], v[i]);
            diff = add(diff, mul(vdiff[i],
                                 make_rcp<const Subs>(
                                     Derivative::create(self.create(new_args),
                                                        {new_args[i]}),
                                     m)));
        }
    }
    return diff;
}

static bool fdiff(const Ptr<RCP<const Basic>> &ret, const Zeta &self,
                  unsigned index)
{
    if (index == 1) {
        *ret = mul(mul(minus_one, self.get_arg1()),
                   zeta(add(self.get_arg1(), one), self.get_arg2()));
        return true;
    } else {
        return false;
    }
}

static bool fdiff(const Ptr<RCP<const Basic>> &ret, const UpperGamma &self,
                  unsigned index)
{
    if (index == 1) {
        *ret = mul(mul(pow(self.get_arg2(), sub(self.get_arg1(), one)),
                       exp(neg(self.get_arg2()))),
                   minus_one);
        return true;
    } else {
        return false;
    }
}

static bool fdiff(const Ptr<RCP<const Basic>> &ret, const LowerGamma &self,
                  unsigned index)
{
    if (index == 1) {
        *ret = mul(pow(self.get_arg2(), sub(self.get_arg1(), one)),
                   exp(neg(self.get_arg2())));
        return true;
    } else {
        return false;
    }
}

static bool fdiff(const Ptr<RCP<const Basic>> &ret, const PolyGamma &self,
                  unsigned index)
{
    if (index == 1) {
        *ret = polygamma(add(self.get_arg1(), one), self.get_arg2());
        return true;
    } else {
        return false;
    }
}

static bool fdiff(const Ptr<RCP<const Basic>> &ret, const Function &self,
                  unsigned index)
{
    // Don't know the derivative, fallback to `Derivative` instances
    return false;
}

template <typename P>
static inline RCP<const Basic> diff_upolyflint(const P &self, const Symbol &x)
{
    if (self.get_var()->__eq__(x)) {
        return P::from_container(self.get_var(), self.get_poly().derivative());
    } else {
        return P::from_dict(self.get_var(), {{}});
    }
}

template <typename Poly, typename Dict>
static inline RCP<const Basic> diff_upoly(const Poly &self, const Symbol &x)
{
    if (self.get_var()->__eq__(x)) {
        Dict d;
        for (auto it = self.begin(); it != self.end(); ++it) {
            if (it->first != 0)
                d[it->first - 1] = it->second * it->first;
        }
        return Poly::from_dict(self.get_var(), std::move(d));
    } else {
        return Poly::from_dict(self.get_var(), {{}});
    }
}

template <typename Container, typename Poly>
static RCP<const Basic> diff_mpoly(const MSymEnginePoly<Container, Poly> &self,
                                   const RCP<const Symbol> &x)
{
    using Dict = typename Container::dict_type;
    using Vec = typename Container::vec_type;
    Dict dict;

    if (self.get_vars().find(x) != self.get_vars().end()) {
        auto i = self.get_vars().begin();
        unsigned int index = 0;
        while (!(*i)->__eq__(*x)) {
            i++;
            index++;
        } // find the index of the variable we are differentiating WRT.
        for (auto bucket : self.get_poly().dict_) {
            if (bucket.first[index] != 0) {
                Vec v = bucket.first;
                v[index]--;
                dict.insert({v, bucket.second * bucket.first[index]});
            }
        }
        vec_basic v;
        v.insert(v.begin(), self.get_vars().begin(), self.get_vars().end());
        return Poly::from_dict(v, std::move(dict));
    } else {
        vec_basic vs;
        vs.insert(vs.begin(), self.get_vars().begin(), self.get_vars().end());
        return Poly::from_dict(vs, {{}});
    }
}

#ifndef debug_methods

void DiffVisitor::bvisit(const Basic &self)
{
    result_ = Derivative::create(self.rcp_from_this(), {x});
}

#else
// Here we do not have a 'Basic' fallback, but rather must implement all
// virtual methods explicitly (if we miss one, the code will not compile).
// This is useful to check that we have implemented all methods that we
// wanted.

#define DIFF0(CLASS)                                                           \
    void DiffVisitor::bvisit(const CLASS &self)                                \
    {                                                                          \
        result_ = Derivative::create(self.rcp_from_this(), {x});               \
    }

DIFF0(UnivariateSeries)
DIFF0(Max)
DIFF0(Min)
#endif

void DiffVisitor::bvisit(const Number &self)
{
    result_ = zero;
}

void DiffVisitor::bvisit(const Constant &self)
{
    result_ = zero;
}

void DiffVisitor::bvisit(const Symbol &self)
{
    if (x->get_name() == self.get_name()) {
        result_ = one;
    } else {
        result_ = zero;
    }
}

void DiffVisitor::bvisit(const Log &self)
{
    apply(self.get_arg());
    result_ = mul(div(one, self.get_arg()), result_);
}

void DiffVisitor::bvisit(const Abs &self)
{
    apply(self.get_arg());
    if (eq(*result_, *zero)) {
        result_ = zero;
    } else {
        result_ = Derivative::create(self.rcp_from_this(), {x});
    }
}

void DiffVisitor::bvisit(const Zeta &self)
{
    result_ = fdiff(self, x, *this);
}

void DiffVisitor::bvisit(const LowerGamma &self)
{
    result_ = fdiff(self, x, *this);
}
void DiffVisitor::bvisit(const UpperGamma &self)
{
    result_ = fdiff(self, x, *this);
}
void DiffVisitor::bvisit(const PolyGamma &self)
{
    result_ = fdiff(self, x, *this);
}
void DiffVisitor::bvisit(const UnevaluatedExpr &self)
{
    result_ = Derivative::create(self.rcp_from_this(), {x});
}
void DiffVisitor::bvisit(const TwoArgFunction &self)
{
    result_ = fdiff(self, x, *this);
}

void DiffVisitor::bvisit(const ASech &self)
{
    apply(self.get_arg());
    result_ = mul(div(minus_one, mul(sqrt(sub(one, pow(self.get_arg(), i2))),
                                     self.get_arg())),
                  result_);
}

void DiffVisitor::bvisit(const ACoth &self)
{
    apply(self.get_arg());
    result_ = mul(div(one, sub(one, pow(self.get_arg(), i2))), result_);
}

void DiffVisitor::bvisit(const ATanh &self)
{
    apply(self.get_arg());
    result_ = mul(div(one, sub(one, pow(self.get_arg(), i2))), result_);
}

void DiffVisitor::bvisit(const ACosh &self)
{
    apply(self.get_arg());
    result_ = mul(div(one, sqrt(sub(pow(self.get_arg(), i2), one))), result_);
}

void DiffVisitor::bvisit(const ACsch &self)
{
    apply(self.get_arg());
    result_ = mul(
        div(minus_one, mul(sqrt(add(one, div(one, pow(self.get_arg(), i2)))),
                           pow(self.get_arg(), i2))),
        result_);
}

void DiffVisitor::bvisit(const ASinh &self)
{
    apply(self.get_arg());
    result_ = mul(div(one, sqrt(add(pow(self.get_arg(), i2), one))), result_);
}

void DiffVisitor::bvisit(const Coth &self)
{
    apply(self.get_arg());
    result_ = mul(div(minus_one, pow(sinh(self.get_arg()), i2)), result_);
}

void DiffVisitor::bvisit(const Tanh &self)
{
    apply(self.get_arg());
    result_ = mul(sub(one, pow(tanh(self.get_arg()), i2)), result_);
}

void DiffVisitor::bvisit(const Sech &self)
{
    apply(self.get_arg());
    result_
        = mul(mul(mul(minus_one, sech(self.get_arg())), tanh(self.get_arg())),
              result_);
}

void DiffVisitor::bvisit(const Cosh &self)
{
    apply(self.get_arg());
    result_ = mul(sinh(self.get_arg()), result_);
}

void DiffVisitor::bvisit(const Csch &self)
{
    apply(self.get_arg());
    result_
        = mul(mul(mul(minus_one, csch(self.get_arg())), coth(self.get_arg())),
              result_);
}

void DiffVisitor::bvisit(const Sinh &self)
{
    apply(self.get_arg());
    result_ = mul(cosh(self.get_arg()), result_);
}

void DiffVisitor::bvisit(const Subs &self)
{
    RCP<const Basic> d = zero, t;
    if (self.get_dict().count(x) == 0) {
        apply(self.get_arg());
        d = result_->subs(self.get_dict());
    }
    for (const auto &p : self.get_dict()) {
        apply(p.second);
        t = result_;
        if (neq(*t, *zero)) {
            if (is_a<Symbol>(*p.first)) {
                d = add(d, mul(t, diff(self.get_arg(),
                                       rcp_static_cast<const Symbol>(p.first))
                                      ->subs(self.get_dict())));
            } else {
                result_ = Derivative::create(self.rcp_from_this(), {x});
                return;
            }
        }
    }
    result_ = d;
}

void DiffVisitor::bvisit(const Derivative &self)
{
    apply(self.get_arg());
    RCP<const Basic> ret = result_;
    if (eq(*ret, *zero)) {
        result_ = zero;
    }
    multiset_basic t = self.get_symbols();
    for (auto &p : t) {
        // If x is already there in symbols multi-set add x to the symbols
        // multi-set
        if (eq(*p, *x)) {
            t.insert(x);
            result_ = Derivative::create(self.get_arg(), t);
            return;
        }
    }
    // Avoid cycles
    if (is_a<Derivative>(*ret)
        && eq(*down_cast<const Derivative &>(*ret).get_arg(),
              *self.get_arg())) {
        t.insert(x);
        result_ = Derivative::create(self.get_arg(), t);
        return;
    }
    for (auto &p : t) {
        ret = diff(ret, rcp_static_cast<const Symbol>(p));
    }
    result_ = ret;
}

static inline RCP<const Symbol> get_dummy(const Basic &b, std::string name)
{
    RCP<const Symbol> s;
    do {
        name = "_" + name;
        s = symbol(name);
    } while (has_symbol(b, *s));
    return s;
}

void DiffVisitor::bvisit(const OneArgFunction &self)
{
    result_ = fdiff(self, x, *this);
}

void DiffVisitor::bvisit(const MultiArgFunction &self)
{
    result_ = fdiff(self, x, *this);
}

void DiffVisitor::bvisit(const LambertW &self)
{
    // check http://en.wikipedia.org/wiki/Lambert_W_function#Derivative
    // for the equation
    apply(self.get_arg());
    RCP<const Basic> lambertw_val = lambertw(self.get_arg());
    result_
        = mul(div(lambertw_val, mul(self.get_arg(), add(lambertw_val, one))),
              result_);
}

void DiffVisitor::bvisit(const Add &self)
{
    SymEngine::umap_basic_num d;
    RCP<const Number> coef = zero, coef2;
    RCP<const Basic> t;
    for (auto &p : self.get_dict()) {
        apply(p.first);
        RCP<const Basic> term = result_;
        if (is_a<Integer>(*term)
            && down_cast<const Integer &>(*term).is_zero()) {
            continue;
        } else if (is_a_Number(*term)) {
            iaddnum(outArg(coef),
                    mulnum(p.second, rcp_static_cast<const Number>(term)));
        } else if (is_a<Add>(*term)) {
            for (auto &q : (down_cast<const Add &>(*term)).get_dict())
                Add::dict_add_term(d, mulnum(q.second, p.second), q.first);
            iaddnum(outArg(coef),
                    mulnum(p.second, down_cast<const Add &>(*term).get_coef()));
        } else {
            Add::as_coef_term(mul(p.second, term), outArg(coef2), outArg(t));
            Add::dict_add_term(d, coef2, t);
        }
    }
    result_ = Add::from_dict(coef, std::move(d));
}

void DiffVisitor::bvisit(const Mul &self)
{
    RCP<const Number> overall_coef = zero;
    umap_basic_num add_dict;
    for (auto &p : self.get_dict()) {
        RCP<const Number> coef = self.get_coef();
        apply(pow(p.first, p.second));
        RCP<const Basic> factor = result_;
        if (is_a<Integer>(*factor)
            && down_cast<const Integer &>(*factor).is_zero())
            continue;
        map_basic_basic d = self.get_dict();
        d.erase(p.first);
        if (is_a_Number(*factor)) {
            imulnum(outArg(coef), rcp_static_cast<const Number>(factor));
        } else if (is_a<Mul>(*factor)) {
            RCP<const Mul> tmp = rcp_static_cast<const Mul>(factor);
            imulnum(outArg(coef), tmp->get_coef());
            for (auto &q : tmp->get_dict()) {
                Mul::dict_add_term_new(outArg(coef), d, q.second, q.first);
            }
        } else {
            RCP<const Basic> exp, t;
            Mul::as_base_exp(factor, outArg(exp), outArg(t));
            Mul::dict_add_term_new(outArg(coef), d, exp, t);
        }
        if (d.size() == 0) {
            iaddnum(outArg(overall_coef), coef);
        } else {
            RCP<const Basic> mul = Mul::from_dict(one, std::move(d));
            Add::coef_dict_add_term(outArg(overall_coef), add_dict, coef, mul);
        }
    }
    result_ = Add::from_dict(overall_coef, std::move(add_dict));
}

void DiffVisitor::bvisit(const Pow &self)
{
    if (is_a_Number(*(self.get_exp()))) {
        apply(self.get_base());
        result_ = mul(
            mul(self.get_exp(), pow(self.get_base(), sub(self.get_exp(), one))),
            result_);
    } else {
        apply(mul(self.get_exp(), log(self.get_base())));
        result_ = mul(self.rcp_from_this(), result_);
    }
}

void DiffVisitor::bvisit(const Sin &self)
{
    apply(self.get_arg());
    result_ = mul(cos(self.get_arg()), result_);
}

void DiffVisitor::bvisit(const Cos &self)
{
    apply(self.get_arg());
    result_ = mul(mul(minus_one, sin(self.get_arg())), result_);
}

void DiffVisitor::bvisit(const Tan &self)
{
    apply(self.get_arg());
    RCP<const Integer> two = integer(2);
    result_ = mul(add(one, pow(tan(self.get_arg()), two)), result_);
}

void DiffVisitor::bvisit(const Cot &self)
{
    apply(self.get_arg());
    RCP<const Integer> two = integer(2);
    result_
        = mul(mul(add(one, pow(cot(self.get_arg()), two)), minus_one), result_);
}

void DiffVisitor::bvisit(const Csc &self)
{
    apply(self.get_arg());
    result_ = mul(mul(mul(cot(self.get_arg()), csc(self.get_arg())), minus_one),
                  result_);
}

void DiffVisitor::bvisit(const Sec &self)
{
    apply(self.get_arg());
    result_ = mul(mul(tan(self.get_arg()), sec(self.get_arg())), result_);
}

void DiffVisitor::bvisit(const ASin &self)
{
    apply(self.get_arg());
    result_ = mul(div(one, sqrt(sub(one, pow(self.get_arg(), i2)))), result_);
}

void DiffVisitor::bvisit(const ACos &self)
{
    apply(self.get_arg());
    result_
        = mul(div(minus_one, sqrt(sub(one, pow(self.get_arg(), i2)))), result_);
}

void DiffVisitor::bvisit(const ASec &self)
{
    apply(self.get_arg());
    result_
        = mul(div(one, mul(pow(self.get_arg(), i2),
                           sqrt(sub(one, div(one, pow(self.get_arg(), i2)))))),
              result_);
}

void DiffVisitor::bvisit(const ACsc &self)
{
    apply(self.get_arg());
    result_ = mul(
        div(minus_one, mul(pow(self.get_arg(), i2),
                           sqrt(sub(one, div(one, pow(self.get_arg(), i2)))))),
        result_);
}

void DiffVisitor::bvisit(const ATan &self)
{
    apply(self.get_arg());
    result_ = mul(div(one, add(one, pow(self.get_arg(), i2))), result_);
}

void DiffVisitor::bvisit(const ACot &self)
{
    apply(self.get_arg());
    result_ = mul(div(minus_one, add(one, pow(self.get_arg(), i2))), result_);
}

void DiffVisitor::bvisit(const ATan2 &self)
{
    apply(div(self.get_num(), self.get_den()));
    result_ = mul(div(pow(self.get_den(), i2),
                      add(pow(self.get_den(), i2), pow(self.get_num(), i2))),
                  result_);
}

void DiffVisitor::bvisit(const Erf &self)
{
    apply(self.get_arg());
    result_ = mul(
        div(mul(integer(2), exp(neg(mul(self.get_arg(), self.get_arg())))),
            sqrt(pi)),
        result_);
}

void DiffVisitor::bvisit(const Erfc &self)
{
    apply(self.get_arg());
    result_ = neg(
        mul(div(mul(integer(2), exp(neg(mul(self.get_arg(), self.get_arg())))),
                sqrt(pi)),
            result_));
}

void DiffVisitor::bvisit(const Gamma &self)
{
    apply(self.get_arg());
    result_ = mul(mul(self.rcp_from_this(), polygamma(zero, self.get_arg())),
                  result_);
}

void DiffVisitor::bvisit(const LogGamma &self)
{
    apply(self.get_arg());
    result_ = mul(polygamma(zero, self.get_arg()), result_);
}

void DiffVisitor::bvisit(const UIntPoly &self)
{
    result_ = diff_upoly<UIntPoly, map_uint_mpz>(self, *x);
}

void DiffVisitor::bvisit(const URatPoly &self)
{
    result_ = diff_upoly<URatPoly, map_uint_mpq>(self, *x);
}

#ifdef HAVE_SYMENGINE_PIRANHA
void DiffVisitor::bvisit(const UIntPolyPiranha &self)
{
    result_ = diff_upoly<UIntPolyPiranha, map_uint_mpz>(self, *x);
}
void DiffVisitor::bvisit(const URatPolyPiranha &self)
{
    result_ = diff_upoly<URatPolyPiranha, map_uint_mpq>(self, *x);
}
#endif

#ifdef HAVE_SYMENGINE_FLINT
void DiffVisitor::bvisit(const UIntPolyFlint &self)
{
    result_ = diff_upolyflint(self, *x);
}

void DiffVisitor::bvisit(const URatPolyFlint &self)
{
    result_ = diff_upolyflint(self, *x);
}
#endif
void DiffVisitor::bvisit(const MIntPoly &self)
{
    result_ = diff_mpoly(self, x);
}

void DiffVisitor::bvisit(const MExprPoly &self)
{
    result_ = diff_mpoly(self, x);
}

void DiffVisitor::bvisit(const UExprPoly &self)
{
    result_ = diff_upoly<UExprPoly, map_int_Expr>(self, *x);
}

void DiffVisitor::bvisit(const FunctionWrapper &self)
{
    result_ = self.diff_impl(x);
}

void DiffVisitor::bvisit(const Beta &self)
{
    RCP<const Basic> beta_arg0 = self.get_args()[0];
    RCP<const Basic> beta_arg1 = self.get_args()[1];
    apply(beta_arg0);
    RCP<const Basic> diff_beta_arg0 = result_;
    apply(beta_arg1);
    RCP<const Basic> diff_beta_arg1 = result_;
    result_ = mul(self.rcp_from_this(),
                  add(mul(polygamma(zero, beta_arg0), diff_beta_arg0),
                      sub(mul(polygamma(zero, beta_arg1), diff_beta_arg1),
                          mul(polygamma(zero, add(beta_arg0, beta_arg1)),
                              add(diff_beta_arg0, diff_beta_arg1)))));
}

void DiffVisitor::bvisit(const Set &self)
{
    throw SymEngineException("Derivative doesn't exist.");
}

void DiffVisitor::bvisit(const Boolean &self)
{
    throw SymEngineException("Derivative doesn't exist.");
}

void DiffVisitor::bvisit(const GaloisField &self)
{
    GaloisFieldDict d;
    if (self.get_var()->__eq__(*x)) {
        d = self.get_poly().gf_diff();
        result_ = GaloisField::from_dict(self.get_var(), std::move(d));
    } else {
        result_ = GaloisField::from_dict(self.get_var(), std::move(d));
    }
}

void DiffVisitor::bvisit(const Piecewise &self)
{
    PiecewiseVec v = self.get_vec();
    for (auto &p : v) {
        apply(p.first);
        p.first = result_;
    }
    result_ = piecewise(std::move(v));
}

const RCP<const Basic> &DiffVisitor::apply(const Basic &b)
{
    apply(b.rcp_from_this());
    return result_;
}

const RCP<const Basic> &DiffVisitor::apply(const RCP<const Basic> &b)
{
    if (not cache) {
        b->accept(*this);
        return result_;
    }
    auto it = visited.find(b);
    if (it == visited.end()) {
        b->accept(*this);
        insert(visited, b, result_);
    } else {
        result_ = it->second;
    }
    return result_;
}

RCP<const Basic> diff(const RCP<const Basic> &arg, const RCP<const Symbol> &x,
                      bool cache)
{
    DiffVisitor v(x, cache);
    return v.apply(arg);
}

RCP<const Basic> Basic::diff(const RCP<const Symbol> &x, bool cache) const
{
    return SymEngine::diff(this->rcp_from_this(), x, cache);
}

//! SymPy style differentiation for non-symbol variables
// Since SymPy's differentiation makes no sense mathematically, it is
// defined separately here for compatibility
RCP<const Basic> sdiff(const RCP<const Basic> &arg, const RCP<const Basic> &x,
                       bool cache)
{
    if (is_a<Symbol>(*x)) {
        return arg->diff(rcp_static_cast<const Symbol>(x), cache);
    } else {
        RCP<const Symbol> d = get_dummy(*arg, "x");
        return ssubs(ssubs(arg, {{x, d}})->diff(d, cache), {{d, x}});
    }
}

} // SymEngine
