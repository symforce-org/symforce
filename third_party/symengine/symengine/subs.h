#ifndef SYMENGINE_SUBS_H
#define SYMENGINE_SUBS_H

#include <symengine/logic.h>
#include <symengine/visitor.h>

namespace SymEngine
{
// xreplace replaces subtrees of a node in the expression tree
// with a new subtree
RCP<const Basic> xreplace(const RCP<const Basic> &x,
                          const map_basic_basic &subs_dict, bool cache = true);
// subs substitutes expressions similar to xreplace, but keeps
// the mathematical equivalence for derivatives and subs
RCP<const Basic> subs(const RCP<const Basic> &x,
                      const map_basic_basic &subs_dict, bool cache = true);
// port of sympy.physics.mechanics.msubs where f'(x) and f(x)
// are considered independent
RCP<const Basic> msubs(const RCP<const Basic> &x,
                       const map_basic_basic &subs_dict, bool cache = true);
// port of sympy's subs where subs inside derivatives are done
RCP<const Basic> ssubs(const RCP<const Basic> &x,
                       const map_basic_basic &subs_dict, bool cache = true);

class XReplaceVisitor : public BaseVisitor<XReplaceVisitor>
{

protected:
    RCP<const Basic> result_;
    const map_basic_basic &subs_dict_;
    map_basic_basic visited;
    bool cache;

public:
    XReplaceVisitor(const map_basic_basic &subs_dict, bool cache = true)
        : subs_dict_(subs_dict), cache(cache)
    {
        if (cache) {
            visited = subs_dict;
        }
    }
    // TODO : Polynomials, Series, Sets
    void bvisit(const Basic &x)
    {
        result_ = x.rcp_from_this();
    }

    void bvisit(const Add &x)
    {
        SymEngine::umap_basic_num d;
        RCP<const Number> coef;

        auto it = subs_dict_.find(x.get_coef());
        if (it != subs_dict_.end()) {
            coef = zero;
            Add::coef_dict_add_term(outArg(coef), d, one, it->second);
        } else {
            coef = x.get_coef();
        }

        for (const auto &p : x.get_dict()) {
            auto it
                = subs_dict_.find(Add::from_dict(zero, {{p.first, p.second}}));
            if (it != subs_dict_.end()) {
                Add::coef_dict_add_term(outArg(coef), d, one, it->second);
            } else {
                it = subs_dict_.find(p.second);
                if (it != subs_dict_.end()) {
                    Add::coef_dict_add_term(outArg(coef), d, one,
                                            mul(it->second, apply(p.first)));
                } else {
                    Add::coef_dict_add_term(outArg(coef), d, p.second,
                                            apply(p.first));
                }
            }
        }
        result_ = Add::from_dict(coef, std::move(d));
    }

    void bvisit(const Mul &x)
    {
        RCP<const Number> coef = one;
        map_basic_basic d;
        for (const auto &p : x.get_dict()) {
            RCP<const Basic> factor_old;
            if (eq(*p.second, *one)) {
                factor_old = p.first;
            } else {
                factor_old = make_rcp<Pow>(p.first, p.second);
            }
            RCP<const Basic> factor = apply(factor_old);
            if (factor == factor_old) {
                // TODO: Check if Mul::dict_add_term is enough
                Mul::dict_add_term_new(outArg(coef), d, p.second, p.first);
            } else if (is_a_Number(*factor)) {
                if (down_cast<const Number &>(*factor).is_zero()) {
                    result_ = factor;
                    return;
                }
                imulnum(outArg(coef), rcp_static_cast<const Number>(factor));
            } else if (is_a<Mul>(*factor)) {
                RCP<const Mul> tmp = rcp_static_cast<const Mul>(factor);
                imulnum(outArg(coef), tmp->get_coef());
                for (const auto &q : tmp->get_dict()) {
                    Mul::dict_add_term_new(outArg(coef), d, q.second, q.first);
                }
            } else {
                RCP<const Basic> exp, t;
                Mul::as_base_exp(factor, outArg(exp), outArg(t));
                Mul::dict_add_term_new(outArg(coef), d, exp, t);
            }
        }

        // Replace the coefficient
        RCP<const Basic> factor = apply(x.get_coef());
        RCP<const Basic> exp, t;
        Mul::as_base_exp(factor, outArg(exp), outArg(t));
        Mul::dict_add_term_new(outArg(coef), d, exp, t);

        result_ = Mul::from_dict(coef, std::move(d));
    }

    void bvisit(const Pow &x)
    {
        RCP<const Basic> base_new = apply(x.get_base());
        RCP<const Basic> exp_new = apply(x.get_exp());
        if (base_new == x.get_base() and exp_new == x.get_exp()) {
            result_ = x.rcp_from_this();
        } else {
            result_ = pow(base_new, exp_new);
        }
    }

    void bvisit(const OneArgFunction &x)
    {
        apply(x.get_arg());
        if (result_ == x.get_arg()) {
            result_ = x.rcp_from_this();
        } else {
            result_ = x.create(result_);
        }
    }

    template <class T>
    void bvisit(const TwoArgBasic<T> &x)
    {
        RCP<const Basic> a = apply(x.get_arg1());
        RCP<const Basic> b = apply(x.get_arg2());
        if (a == x.get_arg1() and b == x.get_arg2())
            result_ = x.rcp_from_this();
        else
            result_ = x.create(a, b);
    }

    void bvisit(const MultiArgFunction &x)
    {
        vec_basic v = x.get_args();
        for (auto &elem : v) {
            elem = apply(elem);
        }
        result_ = x.create(v);
    }

    void bvisit(const FunctionSymbol &x)
    {
        vec_basic v = x.get_args();
        for (auto &elem : v) {
            elem = apply(elem);
        }
        result_ = x.create(v);
    }

    void bvisit(const Contains &x)
    {
        RCP<const Basic> a = apply(x.get_expr());
        auto c = apply(x.get_set());
        if (not is_a_Set(*c))
            throw SymEngineException("expected an object of type Set");
        RCP<const Set> b = rcp_static_cast<const Set>(c);
        if (a == x.get_expr() and b == x.get_set())
            result_ = x.rcp_from_this();
        else
            result_ = x.create(a, b);
    }

    void bvisit(const And &x)
    {
        set_boolean v;
        for (const auto &elem : x.get_container()) {
            auto a = apply(elem);
            if (not is_a_Boolean(*a))
                throw SymEngineException("expected an object of type Boolean");
            v.insert(rcp_static_cast<const Boolean>(a));
        }
        result_ = x.create(v);
    }

    void bvisit(const FiniteSet &x)
    {
        set_basic v;
        for (const auto &elem : x.get_container()) {
            v.insert(apply(elem));
        }
        result_ = x.create(v);
    }

    void bvisit(const ImageSet &x)
    {
        RCP<const Basic> s = apply(x.get_symbol());
        RCP<const Basic> expr = apply(x.get_expr());
        auto bs_ = apply(x.get_baseset());
        if (not is_a_Set(*bs_))
            throw SymEngineException("expected an object of type Set");
        RCP<const Set> bs = rcp_static_cast<const Set>(bs_);
        if (s == x.get_symbol() and expr == x.get_expr()
            and bs == x.get_baseset()) {
            result_ = x.rcp_from_this();
        } else {
            result_ = x.create(s, expr, bs);
        }
    }

    void bvisit(const Union &x)
    {
        set_set v;
        for (const auto &elem : x.get_container()) {
            auto a = apply(elem);
            if (not is_a_Set(*a))
                throw SymEngineException("expected an object of type Set");
            v.insert(rcp_static_cast<const Set>(a));
        }
        result_ = x.create(v);
    }

    void bvisit(const Piecewise &pw)
    {
        PiecewiseVec pwv;
        pwv.reserve(pw.get_vec().size());
        for (const auto &expr_pred : pw.get_vec()) {
            const auto expr = apply(*expr_pred.first);
            const auto pred = apply(*expr_pred.second);
            pwv.emplace_back(
                std::make_pair(expr, rcp_static_cast<const Boolean>(pred)));
        }
        result_ = piecewise(std::move(pwv));
    }

    void bvisit(const Derivative &x)
    {
        auto expr = apply(x.get_arg());
        for (const auto &sym : x.get_symbols()) {
            auto s = apply(sym);
            if (not is_a<Symbol>(*s)) {
                throw SymEngineException("expected an object of type Symbol");
            }
            expr = expr->diff(rcp_static_cast<const Symbol>(s));
        }
        result_ = expr;
    }

    void bvisit(const Subs &x)
    {
        auto expr = apply(x.get_arg());
        map_basic_basic new_subs_dict;
        for (const auto &sym : x.get_dict()) {
            insert(new_subs_dict, apply(sym.first), apply(sym.second));
        }
        result_ = subs(expr, new_subs_dict);
    }

    RCP<const Basic> apply(const Basic &x)
    {
        return apply(x.rcp_from_this());
    }

    RCP<const Basic> apply(const RCP<const Basic> &x)
    {
        if (cache) {
            auto it = visited.find(x);
            if (it != visited.end()) {
                result_ = it->second;
            } else {
                x->accept(*this);
                insert(visited, x, result_);
            }
        } else {
            auto it = subs_dict_.find(x);
            if (it != subs_dict_.end()) {
                result_ = it->second;
            } else {
                x->accept(*this);
            }
        }
        return result_;
    }
};

//! Mappings in the `subs_dict` are applied to the expression tree of `x`
inline RCP<const Basic> xreplace(const RCP<const Basic> &x,
                                 const map_basic_basic &subs_dict, bool cache)
{
    XReplaceVisitor s(subs_dict, cache);
    return s.apply(x);
}

class SubsVisitor : public BaseVisitor<SubsVisitor, XReplaceVisitor>
{
public:
    using XReplaceVisitor::bvisit;

    SubsVisitor(const map_basic_basic &subs_dict_, bool cache = true)
        : BaseVisitor<SubsVisitor, XReplaceVisitor>(subs_dict_, cache)
    {
    }

    void bvisit(const Pow &x)
    {
        RCP<const Basic> base_new = apply(x.get_base());
        RCP<const Basic> exp_new = apply(x.get_exp());
        if (subs_dict_.size() == 1 and is_a<Pow>(*((*subs_dict_.begin()).first))
            and not is_a<Add>(
                    *down_cast<const Pow &>(*(*subs_dict_.begin()).first)
                         .get_exp())) {
            auto &subs_first
                = down_cast<const Pow &>(*(*subs_dict_.begin()).first);
            if (eq(*subs_first.get_base(), *base_new)) {
                auto newexpo = div(exp_new, subs_first.get_exp());
                if (is_a_Number(*newexpo) or is_a<Constant>(*newexpo)) {
                    result_ = pow((*subs_dict_.begin()).second, newexpo);
                    return;
                }
            }
        }
        if (base_new == x.get_base() and exp_new == x.get_exp()) {
            result_ = x.rcp_from_this();
        } else {
            result_ = pow(base_new, exp_new);
        }
    }

    void bvisit(const Derivative &x)
    {
        RCP<const Symbol> s;
        map_basic_basic m, n;
        bool subs;

        for (const auto &p : subs_dict_) {
            // If the derivative arg is to be replaced in its entirety, allow
            // it.
            if (eq(*x.get_arg(), *p.first)) {
                RCP<const Basic> t = p.second;
                for (auto &sym : x.get_symbols()) {
                    if (not is_a<Symbol>(*sym)) {
                        throw SymEngineException("Error, expected a Symbol.");
                    }
                    t = t->diff(rcp_static_cast<const Symbol>(sym));
                }
                result_ = t;
                return;
            }
        }
        for (const auto &p : subs_dict_) {
            subs = true;
            if (eq(*x.get_arg()->subs({{p.first, p.second}}), *x.get_arg()))
                continue;

            // If p.first and p.second are symbols and arg_ is
            // independent of p.second, p.first can be replaced
            if (is_a<Symbol>(*p.first) and is_a<Symbol>(*p.second)
                and eq(*x.get_arg()->diff(
                           rcp_static_cast<const Symbol>(p.second)),
                       *zero)) {
                insert(n, p.first, p.second);
                continue;
            }
            for (const auto &d : x.get_symbols()) {
                if (is_a<Symbol>(*d)) {
                    s = rcp_static_cast<const Symbol>(d);
                    // If p.first or p.second has non zero derivates wrt to s
                    // p.first cannot be replaced
                    if (neq(*zero, *(p.first->diff(s)))
                        || neq(*zero, *(p.second->diff(s)))) {
                        subs = false;
                        break;
                    }
                } else {
                    result_
                        = make_rcp<const Subs>(x.rcp_from_this(), subs_dict_);
                    return;
                }
            }
            if (subs) {
                insert(n, p.first, p.second);
            } else {
                insert(m, p.first, p.second);
            }
        }
        auto t = x.get_arg()->subs(n);
        for (auto &p : x.get_symbols()) {
            auto t2 = p->subs(n);
            if (not is_a<Symbol>(*t2)) {
                throw SymEngineException("Error, expected a Symbol.");
            }
            t = t->diff(rcp_static_cast<const Symbol>(t2));
        }
        if (m.empty()) {
            result_ = t;
        } else {
            result_ = make_rcp<const Subs>(t, m);
        }
    }

    void bvisit(const Subs &x)
    {
        map_basic_basic m, n;
        for (const auto &p : subs_dict_) {
            bool found = false;
            for (const auto &s : x.get_dict()) {
                if (neq(*(s.first->subs({{p.first, p.second}})), *(s.first))) {
                    found = true;
                    break;
                }
            }
            // If p.first is not replaced in arg_ by dict_,
            // store p.first in n to replace in arg_
            if (not found) {
                insert(n, p.first, p.second);
            }
        }
        for (const auto &s : x.get_dict()) {
            insert(m, s.first, apply(s.second));
        }
        RCP<const Basic> presub = x.get_arg()->subs(n);
        if (is_a<Subs>(*presub)) {
            for (auto &q : down_cast<const Subs &>(*presub).get_dict()) {
                insert(m, q.first, q.second);
            }
            result_ = down_cast<const Subs &>(*presub).get_arg()->subs(m);
        } else {
            result_ = presub->subs(m);
        }
    }
};

class MSubsVisitor : public BaseVisitor<MSubsVisitor, XReplaceVisitor>
{
public:
    using XReplaceVisitor::bvisit;

    MSubsVisitor(const map_basic_basic &d, bool cache = true)
        : BaseVisitor<MSubsVisitor, XReplaceVisitor>(d, cache)
    {
    }

    void bvisit(const Derivative &x)
    {
        result_ = x.rcp_from_this();
    }

    void bvisit(const Subs &x)
    {
        map_basic_basic m = x.get_dict();
        for (const auto &p : subs_dict_) {
            m[p.first] = p.second;
        }
        result_ = msubs(x.get_arg(), m);
    }
};

class SSubsVisitor : public BaseVisitor<SSubsVisitor, SubsVisitor>
{
public:
    using XReplaceVisitor::bvisit;

    SSubsVisitor(const map_basic_basic &d, bool cache = true)
        : BaseVisitor<SSubsVisitor, SubsVisitor>(d, cache)
    {
    }

    void bvisit(const Derivative &x)
    {
        apply(x.get_arg());
        auto t = result_;
        multiset_basic m;
        for (auto &p : x.get_symbols()) {
            apply(p);
            m.insert(result_);
        }
        result_ = Derivative::create(t, m);
    }

    void bvisit(const Subs &x)
    {
        map_basic_basic m = x.get_dict();
        for (const auto &p : subs_dict_) {
            m[p.first] = p.second;
        }
        result_ = ssubs(x.get_arg(), m);
    }
};

//! Subs which treat f(t) and Derivative(f(t), t) as separate variables
inline RCP<const Basic> msubs(const RCP<const Basic> &x,
                              const map_basic_basic &subs_dict, bool cache)
{
    MSubsVisitor s(subs_dict, cache);
    return s.apply(x);
}

//! SymPy compatible subs
inline RCP<const Basic> ssubs(const RCP<const Basic> &x,
                              const map_basic_basic &subs_dict, bool cache)
{
    SSubsVisitor s(subs_dict, cache);
    return s.apply(x);
}

inline RCP<const Basic> subs(const RCP<const Basic> &x,
                             const map_basic_basic &subs_dict, bool cache)
{
    SubsVisitor b(subs_dict, cache);
    return b.apply(x);
}

} // namespace SymEngine

#endif // SYMENGINE_SUBS_H
