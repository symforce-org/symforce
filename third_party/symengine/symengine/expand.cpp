#include <symengine/visitor.h>

namespace SymEngine
{

inline RCP<const Number> _mulnum(const RCP<const Number> &x,
                                 const RCP<const Number> &y)
{
    if (eq(*x, *one))
        return y;
    if (eq(*y, *one))
        return x;
    return x->mul(*y);
}

inline void _imulnum(const Ptr<RCP<const Number>> &self,
                     const RCP<const Number> &other)
{
    *self = _mulnum(*self, other);
}

class ExpandVisitor : public BaseVisitor<ExpandVisitor>
{
private:
    umap_basic_num d_;
    RCP<const Number> coeff = zero;
    RCP<const Number> multiply = one;
    bool deep;

public:
    ExpandVisitor(bool deep_ = true) : deep(deep_)
    {
    }
    RCP<const Basic> apply(const Basic &b)
    {
        b.accept(*this);
        return Add::from_dict(coeff, std::move(d_));
    }

    void bvisit(const Basic &x)
    {
        Add::dict_add_term(d_, multiply, x.rcp_from_this());
    }

    void bvisit(const Number &x)
    {
        iaddnum(outArg(coeff),
                _mulnum(multiply, x.rcp_from_this_cast<const Number>()));
    }

    void bvisit(const Add &self)
    {
        RCP<const Number> _multiply = multiply;
        iaddnum(outArg(coeff), _mulnum(multiply, self.get_coef()));
        for (auto &p : self.get_dict()) {
            multiply = _mulnum(_multiply, p.second);
            if (deep) {
                p.first->accept(*this);
            } else {
                Add::dict_add_term(d_, multiply, p.first);
            }
        }
        multiply = _multiply;
    }

    void bvisit(const Mul &self)
    {
        for (auto &p : self.get_dict()) {
            if (!is_a<Symbol>(*p.first)) {
                RCP<const Basic> a, b;
                self.as_two_terms(outArg(a), outArg(b));
                a = expand_if_deep(a);
                b = expand_if_deep(b);
                mul_expand_two(a, b);
                return;
            }
        }
        this->_coef_dict_add_term(multiply, self.rcp_from_this());
    }

    void mul_expand_two(const RCP<const Basic> &a, const RCP<const Basic> &b)
    {
        // Both a and b are assumed to be expanded
        if (is_a<Add>(*a) && is_a<Add>(*b)) {
            iaddnum(outArg(coeff),
                    _mulnum(multiply,
                            _mulnum(down_cast<const Add &>(*a).get_coef(),
                                    down_cast<const Add &>(*b).get_coef())));
// Improves (x+1)**3*(x+2)**3*...(x+350)**3 expansion from 0.97s to 0.93s:
#if defined(HAVE_SYMENGINE_RESERVE)
            d_.reserve(d_.size()
                       + (down_cast<const Add &>(*a)).get_dict().size()
                             * (down_cast<const Add &>(*b)).get_dict().size());
#endif
            // Expand dicts first:
            for (auto &p : (down_cast<const Add &>(*a)).get_dict()) {
                RCP<const Number> temp = _mulnum(p.second, multiply);
                for (auto &q : (down_cast<const Add &>(*b)).get_dict()) {
                    // The main bottleneck here is the mul(p.first, q.first)
                    // command
                    RCP<const Basic> term = mul(p.first, q.first);
                    if (is_a_Number(*term)) {
                        iaddnum(outArg(coeff),
                                _mulnum(_mulnum(temp, q.second),
                                        rcp_static_cast<const Number>(term)));
                    } else {
                        if (is_a<Mul>(*term)
                            && !(down_cast<const Mul &>(*term)
                                     .get_coef()
                                     ->is_one())) {
                            // Tidy up things like {2x: 3} -> {x: 6}
                            RCP<const Number> coef2
                                = down_cast<const Mul &>(*term).get_coef();
                            // We make a copy of the dict_:
                            map_basic_basic d2
                                = down_cast<const Mul &>(*term).get_dict();
                            term = Mul::from_dict(one, std::move(d2));
                            Add::dict_add_term(
                                d_, _mulnum(_mulnum(temp, q.second), coef2),
                                term);
                        } else {
                            Add::dict_add_term(d_, _mulnum(temp, q.second),
                                               term);
                        }
                    }
                }
                Add::dict_add_term(
                    d_, _mulnum(down_cast<const Add &>(*b).get_coef(), temp),
                    p.first);
            }
            // Handle the coefficient of "a":
            RCP<const Number> temp
                = _mulnum(down_cast<const Add &>(*a).get_coef(), multiply);
            for (auto &q : (down_cast<const Add &>(*b)).get_dict()) {
                Add::dict_add_term(d_, _mulnum(temp, q.second), q.first);
            }
            return;
        } else if (is_a<Add>(*a)) {
            mul_expand_two(b, a);
            return;
        } else if (is_a<Add>(*b)) {
            RCP<const Number> a_coef;
            RCP<const Basic> a_term;
            Add::as_coef_term(a, outArg(a_coef), outArg(a_term));
            _imulnum(outArg(a_coef), multiply);

#if defined(HAVE_SYMENGINE_RESERVE)
            d_.reserve(d_.size()
                       + (down_cast<const Add &>(*b)).get_dict().size());
#endif
            for (auto &q : (down_cast<const Add &>(*b)).get_dict()) {
                RCP<const Basic> term = mul(a_term, q.first);
                if (is_a_Number(*term)) {
                    iaddnum(outArg(coeff),
                            _mulnum(_mulnum(q.second, a_coef),
                                    rcp_static_cast<const Number>(term)));
                } else {
                    if (is_a<Mul>(*term)
                        && !(down_cast<const Mul &>(*term)
                                 .get_coef()
                                 ->is_one())) {
                        // Tidy up things like {2x: 3} -> {x: 6}
                        RCP<const Number> coef2
                            = down_cast<const Mul &>(*term).get_coef();
                        // We make a copy of the dict_:
                        map_basic_basic d2
                            = down_cast<const Mul &>(*term).get_dict();
                        term = Mul::from_dict(one, std::move(d2));
                        Add::dict_add_term(
                            d_, _mulnum(_mulnum(q.second, a_coef), coef2),
                            term);
                    } else {
                        // TODO: check if it's a Add
                        Add::dict_add_term(d_, _mulnum(a_coef, q.second), term);
                    }
                }
            }
            if (eq(*a_term, *one)) {
                iaddnum(outArg(coeff),
                        _mulnum(down_cast<const Add &>(*b).get_coef(), a_coef));
            } else {
                Add::dict_add_term(
                    d_, _mulnum(down_cast<const Add &>(*b).get_coef(), a_coef),
                    a_term);
            }
            return;
        }
        _coef_dict_add_term(multiply, mul(a, b));
    }

    void square_expand(umap_basic_num &base_dict)
    {
        auto m = base_dict.size();
#if defined(HAVE_SYMENGINE_RESERVE)
        d_.reserve(d_.size() + m * (m + 1) / 2);
#endif
        RCP<const Basic> t;
        RCP<const Number> coef, two = integer(2);
        for (auto p = base_dict.begin(); p != base_dict.end(); ++p) {
            for (auto q = p; q != base_dict.end(); ++q) {
                if (q == p) {
                    _coef_dict_add_term(
                        _mulnum(mulnum((*p).second, (*p).second), multiply),
                        pow((*p).first, two));
                } else {
                    _coef_dict_add_term(
                        _mulnum(multiply, _mulnum((*p).second,
                                                  _mulnum((*q).second, two))),
                        mul((*q).first, (*p).first));
                }
            }
        }
    }

    void pow_expand(umap_basic_num &base_dict, unsigned n)
    {
        map_vec_mpz r;
        unsigned m = numeric_cast<unsigned>(base_dict.size());
        multinomial_coefficients_mpz(m, n, r);
// This speeds up overall expansion. For example for the benchmark
// (y + x + z + w)**60 it improves the timing from 135ms to 124ms.
#if defined(HAVE_SYMENGINE_RESERVE)
        d_.reserve(d_.size() + 2 * r.size());
#endif
        for (auto &p : r) {
            auto power = p.first.begin();
            auto i2 = base_dict.begin();
            map_basic_basic d;
            RCP<const Number> overall_coeff = one;
            for (; power != p.first.end(); ++power, ++i2) {
                if (*power > 0) {
                    RCP<const Integer> exp = integer(std::move(*power));
                    RCP<const Basic> base = i2->first;
                    if (is_a<Integer>(*base)) {
                        _imulnum(outArg(overall_coeff),
                                 rcp_static_cast<const Number>(
                                     down_cast<const Integer &>(*base).powint(
                                         *exp)));
                    } else if (is_a<Symbol>(*base)) {
                        Mul::dict_add_term(d, exp, base);
                    } else {
                        RCP<const Basic> exp2, t, tmp;
                        tmp = pow(base, exp);
                        if (is_a<Mul>(*tmp)) {
                            for (auto &p :
                                 (down_cast<const Mul &>(*tmp)).get_dict()) {
                                Mul::dict_add_term_new(outArg(overall_coeff), d,
                                                       p.second, p.first);
                            }
                            _imulnum(outArg(overall_coeff),
                                     (down_cast<const Mul &>(*tmp)).get_coef());
                        } else if (is_a_Number(*tmp)) {
                            _imulnum(outArg(overall_coeff),
                                     rcp_static_cast<const Number>(tmp));
                        } else {
                            Mul::as_base_exp(tmp, outArg(exp2), outArg(t));
                            Mul::dict_add_term_new(outArg(overall_coeff), d,
                                                   exp2, t);
                        }
                    }
                    if (!(i2->second->is_one())) {
                        _imulnum(outArg(overall_coeff),
                                 pownum(i2->second,
                                        rcp_static_cast<const Number>(exp)));
                    }
                }
            }
            RCP<const Basic> term = Mul::from_dict(overall_coeff, std::move(d));
            RCP<const Number> coef2 = integer(p.second);
            if (is_a_Number(*term)) {
                iaddnum(outArg(coeff),
                        _mulnum(_mulnum(multiply,
                                        rcp_static_cast<const Number>(term)),
                                coef2));
            } else {
                if (is_a<Mul>(*term)
                    && !(down_cast<const Mul &>(*term).get_coef()->is_one())) {
                    // Tidy up things like {2x: 3} -> {x: 6}
                    _imulnum(outArg(coef2),
                             down_cast<const Mul &>(*term).get_coef());
                    // We make a copy of the dict_:
                    map_basic_basic d2
                        = down_cast<const Mul &>(*term).get_dict();
                    term = Mul::from_dict(one, std::move(d2));
                }
                Add::dict_add_term(d_, _mulnum(multiply, coef2), term);
            }
        }
    }

    void bvisit(const Pow &self)
    {
        RCP<const Basic> _base = expand_if_deep(self.get_base());
        // TODO add all types of polys
        if (is_a<Integer>(*self.get_exp()) && is_a<UExprPoly>(*_base)) {
            unsigned q = numeric_cast<unsigned>(
                down_cast<const Integer &>(*self.get_exp()).as_uint());
            RCP<const UExprPoly> p = rcp_static_cast<const UExprPoly>(_base);
            RCP<const UExprPoly> r = pow_upoly(*p, q);
            _coef_dict_add_term(multiply, r);
            return;
        }
        if (is_a<Integer>(*self.get_exp()) && is_a<UIntPoly>(*_base)) {
            unsigned q = numeric_cast<unsigned>(
                down_cast<const Integer &>(*self.get_exp()).as_uint());
            RCP<const UIntPoly> p = rcp_static_cast<const UIntPoly>(_base);
            RCP<const UIntPoly> r = pow_upoly(*p, q);
            _coef_dict_add_term(multiply, r);
            return;
        }

        if (!is_a<Integer>(*self.get_exp()) || !is_a<Add>(*_base)) {
            if (neq(*_base, *self.get_base())) {
                Add::dict_add_term(d_, multiply, pow(_base, self.get_exp()));
            } else {
                Add::dict_add_term(d_, multiply, self.rcp_from_this());
            }
            return;
        }

        integer_class n
            = down_cast<const Integer &>(*self.get_exp()).as_integer_class();
        if (n < 0)
            return _coef_dict_add_term(
                multiply, div(one, expand_if_deep(pow(_base, integer(-n)))));
        RCP<const Add> base = rcp_static_cast<const Add>(_base);
        umap_basic_num base_dict = base->get_dict();
        if (!(base->get_coef()->is_zero())) {
            // Add the numerical coefficient into the dictionary. This
            // allows a little bit easier treatment below.
            insert(base_dict, base->get_coef(), one);
        } else
            iaddnum(outArg(coeff), base->get_coef());
        if (n == 2)
            return square_expand(base_dict);
        else
            return pow_expand(base_dict, numeric_cast<unsigned>(mp_get_ui(n)));
    }

    inline void _coef_dict_add_term(const RCP<const Number> &c,
                                    const RCP<const Basic> &term)
    {
        if (is_a_Number(*term)) {
            iaddnum(outArg(coeff),
                    _mulnum(c, rcp_static_cast<const Number>(term)));
        } else if (is_a<Add>(*term)) {
            for (const auto &q : (down_cast<const Add &>(*term)).get_dict())
                Add::dict_add_term(d_, _mulnum(q.second, c), q.first);
            iaddnum(outArg(coeff),
                    _mulnum(down_cast<const Add &>(*term).get_coef(), c));
        } else {
            RCP<const Number> coef2;
            RCP<const Basic> t;
            Add::as_coef_term(term, outArg(coef2), outArg(t));
            Add::dict_add_term(d_, _mulnum(c, coef2), t);
        }
    }

private:
    RCP<const Basic> expand_if_deep(const RCP<const Basic> &expr)
    {
        if (deep) {
            return expand(expr);
        } else {
            return expr;
        }
    }
};

//! Expands `self`
RCP<const Basic> expand(const RCP<const Basic> &self, bool deep)
{
    ExpandVisitor v(deep);
    return v.apply(*self);
}

} // SymEngine
