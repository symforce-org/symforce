#ifndef SYMENGINE_BASIC_CONVERSIONS_H
#define SYMENGINE_BASIC_CONVERSIONS_H

#include <symengine/visitor.h>

namespace SymEngine
{

// convert a `basic`, to a UPoly `P` (eg. UIntPoly, UExprPoly, UIntPolyFlint)
// using `gen` as the genarator. Throws, if poly constructions not possible.
// `ex` is the optional parameter for expanding the given `basic` or not.
template <typename P>
RCP<const P> from_basic(const RCP<const Basic> &basic,
                        const RCP<const Basic> &gen, bool ex = false);
// convert a `basic`, to a UPoly `P` (eg. UIntPoly, UExprPoly, UIntPolyFlint)
// after finding out the generator automatically. Throws, if number
// of generators found != 1, or poly construction not possible.
// `ex` is the optional parameter for expanding the given `basic` or not.

template <typename P>
enable_if_t<is_a_UPoly<P>::value, RCP<const P>>
from_basic(const RCP<const Basic> &basic, bool ex = false);

template <typename T, typename P>
enable_if_t<std::is_same<T, UExprDict>::value, T>
_basic_to_upoly(const RCP<const Basic> &basic, const RCP<const Basic> &gen);

template <typename T, typename P>
enable_if_t<std::is_base_of<UIntPolyBase<T, P>, P>::value, T>
_basic_to_upoly(const RCP<const Basic> &basic, const RCP<const Basic> &gen);

template <typename T, typename P>
enable_if_t<std::is_base_of<URatPolyBase<T, P>, P>::value, T>
_basic_to_upoly(const RCP<const Basic> &basic, const RCP<const Basic> &gen);

template <typename P, typename V>
class BasicToUPolyBase : public BaseVisitor<V>
{
public:
    RCP<const Basic> gen;
    using D = typename P::container_type;
    D dict;

    BasicToUPolyBase(const RCP<const Basic> &gen_)
    {
        gen = gen_;
    }

    D apply(const Basic &b)
    {
        b.accept(*this);
        return std::move(dict);
    }

    void dict_set(unsigned int pow, const Basic &x)
    {
        down_cast<V *>(this)->dict_set(pow, x);
    }

    void bvisit(const Pow &x)
    {
        if (is_a<const Integer>(*x.get_exp())) {
            int i = numeric_cast<int>(
                down_cast<const Integer &>(*x.get_exp()).as_int());
            if (i > 0) {
                dict
                    = pow_upoly(*P::from_container(gen, _basic_to_upoly<D, P>(
                                                            x.get_base(), gen)),
                                i)
                          ->get_poly();
                return;
            }
        }

        RCP<const Basic> genbase = gen, genpow = one, coef = one, tmp;
        if (is_a<const Pow>(*gen)) {
            genbase = down_cast<const Pow &>(*gen).get_base();
            genpow = down_cast<const Pow &>(*gen).get_exp();
        }

        if (eq(*genbase, *x.get_base())) {

            set_basic expos;

            if (is_a<const Add>(*x.get_exp())) {
                RCP<const Add> addx = rcp_static_cast<const Add>(x.get_exp());
                for (auto const &it : addx->get_dict())
                    expos.insert(mul(it.first, it.second));
                if (not addx->get_coef()->is_zero())
                    expos.insert(addx->get_coef());
            } else {
                expos.insert(x.get_exp());
            }

            int powr = 0;
            for (auto const &it : expos) {
                tmp = div(it, genpow);
                if (is_a<const Integer>(*tmp)) {
                    RCP<const Integer> i = rcp_static_cast<const Integer>(tmp);
                    if (i->is_positive()) {
                        powr = static_cast<int>(i->as_int());
                        continue;
                    }
                }
                coef = mul(coef, pow(genbase, it));
            }
            dict_set(powr, *coef);
        } else {
            this->bvisit((const Basic &)x);
        }
    }

    void bvisit(const Add &x)
    {
        D res = apply(*x.get_coef());
        for (auto const &it : x.get_dict())
            res += apply(*it.first) * apply(*it.second);
        dict = std::move(res);
    }

    void bvisit(const Mul &x)
    {
        D res = apply(*x.get_coef());
        for (auto const &it : x.get_dict())
            res *= apply(*pow(it.first, it.second));
        dict = std::move(res);
    }

    void bvisit(const Integer &x)
    {
        integer_class i = x.as_integer_class();
        dict = P::container_from_dict(gen, {{0, typename P::coef_type(i)}});
    }

    template <
        typename Poly,
        typename = enable_if_t<
            ((std::is_base_of<UIntPolyBase<typename P::container_type, P>,
                              P>::value
              and std::is_base_of<
                  UIntPolyBase<typename Poly::container_type, Poly>,
                  Poly>::value)
             or (std::is_base_of<URatPolyBase<typename P::container_type, P>,
                                 P>::value
                 and (std::is_base_of<
                          UIntPolyBase<typename Poly::container_type, Poly>,
                          Poly>::value
                      or std::is_base_of<
                          URatPolyBase<typename Poly::container_type, Poly>,
                          Poly>::value))
             or (std::is_same<P, UExprPoly>::value
                 and std::is_base_of<
                     UPolyBase<typename Poly::container_type, Poly>,
                     Poly>::value))
            and not std::is_same<Poly, GaloisField>::value>>
    void bvisit(const Poly &x)
    {
        dict = (P::from_poly(x))->get_poly();
    }

    void bvisit(const Basic &x)
    {
        RCP<const Basic> genpow = one, genbase = gen, powr;
        if (is_a<const Pow>(*gen)) {
            genpow = down_cast<const Pow &>(*gen).get_exp();
            genbase = down_cast<const Pow &>(*gen).get_base();
        }
        if (eq(*genbase, x)) {
            powr = div(one, genpow);
            if (is_a<const Integer>(*powr)) {
                int i = numeric_cast<int>(
                    down_cast<const Integer &>(*powr).as_int());
                if (i > 0) {
                    dict = P::container_from_dict(
                        gen, {{i, typename P::coef_type(1)}});
                    return;
                }
            }
        }
        if (is_a<const Symbol>(*gen)) {
            if (has_symbol(x, *gen)) {
                throw SymEngineException("Not a Polynomial");
            }
        }
        dict_set(0, x);
    }
};

template <typename Poly>
class BasicToUIntPoly : public BasicToUPolyBase<Poly, BasicToUIntPoly<Poly>>
{
public:
    using BasicToUPolyBase<Poly, BasicToUIntPoly>::bvisit;
    using BasicToUPolyBase<Poly, BasicToUIntPoly>::apply;

    BasicToUIntPoly(const RCP<const Basic> &gen)
        : BasicToUPolyBase<Poly, BasicToUIntPoly<Poly>>(gen)
    {
    }

    void bvisit(const Rational &x)
    {
        throw SymEngineException("Non-integer found");
    }

    void dict_set(unsigned int pow, const Basic &x)
    {
        if (is_a<const Integer>(x))
            this->dict = Poly::container_from_dict(
                this->gen,
                {{pow, down_cast<const Integer &>(x).as_integer_class()}});
        else
            throw SymEngineException("Non-integer found");
    }
};

class BasicToUExprPoly : public BasicToUPolyBase<UExprPoly, BasicToUExprPoly>
{
public:
    using BasicToUPolyBase<UExprPoly, BasicToUExprPoly>::bvisit;
    using BasicToUPolyBase<UExprPoly, BasicToUExprPoly>::apply;

    BasicToUExprPoly(const RCP<const Basic> &gen) : BasicToUPolyBase(gen) {}

    void bvisit(const Rational &x)
    {
        dict = UExprDict(x.rcp_from_this());
    }

    void dict_set(unsigned int pow, const Basic &x)
    {
        dict = UExprDict({{pow, x.rcp_from_this()}});
    }
};

template <typename Poly>
class BasicToURatPoly : public BasicToUPolyBase<Poly, BasicToURatPoly<Poly>>
{
public:
    using BasicToUPolyBase<Poly, BasicToURatPoly>::bvisit;
    using BasicToUPolyBase<Poly, BasicToURatPoly>::apply;

    BasicToURatPoly(const RCP<const Basic> &gen)
        : BasicToUPolyBase<Poly, BasicToURatPoly<Poly>>(gen)
    {
    }

    void bvisit(const Rational &x)
    {
        this->dict = URatDict(x.as_rational_class());
    }

    void dict_set(unsigned int pow, const Basic &x)
    {
        if (is_a<const Integer>(x))
            this->dict = Poly::container_from_dict(
                this->gen, {{pow, rational_class(static_cast<const Integer &>(x)
                                                     .as_integer_class())}});
        else if (is_a<const Rational>(x))
            this->dict = Poly::container_from_dict(
                this->gen,
                {{pow, static_cast<const Rational &>(x).as_rational_class()}});
        else
            throw SymEngineException("Non-rational found");
    }
};

template <typename T, typename P>
enable_if_t<std::is_same<T, UExprDict>::value, T>
_basic_to_upoly(const RCP<const Basic> &basic, const RCP<const Basic> &gen)
{
    BasicToUExprPoly v(gen);
    return v.apply(*basic);
}

template <typename T, typename P>
enable_if_t<std::is_base_of<UIntPolyBase<T, P>, P>::value, T>
_basic_to_upoly(const RCP<const Basic> &basic, const RCP<const Basic> &gen)
{
    BasicToUIntPoly<P> v(gen);
    return v.apply(*basic);
}

template <typename T, typename P>
enable_if_t<std::is_base_of<URatPolyBase<T, P>, P>::value, T>
_basic_to_upoly(const RCP<const Basic> &basic, const RCP<const Basic> &gen)
{
    BasicToURatPoly<P> v(gen);
    return v.apply(*basic);
}

template <typename P>
RCP<const P> from_basic(const RCP<const Basic> &basic,
                        const RCP<const Basic> &gen, bool ex)
{
    RCP<const Basic> exp = basic;
    if (ex)
        exp = expand(basic);
    return P::from_container(
        gen, _basic_to_upoly<typename P::container_type, P>(exp, gen));
}

template <typename P>
enable_if_t<is_a_UPoly<P>::value, RCP<const P>>
from_basic(const RCP<const Basic> &basic, bool ex)
{
    RCP<const Basic> exp = basic;
    if (ex)
        exp = expand(basic);

    umap_basic_num tmp = _find_gens_poly(exp);

    if (tmp.size() != 1)
        throw SymEngineException("Did not find exactly 1 generator");

    RCP<const Basic> gen = pow(tmp.begin()->first, tmp.begin()->second);
    return P::from_container(
        gen, _basic_to_upoly<typename P::container_type, P>(exp, gen));
}

template <typename P>
enable_if_t<std::is_same<MIntPoly, P>::value, typename P::container_type>
_basic_to_mpoly(const RCP<const Basic> &basic, const set_basic &gens);

template <typename P, typename V>
class BasicToMPolyBase : public BaseVisitor<V>
{
public:
    using Dict = typename P::container_type;
    using Vec = typename Dict::vec_type;
    Dict dict;
    set_basic gens;
    std::unordered_map<RCP<const Basic>, vec_basic, RCPBasicHash, RCPBasicKeyEq>
        gens_pow;
    umap_basic_uint gens_map;

    BasicToMPolyBase(const set_basic &gens_)
    {
        gens = gens_;
        dict.vec_size = static_cast<int>(gens.size());

        RCP<const Basic> genpow, genbase;
        unsigned int i = 0;

        for (auto it : gens) {
            genpow = one;
            genbase = it;
            if (is_a<const Pow>(*it)) {
                genpow = down_cast<const Pow &>(*it).get_exp();
                genbase = down_cast<const Pow &>(*it).get_base();
            }
            auto ite = gens_pow.find(genbase);
            if (ite == gens_pow.end())
                gens_pow[genbase] = {genpow};
            else
                gens_pow[genbase].push_back(genpow);
            gens_map[it] = i++;
        }
    }

    Dict apply(const Basic &b)
    {
        b.accept(*this);
        return std::move(dict);
    }

    void dict_set(Vec pow, const Basic &x)
    {
        down_cast<V *>(this)->dict_set(pow, x);
    }

    void bvisit(const Pow &x)
    {
        if (is_a<const Integer>(*x.get_exp())) {
            int i = numeric_cast<int>(
                down_cast<const Integer &>(*x.get_exp()).as_int());
            if (i > 0) {
                dict = Dict::pow(_basic_to_mpoly<P>(x.get_base(), gens), i);
                return;
            }
        }

        Vec zero_v(gens.size(), 0);
        RCP<const Basic> coef = one, tmp;
        RCP<const Integer> i;
        bool found;
        auto ite = gens_pow.find(x.get_base());

        if (ite != gens_pow.end()) {

            set_basic expos;

            if (is_a<const Add>(*x.get_exp())) {
                RCP<const Add> addx = rcp_static_cast<const Add>(x.get_exp());
                for (auto const &it : addx->get_dict())
                    expos.insert(mul(it.first, it.second));
                if (not addx->get_coef()->is_zero())
                    expos.insert(addx->get_coef());
            } else {
                expos.insert(x.get_exp());
            }

            for (auto const &it : expos) {

                found = false;

                for (auto powr : ite->second) {
                    tmp = div(it, powr);
                    if (is_a<const Integer>(*tmp)) {
                        i = rcp_static_cast<const Integer>(tmp);
                        if (i->is_positive()) {
                            zero_v[gens_map[pow(ite->first, powr)]]
                                = static_cast<int>(i->as_int());
                            found = true;
                            break;
                        }
                    }
                }

                if (not found)
                    coef = mul(coef, pow(ite->first, it));
            }
            dict_set(zero_v, *coef);

        } else {
            dict_set(zero_v, x);
        }
    }

    void bvisit(const Add &x)
    {
        Dict res = apply(*x.get_coef());
        for (auto const &it : x.get_dict())
            res += apply(*it.first) * apply(*it.second);
        dict = std::move(res);
    }

    void bvisit(const Mul &x)
    {
        Dict res = apply(*x.get_coef());
        for (auto const &it : x.get_dict())
            res *= apply(*pow(it.first, it.second));
        dict = std::move(res);
    }

    void bvisit(const Integer &x)
    {
        integer_class i = x.as_integer_class();
        Vec zero_v(gens.size(), 0);
        dict = P::container_from_dict(gens, {{zero_v, i}});
    }

    void bvisit(const Basic &x)
    {
        RCP<const Basic> powr;
        Vec zero_v(gens.size(), 0);

        auto it = gens_pow.find(x.rcp_from_this());
        if (it != gens_pow.end()) {

            for (auto pows : it->second) {
                powr = div(one, pows);
                if (is_a<const Integer>(*powr)) {
                    int i = numeric_cast<int>(
                        down_cast<const Integer &>(*powr).as_int());
                    if (i > 0) {
                        // can be optimized
                        zero_v[gens_map[pow(it->first, pows)]] = i;
                        dict = P::container_from_dict(
                            gens, {{zero_v, typename P::coef_type(1)}});
                        return;
                    }
                }
            }
        }

        dict_set(zero_v, x);
    }
};

class BasicToMIntPoly : public BasicToMPolyBase<MIntPoly, BasicToMIntPoly>
{
public:
    using BasicToMPolyBase<MIntPoly, BasicToMIntPoly>::bvisit;
    using BasicToMPolyBase<MIntPoly, BasicToMIntPoly>::apply;

    BasicToMIntPoly(const set_basic &gens) : BasicToMPolyBase(gens) {}

    void bvisit(const Rational &x)
    {
        throw SymEngineException("Non-integer found");
    }

    void dict_set(vec_uint pow, const Basic &x)
    {
        if (is_a<const Integer>(x))
            dict = MIntPoly::container_from_dict(
                gens,
                {{pow, down_cast<const Integer &>(x).as_integer_class()}});
        else
            throw SymEngineException("Non-integer found");
    }
};

class BasicToMExprPoly : public BasicToMPolyBase<MExprPoly, BasicToMExprPoly>
{
public:
    using BasicToMPolyBase<MExprPoly, BasicToMExprPoly>::bvisit;
    using BasicToMPolyBase<MExprPoly, BasicToMExprPoly>::apply;

    BasicToMExprPoly(const set_basic &gens) : BasicToMPolyBase(gens) {}

    void bvisit(const Rational &x)
    {
        Vec v(gens.size(), 0);
        dict = MExprPoly::container_from_dict(gens, {{v, x.rcp_from_this()}});
    }

    void dict_set(vec_int pow, const Basic &x)
    {
        dict = MExprPoly::container_from_dict(gens, {{pow, x.rcp_from_this()}});
    }
};

template <typename P>
enable_if_t<std::is_same<MIntPoly, P>::value, typename P::container_type>
_basic_to_mpoly(const RCP<const Basic> &basic, const set_basic &gens)
{
    BasicToMIntPoly v(gens);
    return v.apply(*basic);
}

template <typename P>
enable_if_t<std::is_same<MExprPoly, P>::value, typename P::container_type>
_basic_to_mpoly(const RCP<const Basic> &basic, const set_basic &gens)
{
    BasicToMExprPoly v(gens);
    return v.apply(*basic);
}

template <typename P>
RCP<const P> from_basic(const RCP<const Basic> &basic, set_basic &gens,
                        bool ex = false)
{
    RCP<const Basic> exp = basic;
    if (ex)
        exp = expand(basic);
    // need to add a check to see if generators are valid
    // for eg. we dont want x and x**2 as the gens
    return P::from_container(gens, _basic_to_mpoly<P>(exp, gens));
}

template <typename P>
enable_if_t<
    std::is_base_of<MSymEnginePoly<typename P::container_type, P>, P>::value,
    RCP<const P>>
from_basic(const RCP<const Basic> &basic, bool ex = false)
{
    RCP<const Basic> exp = basic;
    if (ex)
        exp = expand(basic);

    umap_basic_num tmp = _find_gens_poly(exp);
    set_basic gens;
    for (auto it : tmp)
        gens.insert(pow(it.first, it.second));

    return P::from_container(gens, _basic_to_mpoly<P>(exp, gens));
}
} // namespace SymEngine

#endif
