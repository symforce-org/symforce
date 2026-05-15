/**
 *  \file uintpoly_piranha.h
 *  Class for Polynomial: UIntPolyPiranha
 **/
#ifndef SYMENGINE_UINTPOLY_PIRANHA_H
#define SYMENGINE_UINTPOLY_PIRANHA_H

#include <symengine/polys/upolybase.h>
#include <symengine/expression.h>
#include <symengine/dict.h>
#include <memory>

#ifdef HAVE_SYMENGINE_PIRANHA
#include <piranha/monomial.hpp>
#include <piranha/polynomial.hpp>
#include <piranha/mp_rational.hpp>
#include <piranha/mp_integer.hpp>
#include <piranha/math.hpp>
#include <piranha/type_traits.hpp>

#if SYMENGINE_INTEGER_CLASS != SYMENGINE_PIRANHA
namespace piranha
{

// overloading pow for pirahna::math::evaluate
namespace math
{
using namespace SymEngine;
template <typename U, typename V>
struct pow_impl<V, U,
                enable_if_t<std::is_integral<U>::value
                            and (std::is_same<V, integer_class>::value
                                 or std::is_same<V, rational_class>::value)>> {
    template <typename T2>
    V operator()(const V &r, const T2 &x) const
    {
        V res;
        mp_pow_ui(res, r, x);
        return res;
    }
};

template <>
struct gcd_impl<SymEngine::integer_class, SymEngine::integer_class> {
    SymEngine::integer_class operator()(const SymEngine::integer_class &r,
                                        const SymEngine::integer_class &x) const
    {
        SymEngine::integer_class res;
        mp_gcd(res, r, x);
        return res;
    }
};

template <>
struct divexact_impl<SymEngine::integer_class> {
    void operator()(SymEngine::integer_class &r,
                    const SymEngine::integer_class &x,
                    const SymEngine::integer_class &y) const
    {
        SymEngine::integer_class rem;
        mp_tdiv_qr(r, rem, x, y);
        if (rem != SymEngine::integer_class(0)) {
            piranha_throw(inexact_division);
        }
    }
};
template <>
struct divexact_impl<SymEngine::rational_class> {
    void operator()(SymEngine::rational_class &r,
                    const SymEngine::rational_class &x,
                    const SymEngine::rational_class &y) const
    {
        r = x / y;
    }
};
} // namespace math

template <>
struct has_exact_ring_operations<SymEngine::integer_class> {
    static const bool value = true;
};
template <>
struct has_exact_ring_operations<SymEngine::rational_class> {
    static const bool value = true;
};
} // namespace piranha
#endif

// need definition for piranha::rational too
namespace piranha
{
namespace math
{
template <>
struct gcd_impl<SymEngine::rational_class, SymEngine::rational_class> {
    SymEngine::rational_class
    operator()(const SymEngine::rational_class &r,
               const SymEngine::rational_class &x) const
    {
        return SymEngine::rational_class(1);
    }
};
} // namespace math
} // namespace piranha

namespace SymEngine
{
using pmonomial = piranha::monomial<unsigned int>;
using pintpoly = piranha::polynomial<integer_class, pmonomial>;
using pratpoly = piranha::polynomial<rational_class, pmonomial>;

template <typename Cf, typename Container>
class PiranhaForIter
{
public:
    typedef decltype(std::declval<Container &>()._container().begin()) ptr_type;

private:
    ptr_type ptr_;

public:
    PiranhaForIter(ptr_type ptr) : ptr_{ptr} {}

    bool operator==(const PiranhaForIter &rhs)
    {
        return (ptr_ == rhs.ptr_);
    }

    bool operator!=(const PiranhaForIter &rhs)
    {
        return (ptr_ != rhs.ptr_);
    }

    PiranhaForIter operator++()
    {
        ptr_++;
        return *this;
    }

    std::pair<unsigned int, const Cf &> operator*()
    {
        return std::make_pair(*(ptr_->m_key.begin()), ptr_->m_cf);
    }

    std::shared_ptr<std::pair<unsigned int, const Cf &>> operator->()
    {
        return std::make_shared<std::pair<unsigned int, const Cf &>>(
            *(ptr_->m_key.begin()), ptr_->m_cf);
    }
};

template <typename Container, template <typename X, typename Y> class BaseType,
          typename Poly>
class UPiranhaPoly : public BaseType<Container, Poly>
{
public:
    using Cf = typename BaseType<Container, Poly>::coef_type;
    using term = typename Container::term_type;

    UPiranhaPoly(const RCP<const Basic> &var, Container &&dict)
        : BaseType<Container, Poly>(var, std::move(dict))
    {
    }

    int compare(const Basic &o) const
    {
        SYMENGINE_ASSERT(is_a<Poly>(o))
        const Poly &s = down_cast<const Poly &>(o);
        int cmp = this->get_var()->compare(*s.get_var());
        if (cmp != 0)
            return cmp;
        if (this->get_poly() == s.get_poly())
            return 0;
        return (this->get_poly().hash() < s.get_poly().hash()) ? -1 : 1;
    }

    static Container container_from_dict(const RCP<const Basic> &var,
                                         std::map<unsigned, Cf> &&d)
    {
        Container p;
        piranha::symbol_set ss({{piranha::symbol(detail::poly_print(var))}});
        p.set_symbol_set(ss);
        for (auto &it : d)
            if (it.second != 0)
                p.insert(term(it.second, pmonomial{it.first}));

        return p;
    }

    static RCP<const Poly> from_vec(const RCP<const Basic> &var,
                                    const std::vector<Cf> &v)
    {
        Container p;
        piranha::symbol_set ss({{piranha::symbol(detail::poly_print(var))}});
        p.set_symbol_set(ss);
        for (unsigned int i = 0; i < v.size(); i++) {
            if (v[i] != 0) {
                p.insert(term(v[i], pmonomial{i}));
            }
        }
        return make_rcp<const Poly>(var, std::move(p));
    }

    template <typename FromPoly>
    static enable_if_t<is_a_UPoly<FromPoly>::value, RCP<const Poly>>
    from_poly(const FromPoly &f)
    {
        Container p;
        piranha::symbol_set ss(
            {{piranha::symbol(detail::poly_print(f.get_var()))}});
        p.set_symbol_set(ss);
        for (auto it = f.begin(); it != f.end(); ++it)
            p.insert(term(it->second, pmonomial{it->first}));
        return Poly::from_container(f.get_var(), std::move(p));
    }

    Cf eval(const Cf &x) const
    {
        const std::unordered_map<std::string, Cf> t
            = {{detail::poly_print(this->get_var()), x}};
        return piranha::math::evaluate<Cf, Container>(this->get_poly(), t);
    }

    Cf get_coeff(unsigned int x) const
    {
        return this->get_poly().find_cf(pmonomial{x});
    }

    const Cf &get_coeff_ref(unsigned int x) const
    {
        static Cf pzero(0);

        term temp = term(0, pmonomial{x});
        auto it = this->get_poly()._container().find(temp);
        if (it == this->get_poly()._container().end())
            return pzero;
        return it->m_cf;
    }

    int size() const
    {
        if (this->get_poly().size() == 0)
            return 0;
        return this->get_degree() + 1;
    }

    // begin() and end() are unordered
    // obegin() and oend() are ordered, from highest degree to lowest
    typedef PiranhaForIter<Cf, Container> iterator;
    typedef ContainerRevIter<Poly, const Cf &> r_iterator;
    iterator begin() const
    {
        return iterator(this->get_poly()._container().begin());
    }
    iterator end() const
    {
        return iterator(this->get_poly()._container().end());
    }
    r_iterator obegin() const
    {
        return r_iterator(this->template rcp_from_this_cast<Poly>(),
                          (long)size() - 1);
    }
    r_iterator oend() const
    {
        return r_iterator(this->template rcp_from_this_cast<Poly>(), -1);
    }
};

class UIntPolyPiranha
    : public UPiranhaPoly<pintpoly, UIntPolyBase, UIntPolyPiranha>
{
public:
    IMPLEMENT_TYPEID(SYMENGINE_UINTPOLYPIRANHA)
    //! Constructor of UIntPolyPiranha class
    UIntPolyPiranha(const RCP<const Basic> &var, pintpoly &&dict);
    //! \return size of the hash
    hash_t __hash__() const;

}; // UIntPolyPiranha

class URatPolyPiranha
    : public UPiranhaPoly<pratpoly, URatPolyBase, URatPolyPiranha>
{
public:
    IMPLEMENT_TYPEID(SYMENGINE_URATPOLYPIRANHA)
    //! Constructor of UIntPolyPiranha class
    URatPolyPiranha(const RCP<const Basic> &var, pratpoly &&dict);
    //! \return size of the hash
    hash_t __hash__() const;
};

inline RCP<const UIntPolyPiranha> gcd_upoly(const UIntPolyPiranha &a,
                                            const UIntPolyPiranha &b)
{
    if (!(a.get_var()->__eq__(*b.get_var())))
        throw SymEngineException("Error: variables must agree.");

    pintpoly gcdx(std::get<0>(pintpoly::gcd(a.get_poly(), b.get_poly())));
    // following the convention, that leading coefficient should be positive
    if (gcdx.find_cf(pmonomial{gcdx.degree()}) < 0)
        piranha::math::negate(gcdx);
    return make_rcp<const UIntPolyPiranha>(a.get_var(), std::move(gcdx));
}

inline RCP<const URatPolyPiranha> gcd_upoly(const URatPolyPiranha &a,
                                            const URatPolyPiranha &b)
{
    if (!(a.get_var()->__eq__(*b.get_var())))
        throw SymEngineException("Error: variables must agree.");

    pratpoly gcdx(std::get<0>(pratpoly::gcd(a.get_poly(), b.get_poly())));
    // following the convention, that polynomial should be monic
    gcdx *= (1 / gcdx.find_cf(pmonomial{gcdx.degree()}));
    return make_rcp<const URatPolyPiranha>(a.get_var(), std::move(gcdx));
}

inline RCP<const UIntPolyPiranha> lcm_upoly(const UIntPolyPiranha &a,
                                            const UIntPolyPiranha &b)
{
    if (!(a.get_var()->__eq__(*b.get_var())))
        throw SymEngineException("Error: variables must agree.");

    pintpoly lcmx(std::get<0>(pintpoly::gcd(a.get_poly(), b.get_poly())));
    lcmx = (a.get_poly() * b.get_poly()) / lcmx;
    if (lcmx.find_cf(pmonomial{lcmx.degree()}) < 0)
        piranha::math::negate(lcmx);
    return make_rcp<const UIntPolyPiranha>(a.get_var(), std::move(lcmx));
}

inline RCP<const URatPolyPiranha> lcm_upoly(const URatPolyPiranha &a,
                                            const URatPolyPiranha &b)
{
    if (!(a.get_var()->__eq__(*b.get_var())))
        throw SymEngineException("Error: variables must agree.");

    pratpoly lcmx(std::get<0>(pratpoly::gcd(a.get_poly(), b.get_poly())));
    lcmx = (a.get_poly() * b.get_poly()) / lcmx;
    lcmx *= (1 / lcmx.find_cf(pmonomial{lcmx.degree()}));
    return make_rcp<const URatPolyPiranha>(a.get_var(), std::move(lcmx));
}

template <typename Container, template <typename X, typename Y> class BaseType,
          typename Poly>
RCP<const Poly> pow_upoly(const UPiranhaPoly<Container, BaseType, Poly> &a,
                          unsigned int p)
{
    return make_rcp<const Poly>(a.get_var(),
                                std::move(piranha::math::pow(a.get_poly(), p)));
}

template <typename Container, template <typename X, typename Y> class BaseType,
          typename Poly>
bool divides_upoly(const UPiranhaPoly<Container, BaseType, Poly> &a,
                   const Poly &b, const Ptr<RCP<const Poly>> &res)
{
    if (!(a.get_var()->__eq__(*b.get_var())))
        throw SymEngineException("Error: variables must agree.");

    try {
        Container z;
        piranha::math::divexact(z, b.get_poly(), a.get_poly());
        *res = Poly::from_container(a.get_var(), std::move(z));
        return true;
    } catch (const piranha::math::inexact_division &) {
        return false;
    }
}
} // namespace SymEngine

#endif // HAVE_SYMENGINE_PIRANHA

#endif // SYMENGINE_UINTPOLY_PIRANHA_H
