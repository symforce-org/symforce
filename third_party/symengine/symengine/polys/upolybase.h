/**
 *  \file polynomial_int.h
 *  Class for Univariate Polynomial Base
 **/
#ifndef SYMENGINE_UINT_BASE_H
#define SYMENGINE_UINT_BASE_H

#include <symengine/basic.h>
#include <symengine/pow.h>
#include <symengine/add.h>
#include <symengine/rational.h>
#include <symengine/expression.h>
#include <memory>

#ifdef HAVE_SYMENGINE_FLINT
#include <symengine/flint_wrapper.h>
using fz_t = SymEngine::fmpz_wrapper;
using fq_t = SymEngine::fmpq_wrapper;
#endif
#ifdef HAVE_SYMENGINE_PIRANHA
#include <piranha/mp_integer.hpp>
#include <piranha/mp_rational.hpp>
#endif

namespace SymEngine
{
// misc methods

#if SYMENGINE_INTEGER_CLASS == SYMENGINE_GMPXX                                 \
    || SYMENGINE_INTEGER_CLASS == SYMENGINE_GMP
#ifdef HAVE_SYMENGINE_FLINT
inline integer_class to_mp_class(const fz_t &i)
{
    integer_class x;
    fmpz_get_mpz(x.get_mpz_t(), i.get_fmpz_t());
    return x;
}
inline rational_class to_mp_class(const fq_t &i)
{
    rational_class x;
    fmpq_get_mpq(x.get_mpq_t(), i.get_fmpq_t());
    return x;
}
#endif

#ifdef HAVE_SYMENGINE_PIRANHA
inline integer_class to_mp_class(const piranha::integer &i)
{
    integer_class x;
    mpz_set(x.get_mpz_t(), i.get_mpz_view());
    return x;
}
inline rational_class to_mp_class(const piranha::rational &i)
{
    rational_class x;
    mpq_set(x.get_mpq_t(), i.get_mpq_view());
    return x;
}
#endif

#elif SYMENGINE_INTEGER_CLASS == SYMENGINE_PIRANHA
#ifdef HAVE_SYMENGINE_FLINT
inline integer_class to_mp_class(const fz_t &i)
{
    integer_class x;
    fmpz_get_mpz(get_mpz_t(x), i.get_fmpz_t());
    return x;
}
inline rational_class to_mp_class(const fq_t &i)
{
    rational_class s;
    fmpz_get_mpz(get_mpz_t(s._num()), i.get_num().get_fmpz_t());
    fmpz_get_mpz(get_mpz_t(s._den()), i.get_den().get_fmpz_t());
    return s;
}
#endif

#elif SYMENGINE_INTEGER_CLASS == SYMENGINE_FLINT
#ifdef HAVE_SYMENGINE_PIRANHA
inline integer_class to_mp_class(const piranha::integer &x)
{
    return integer_class(x.get_mpz_view());
}
inline rational_class to_mp_class(const piranha::rational &x)
{
    return rational_class(x.get_mpq_view());
}
#endif

#endif

inline integer_class to_mp_class(const integer_class &i)
{
    return i;
}

inline rational_class to_mp_class(const rational_class &i)
{
    return i;
}

// dict wrapper
template <typename Key, typename Value, typename Wrapper>
class ODictWrapper
{
public:
    std::map<Key, Value> dict_;
    typedef Key key_type;

public:
    ODictWrapper() SYMENGINE_NOEXCEPT {}
    ~ODictWrapper() SYMENGINE_NOEXCEPT {}

    ODictWrapper(const int &i)
    {
        if (i != 0)
            dict_ = {{0, Value(i)}};
    }

    ODictWrapper(const std::map<Key, Value> &p)
    {
        for (auto &iter : p) {
            if (iter.second != Value(0))
                dict_[iter.first] = iter.second;
        }
    }

    ODictWrapper(std::map<Key, Value> &&p)
    {
        for (auto &iter : p) {
            if (iter.second != Value(0)) {
                auto erase = iter;
                iter++;
                p.erase(erase);
            } else {
                iter++;
            }
        }
        dict_ = p;
    }

    ODictWrapper(const Value &p)
    {
        if (p != Value(0))
            dict_[0] = p;
    }

    ODictWrapper(std::string s)
    {
        dict_[1] = Value(1);
    }

    static Wrapper from_vec(const std::vector<Value> &v)
    {
        Wrapper x;
        x.dict_ = {};
        for (unsigned int i = 0; i < v.size(); i++) {
            if (v[i] != Value(0)) {
                x.dict_[i] = v[i];
            }
        }
        return x;
    }

    Wrapper &operator=(Wrapper &&other) SYMENGINE_NOEXCEPT
    {
        if (this != &other)
            dict_ = std::move(other.dict_);
        return static_cast<Wrapper &>(*this);
    }

    friend Wrapper operator+(const Wrapper &a, const Wrapper &b)
    {
        Wrapper c = a;
        c += b;
        return c;
    }

    Wrapper &operator+=(const Wrapper &other)
    {
        for (auto &iter : other.dict_) {
            auto t = dict_.lower_bound(iter.first);
            if (t != dict_.end() and t->first == iter.first) {
                t->second += iter.second;
                if (t->second == 0) {
                    dict_.erase(t);
                }
            } else {
                dict_.insert(t, {iter.first, iter.second});
            }
        }
        return static_cast<Wrapper &>(*this);
    }

    friend Wrapper operator-(const Wrapper &a, const Wrapper &b)
    {
        Wrapper c = a;
        c -= b;
        return c;
    }

    Wrapper operator-() const
    {
        ODictWrapper c = *this;
        for (auto &iter : c.dict_)
            iter.second *= -1;
        return static_cast<Wrapper &>(c);
    }

    Wrapper &operator-=(const Wrapper &other)
    {
        for (auto &iter : other.dict_) {
            auto t = dict_.lower_bound(iter.first);
            if (t != dict_.end() and t->first == iter.first) {
                t->second -= iter.second;
                if (t->second == 0) {
                    dict_.erase(t);
                }
            } else {
                dict_.insert(t, {iter.first, -iter.second});
            }
        }
        return static_cast<Wrapper &>(*this);
    }

    static Wrapper mul(const Wrapper &a, const Wrapper &b)
    {
        if (a.get_dict().empty())
            return a;
        if (b.get_dict().empty())
            return b;

        Wrapper p;
        for (const auto &i1 : a.dict_)
            for (const auto &i2 : b.dict_)
                p.dict_[i1.first + i2.first] += i1.second * i2.second;

        for (auto it = p.dict_.cbegin(); it != p.dict_.cend();) {
            if (it->second == 0) {
                p.dict_.erase(it++);
            } else {
                ++it;
            }
        }
        return p;
    }

    static Wrapper pow(const Wrapper &a, unsigned int p)
    {
        Wrapper tmp = a, res(1);

        while (p != 1) {
            if (p % 2 == 0) {
                tmp = tmp * tmp;
            } else {
                res = res * tmp;
                tmp = tmp * tmp;
            }
            p >>= 1;
        }

        return (res * tmp);
    }

    template <typename FromPoly>
    static Wrapper from_poly(const FromPoly &p)
    {
        Wrapper t;
        for (auto it = p.begin(); it != p.end(); ++it)
            t.dict_[it->first] = it->second;
        return t;
    }

    friend Wrapper operator*(const Wrapper &a, const Wrapper &b)
    {
        return Wrapper::mul(a, b);
    }

    Wrapper &operator*=(const Wrapper &other)
    {
        if (dict_.empty())
            return static_cast<Wrapper &>(*this);

        if (other.dict_.empty()) {
            dict_.clear();
            return static_cast<Wrapper &>(*this);
        }

        // ! other is a just constant term
        if (other.dict_.size() == 1
            and other.dict_.find(0) != other.dict_.end()) {
            auto t = other.dict_.begin();
            for (auto &i1 : dict_)
                i1.second *= t->second;
            return static_cast<Wrapper &>(*this);
        }

        Wrapper res = Wrapper::mul(static_cast<Wrapper &>(*this), other);
        res.dict_.swap(this->dict_);
        return static_cast<Wrapper &>(*this);
    }

    friend bool operator==(const Wrapper &a, const Wrapper &b)
    {
        return a.dict_ == b.dict_;
    }

    bool operator!=(const Wrapper &other) const
    {
        return not(*this == other);
    }

    const std::map<Key, Value> &get_dict() const
    {
        return dict_;
    }

    size_t size() const
    {
        return dict_.size();
    }

    bool empty() const
    {
        return dict_.empty();
    }

    Key degree() const
    {
        if (dict_.empty())
            return Key(0);
        return dict_.rbegin()->first;
    }

    Value get_coeff(Key x) const
    {
        auto ite = dict_.find(x);
        if (ite != dict_.end())
            return ite->second;
        return Value(0);
    }

    Value get_lc() const
    {
        if (dict_.empty())
            return Value(0);
        return dict_.rbegin()->second;
    }
};

umap_basic_num _find_gens_poly(const RCP<const Basic> &x);

template <typename Container, typename Poly>
class UPolyBase : public Basic
{
private:
    RCP<const Basic> var_;
    Container poly_;

public:
    UPolyBase(const RCP<const Basic> &var, Container &&container)
        : var_{var}, poly_{container}
    {
    }

    typedef Container container_type;

    //! \returns `-1`,`0` or `1` after comparing
    int compare(const Basic &o) const override = 0;
    hash_t __hash__() const override = 0;

    // return `degree` + 1. `0` returned for zero poly.
    virtual int size() const = 0;

    //! \returns `true` if two objects are equal
    inline bool __eq__(const Basic &o) const override
    {
        if (is_a<Poly>(o))
            return eq(*var_, *(down_cast<const Poly &>(o).var_))
                   and poly_ == down_cast<const Poly &>(o).poly_;
        return false;
    }

    inline const RCP<const Basic> &get_var() const
    {
        return var_;
    }

    inline const Container &get_poly() const
    {
        return poly_;
    }

    inline vec_basic get_args() const override
    {
        return {};
    }

    static RCP<const Poly> from_container(const RCP<const Basic> &var,
                                          Container &&d)
    {
        return make_rcp<const Poly>(var, std::move(d));
    }
};

template <typename Cont, typename Poly>
class UExprPolyBase : public UPolyBase<Cont, Poly>
{
public:
    typedef Expression coef_type;

    UExprPolyBase(const RCP<const Basic> &var, Cont &&container)
        : UPolyBase<Cont, Poly>(var, std::move(container))
    {
    }

    inline int get_degree() const
    {
        return this->get_poly().degree();
    }

    static RCP<const Poly> from_dict(const RCP<const Basic> &var,
                                     std::map<int, Expression> &&d)
    {
        return Poly::from_container(
            var, Poly::container_from_dict(var, std::move(d)));
    }

    RCP<const Basic> as_symbolic() const
    {
        auto it = (down_cast<const Poly &>(*this)).begin();
        auto end = (down_cast<const Poly &>(*this)).end();

        vec_basic args;
        for (; it != end; ++it) {
            if (it->first == 0)
                args.push_back(it->second.get_basic());
            else if (it->first == 1) {
                if (it->second == Expression(1))
                    args.push_back(this->get_var());
                else
                    args.push_back(
                        mul(it->second.get_basic(), this->get_var()));
            } else if (it->second == 1)
                args.push_back(pow(this->get_var(), integer(it->first)));
            else
                args.push_back(mul(it->second.get_basic(),
                                   pow(this->get_var(), integer(it->first))));
        }
        if (this->get_poly().empty())
            args.push_back(zero);
        return SymEngine::add(args);
    }
};
// super class for all non-expr polys, all methods which are
// common for all non-expr polys go here eg. degree, eval etc.
template <typename Container, typename Poly, typename Cf>
class UNonExprPoly : public UPolyBase<Container, Poly>
{
public:
    typedef Cf coef_type;

    UNonExprPoly(const RCP<const Basic> &var, Container &&container)
        : UPolyBase<Container, Poly>(var, std::move(container))
    {
    }

    // return coefficient of degree 'i'
    virtual Cf get_coeff(unsigned int i) const = 0;
    // return value of poly when ealudated at `x`
    virtual Cf eval(const Cf &x) const = 0;

    std::vector<Cf> multieval(const std::vector<Cf> &v) const
    {
        // this is not the optimal algorithm
        std::vector<Cf> res(v.size());
        for (unsigned int i = 0; i < v.size(); ++i)
            res[i] = eval(v[i]);
        return res;
    }

    inline int get_degree() const
    {
        return numeric_cast<int>(this->get_poly().degree());
    }

    Cf get_lc() const
    {
        return get_coeff(get_degree());
    }

    static RCP<const Poly> from_dict(const RCP<const Basic> &var,
                                     std::map<unsigned, Cf> &&d)
    {
        return Poly::from_container(
            var, Poly::container_from_dict(var, std::move(d)));
    }
};

template <typename Container, typename Poly>
class UIntPolyBase : public UNonExprPoly<Container, Poly, integer_class>
{
public:
    UIntPolyBase(const RCP<const Basic> &var, Container &&container)
        : UNonExprPoly<Container, Poly, integer_class>(var,
                                                       std::move(container))
    {
    }

    RCP<const Basic> as_symbolic() const
    {
        auto it = (down_cast<const Poly &>(*this)).begin();
        auto end = (down_cast<const Poly &>(*this)).end();

        vec_basic args;
        for (; it != end; ++it) {
            integer_class m = it->second;

            if (it->first == 0) {
                args.push_back(integer(m));
            } else if (it->first == 1) {
                if (m == 1) {
                    args.push_back(this->get_var());
                } else {
                    args.push_back(
                        Mul::from_dict(integer(m), {{this->get_var(), one}}));
                }
            } else {
                if (m == 1) {
                    args.push_back(pow(this->get_var(), integer(it->first)));
                } else {
                    args.push_back(Mul::from_dict(
                        integer(m), {{this->get_var(), integer(it->first)}}));
                }
            }
        }
        return SymEngine::add(args);
    }
};

template <typename Container, typename Poly>
class URatPolyBase : public UNonExprPoly<Container, Poly, rational_class>
{
public:
    URatPolyBase(const RCP<const Basic> &var, Container &&container)
        : UNonExprPoly<Container, Poly, rational_class>(var,
                                                        std::move(container))
    {
    }

    RCP<const Basic> as_symbolic() const
    {
        auto it = (down_cast<const Poly &>(*this)).begin();
        auto end = (down_cast<const Poly &>(*this)).end();

        vec_basic args;
        for (; it != end; ++it) {
            rational_class m = it->second;

            if (it->first == 0) {
                args.push_back(Rational::from_mpq(m));
            } else if (it->first == 1) {
                if (m == 1) {
                    args.push_back(this->get_var());
                } else {
                    args.push_back(Mul::from_dict(Rational::from_mpq(m),
                                                  {{this->get_var(), one}}));
                }
            } else {
                if (m == 1) {
                    args.push_back(pow(this->get_var(), integer(it->first)));
                } else {
                    args.push_back(Mul::from_dict(
                        Rational::from_mpq(m),
                        {{this->get_var(), integer(it->first)}}));
                }
            }
        }
        return SymEngine::add(args);
    }
};

template <typename T, typename Int>
class ContainerBaseIter
{
protected:
    RCP<const T> ptr_;
    long i_;

public:
    ContainerBaseIter(RCP<const T> ptr, long x) : ptr_{ptr}, i_{x} {}

    friend bool operator==(const ContainerBaseIter &lhs,
                           const ContainerBaseIter &rhs)
    {
        return (lhs.ptr_ == rhs.ptr_) and (lhs.i_ == rhs.i_);
    }

    bool operator!=(const ContainerBaseIter &rhs)
    {
        return not(*this == rhs);
    }

    std::pair<long, Int> operator*()
    {
        return std::make_pair(i_, ptr_->get_coeff_ref(i_));
    }

    std::shared_ptr<std::pair<unsigned, Int>> operator->()
    {
        return std::make_shared<std::pair<unsigned, Int>>(
            numeric_cast<unsigned>(i_),
            ptr_->get_coeff_ref(numeric_cast<unsigned>(i_)));
    }
};

template <typename T, typename Int>
class ContainerForIter : public ContainerBaseIter<T, Int>
{
public:
    ContainerForIter(RCP<const T> ptr, long x)
        : ContainerBaseIter<T, Int>(ptr, x)
    {
        if (this->ptr_->get_coeff_ref(numeric_cast<unsigned>(this->i_)) == 0
            and this->i_ < this->ptr_->size()) {
            ++(*this);
        }
    }

    ContainerForIter operator++()
    {
        this->i_++;
        while (this->i_ < this->ptr_->size()) {
            if (this->ptr_->get_coeff_ref(numeric_cast<unsigned>(this->i_))
                != 0)
                break;
            this->i_++;
        }
        return *this;
    }
};

template <typename T, typename Int>
class ContainerRevIter : public ContainerBaseIter<T, Int>
{
public:
    ContainerRevIter(RCP<const T> ptr, long x)
        : ContainerBaseIter<T, Int>(ptr, x)
    {
    }

    ContainerRevIter operator++()
    {
        this->i_--;
        while (this->i_ >= 0) {
            if (this->ptr_->get_coeff_ref(numeric_cast<unsigned>(this->i_))
                != 0)
                break;
            this->i_--;
        }
        return *this;
    }
};

template <typename P>
struct is_a_UPoly {
    static const bool value
        = std::is_base_of<UPolyBase<typename P::container_type, P>, P>::value;
};

template <typename Poly>
RCP<const Poly> add_upoly(const Poly &a, const Poly &b)
{
    if (!(a.get_var()->__eq__(*b.get_var())))
        throw SymEngineException("Error: variables must agree.");

    auto dict = a.get_poly();
    dict += b.get_poly();
    return Poly::from_container(a.get_var(), std::move(dict));
}

template <typename Poly>
RCP<const Poly> neg_upoly(const Poly &a)
{
    auto dict = a.get_poly();
    dict = -dict;
    return Poly::from_container(a.get_var(), std::move(dict));
}

template <typename Poly>
RCP<const Poly> sub_upoly(const Poly &a, const Poly &b)
{
    if (!(a.get_var()->__eq__(*b.get_var())))
        throw SymEngineException("Error: variables must agree.");

    auto dict = a.get_poly();
    dict -= b.get_poly();
    return Poly::from_container(a.get_var(), std::move(dict));
}

template <typename Poly>
RCP<const Poly> mul_upoly(const Poly &a, const Poly &b)
{
    if (!(a.get_var()->__eq__(*b.get_var())))
        throw SymEngineException("Error: variables must agree.");

    auto dict = a.get_poly();
    dict *= b.get_poly();
    return Poly::from_container(a.get_var(), std::move(dict));
}

template <typename Poly>
RCP<const Poly> quo_upoly(const Poly &a, const Poly &b)
{
    if (!(a.get_var()->__eq__(*b.get_var())))
        throw SymEngineException("Error: variables must agree.");

    auto dict = a.get_poly();
    dict /= b.get_poly();
    return Poly::from_dict(a.get_var(), std::move(dict));
}
} // namespace SymEngine

#endif // SYMENGINE_UINT_BASE_H
