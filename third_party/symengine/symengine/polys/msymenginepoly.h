#ifndef SYMENGINE_POLYNOMIALS_MULTIVARIATE
#define SYMENGINE_POLYNOMIALS_MULTIVARIATE

#include <symengine/expression.h>
#include <symengine/monomials.h>
#include <symengine/polys/uintpoly.h>
#include <symengine/polys/uexprpoly.h>
#include <symengine/symengine_casts.h>

namespace SymEngine
{

template <typename Vec, typename Value, typename Wrapper>
class UDictWrapper
{
public:
    using Dict = std::unordered_map<Vec, Value, vec_hash<Vec>>;
    Dict dict_;
    unsigned int vec_size;

    typedef Vec vec_type;
    typedef Value coef_type;
    typedef Dict dict_type;

    UDictWrapper(unsigned int s) SYMENGINE_NOEXCEPT
    {
        vec_size = s;
    }

    UDictWrapper() SYMENGINE_NOEXCEPT {}

    ~UDictWrapper() SYMENGINE_NOEXCEPT {}

    UDictWrapper(Dict &&p, unsigned int sz)
    {
        auto iter = p.begin();
        while (iter != p.end()) {
            if (iter->second == 0) {
                auto toErase = iter;
                iter++;
                p.erase(toErase);
            } else {
                iter++;
            }
        }

        dict_ = p;
        vec_size = sz;
    }

    UDictWrapper(const Dict &p, unsigned int sz)
    {
        for (auto &iter : p) {
            if (iter.second != Value(0))
                dict_[iter.first] = iter.second;
        }
        vec_size = sz;
    }

    Wrapper &operator=(Wrapper &&other)
    {
        if (this != &other)
            dict_ = std::move(other.dict_);
        return static_cast<Wrapper &>(*this);
    }

    friend Wrapper operator+(const Wrapper &a, const Wrapper &b)
    {
        SYMENGINE_ASSERT(a.vec_size == b.vec_size)
        Wrapper c = a;
        c += b;
        return c;
    }

    // both wrappers must have "aligned" vectors, ie same size
    // and vector positions refer to the same generators
    Wrapper &operator+=(const Wrapper &other)
    {
        SYMENGINE_ASSERT(vec_size == other.vec_size)

        for (auto &iter : other.dict_) {
            auto t = dict_.find(iter.first);
            if (t != dict_.end()) {
                t->second += iter.second;
                if (t->second == 0)
                    dict_.erase(t);
            } else {
                dict_.insert(t, {iter.first, iter.second});
            }
        }
        return static_cast<Wrapper &>(*this);
    }

    friend Wrapper operator-(const Wrapper &a, const Wrapper &b)
    {
        SYMENGINE_ASSERT(a.vec_size == b.vec_size)

        Wrapper c = a;
        c -= b;
        return c;
    }

    Wrapper operator-() const
    {
        auto c = *this;
        for (auto &iter : c.dict_)
            iter.second *= -1;
        return static_cast<Wrapper &>(c);
    }

    // both wrappers must have "aligned" vectors, ie same size
    // and vector positions refer to the same generators
    Wrapper &operator-=(const Wrapper &other)
    {
        SYMENGINE_ASSERT(vec_size == other.vec_size)

        for (auto &iter : other.dict_) {
            auto t = dict_.find(iter.first);
            if (t != dict_.end()) {
                t->second -= iter.second;
                if (t->second == 0)
                    dict_.erase(t);
            } else {
                dict_.insert(t, {iter.first, -iter.second});
            }
        }
        return static_cast<Wrapper &>(*this);
    }

    static Wrapper mul(const Wrapper &a, const Wrapper &b)
    {
        SYMENGINE_ASSERT(a.vec_size == b.vec_size)

        Wrapper p(a.vec_size);
        for (auto const &a_ : a.dict_) {
            for (auto const &b_ : b.dict_) {

                Vec target(a.vec_size, 0);
                for (unsigned int i = 0; i < a.vec_size; i++)
                    target[i] = a_.first[i] + b_.first[i];

                if (p.dict_.find(target) == p.dict_.end()) {
                    p.dict_.insert({target, a_.second * b_.second});
                } else {
                    p.dict_.find(target)->second += a_.second * b_.second;
                }
            }
        }

        for (auto it = p.dict_.begin(); it != p.dict_.end();) {
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
        Wrapper tmp = a, res(a.vec_size);

        Vec zero_v(a.vec_size, 0);
        res.dict_[zero_v] = 1_z;

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

    friend Wrapper operator*(const Wrapper &a, const Wrapper &b)
    {
        SYMENGINE_ASSERT(a.vec_size == b.vec_size)
        return Wrapper::mul(a, b);
    }

    Wrapper &operator*=(const Wrapper &other)
    {
        SYMENGINE_ASSERT(vec_size == other.vec_size)

        if (dict_.empty())
            return static_cast<Wrapper &>(*this);

        if (other.dict_.empty()) {
            dict_.clear();
            return static_cast<Wrapper &>(*this);
        }

        Vec zero_v(vec_size, 0);
        // ! other is a just constant term
        if (other.dict_.size() == 1
            and other.dict_.find(zero_v) != other.dict_.end()) {
            auto t = other.dict_.begin();
            for (auto &i1 : dict_)
                i1.second *= t->second;
            return static_cast<Wrapper &>(*this);
        }

        Wrapper res = Wrapper::mul(static_cast<Wrapper &>(*this), other);
        res.dict_.swap(this->dict_);
        return static_cast<Wrapper &>(*this);
    }

    bool operator==(const Wrapper &other) const
    {
        return dict_ == other.dict_;
    }

    bool operator!=(const Wrapper &other) const
    {
        return not(*this == other);
    }

    const Dict &get_dict() const
    {
        return dict_;
    }

    bool empty() const
    {
        return dict_.empty();
    }

    Value get_coeff(Vec &x) const
    {
        auto ite = dict_.find(x);
        if (ite != dict_.end())
            return ite->second;
        return Value(0);
    }

    Wrapper translate(const vec_uint &translator, unsigned int size) const
    {
        SYMENGINE_ASSERT(translator.size() == vec_size)
        SYMENGINE_ASSERT(size >= vec_size)

        Dict d;

        for (auto it : dict_) {
            Vec changed;
            changed.resize(size, 0);
            for (unsigned int i = 0; i < vec_size; i++)
                changed[translator[i]] = it.first[i];
            d.insert({changed, it.second});
        }

        return Wrapper(std::move(d), size);
    }
};

class MIntDict : public UDictWrapper<vec_uint, integer_class, MIntDict>
{
public:
    MIntDict(unsigned int s) SYMENGINE_NOEXCEPT : UDictWrapper(s) {}

    MIntDict() SYMENGINE_NOEXCEPT {}

    ~MIntDict() SYMENGINE_NOEXCEPT {}

    MIntDict(MIntDict &&other) SYMENGINE_NOEXCEPT
        : UDictWrapper(std::move(other))
    {
    }

    MIntDict(umap_uvec_mpz &&p, unsigned int sz)
        : UDictWrapper(std::move(p), sz)
    {
    }

    MIntDict(const umap_uvec_mpz &p, unsigned int sz) : UDictWrapper(p, sz) {}

    MIntDict(const MIntDict &) = default;

    MIntDict &operator=(const MIntDict &) = default;
};

class MExprDict : public UDictWrapper<vec_int, Expression, MExprDict>
{
public:
    MExprDict(unsigned int s) SYMENGINE_NOEXCEPT : UDictWrapper(s) {}

    MExprDict() SYMENGINE_NOEXCEPT {}

    ~MExprDict() SYMENGINE_NOEXCEPT {}

    MExprDict(MExprDict &&other) SYMENGINE_NOEXCEPT
        : UDictWrapper(std::move(other))
    {
    }

    MExprDict(umap_vec_expr &&p, unsigned int sz)
        : UDictWrapper(std::move(p), sz)
    {
    }

    MExprDict(const umap_vec_expr &p, unsigned int sz) : UDictWrapper(p, sz) {}

    MExprDict(const MExprDict &) = default;

    MExprDict &operator=(const MExprDict &) = default;
};

template <typename Container, typename Poly>
class MSymEnginePoly : public Basic
{
private:
    Container poly_;
    set_basic vars_;

public:
    typedef Container container_type;
    typedef typename Container::coef_type coef_type;

    MSymEnginePoly(const set_basic &vars, Container &&dict)
        : poly_{dict}, vars_{vars}
    {
    }

    static RCP<const Poly> from_container(const set_basic &vars, Container &&d)
    {
        return make_rcp<const Poly>(vars, std::move(d));
    }

    int compare(const Basic &o) const override
    {
        SYMENGINE_ASSERT(is_a<Poly>(o))

        const Poly &s = down_cast<const Poly &>(o);

        if (vars_.size() != s.vars_.size())
            return vars_.size() < s.vars_.size() ? -1 : 1;
        if (poly_.dict_.size() != s.poly_.dict_.size())
            return poly_.dict_.size() < s.poly_.dict_.size() ? -1 : 1;

        int cmp = unified_compare(vars_, s.vars_);
        if (cmp != 0)
            return cmp;

        return unified_compare(poly_.dict_, s.poly_.dict_);
    }

    template <typename FromPoly>
    static enable_if_t<is_a_UPoly<FromPoly>::value, RCP<const Poly>>
    from_poly(const FromPoly &p)
    {
        Container c;
        for (auto it = p.begin(); it != p.end(); ++it)
            c.dict_[{it->first}] = it->second;
        c.vec_size = 1;

        return Poly::from_container({p.get_var()}, std::move(c));
    }

    static RCP<const Poly> from_dict(const vec_basic &v,
                                     typename Container::dict_type &&d)
    {
        set_basic s;
        std::map<RCP<const Basic>, unsigned int, RCPBasicKeyLess> m;
        // Symbols in the vector are sorted by placeing them in an map image
        // of the symbols in the map is their original location in the vector

        for (unsigned int i = 0; i < v.size(); i++) {
            m.insert({v[i], i});
            s.insert(v[i]);
        }

        // vec_uint translator represents the permutation of the exponents
        vec_uint trans(s.size());
        auto mptr = m.begin();
        for (unsigned int i = 0; i < s.size(); i++) {
            trans[mptr->second] = i;
            mptr++;
        }

        Container x(std::move(d), numeric_cast<unsigned>(s.size()));
        return Poly::from_container(
            s, std::move(x.translate(trans, numeric_cast<unsigned>(s.size()))));
    }

    static Container container_from_dict(const set_basic &s,
                                         typename Container::dict_type &&d)
    {
        return Container(std::move(d), numeric_cast<unsigned>(s.size()));
    }

    inline vec_basic get_args() const override
    {
        return {};
    }

    inline const Container &get_poly() const
    {
        return poly_;
    }

    inline const set_basic &get_vars() const
    {
        return vars_;
    }

    bool __eq__(const Basic &o) const override
    {
        // TODO : fix for when vars are different, but there is an intersection
        if (not is_a<Poly>(o))
            return false;
        const Poly &o_ = down_cast<const Poly &>(o);
        // compare constants without regards to vars
        if (1 == poly_.dict_.size() && 1 == o_.poly_.dict_.size()) {
            if (poly_.dict_.begin()->second != o_.poly_.dict_.begin()->second)
                return false;
            if (poly_.dict_.begin()->first == o_.poly_.dict_.begin()->first
                && unified_eq(vars_, o_.vars_))
                return true;
            typename Container::vec_type v1, v2;
            v1.resize(vars_.size(), 0);
            v2.resize(o_.vars_.size(), 0);
            if (poly_.dict_.begin()->first == v1
                || o_.poly_.dict_.begin()->first == v2)
                return true;
            return false;
        } else if (0 == poly_.dict_.size() && 0 == o_.poly_.dict_.size()) {
            return true;
        } else {
            return (unified_eq(vars_, o_.vars_)
                    && unified_eq(poly_.dict_, o_.poly_.dict_));
        }
    }
};

class MIntPoly : public MSymEnginePoly<MIntDict, MIntPoly>
{
public:
    MIntPoly(const set_basic &vars, MIntDict &&dict)
        : MSymEnginePoly(vars, std::move(dict)){SYMENGINE_ASSIGN_TYPEID()}

          IMPLEMENT_TYPEID(SYMENGINE_MINTPOLY)

              hash_t __hash__() const override;
    RCP<const Basic> as_symbolic() const;

    integer_class eval(
        std::map<RCP<const Basic>, integer_class, RCPBasicKeyLess> &vals) const;
};

class MExprPoly : public MSymEnginePoly<MExprDict, MExprPoly>
{
public:
    MExprPoly(const set_basic &vars, MExprDict &&dict)
        : MSymEnginePoly(vars, std::move(dict)){SYMENGINE_ASSIGN_TYPEID()}

          IMPLEMENT_TYPEID(SYMENGINE_MEXPRPOLY)

              hash_t __hash__() const override;
    RCP<const Basic> as_symbolic() const;
    Expression
    eval(std::map<RCP<const Basic>, Expression, RCPBasicKeyLess> &vals) const;
};

// reconciles the positioning of the exponents in the vectors in the
// Dict dict_ of the arguments with the positioning of the exponents in
// the correspondng vectors of the output of the function. f1 and f2 are
// vectors whose indices are the positions in the arguments and whose values
// are the positions in the output.  set_basic s is the set of symbols of
// the output, and s1 and s2 are the sets of the symbols of the inputs.
unsigned int reconcile(vec_uint &v1, vec_uint &v2, set_basic &s,
                       const set_basic &s1, const set_basic &s2);

template <typename Poly, typename Container>
set_basic get_translated_container(Container &x, Container &y, const Poly &a,
                                   const Poly &b)
{
    vec_uint v1, v2;
    set_basic s;

    unsigned int sz = reconcile(v1, v2, s, a.get_vars(), b.get_vars());
    x = a.get_poly().translate(v1, sz);
    y = b.get_poly().translate(v2, sz);

    return s;
}

template <typename Poly>
RCP<const Poly> add_mpoly(const Poly &a, const Poly &b)
{
    typename Poly::container_type x, y;
    set_basic s = get_translated_container(x, y, a, b);
    x += y;
    return Poly::from_container(s, std::move(x));
}

template <typename Poly>
RCP<const Poly> sub_mpoly(const Poly &a, const Poly &b)
{
    typename Poly::container_type x, y;
    set_basic s = get_translated_container(x, y, a, b);
    x -= y;
    return Poly::from_container(s, std::move(x));
}

template <typename Poly>
RCP<const Poly> mul_mpoly(const Poly &a, const Poly &b)
{
    typename Poly::container_type x, y;
    set_basic s = get_translated_container(x, y, a, b);
    x *= y;
    return Poly::from_container(s, std::move(x));
}

template <typename Poly>
RCP<const Poly> neg_mpoly(const Poly &a)
{
    auto x = a.get_poly();
    return Poly::from_container(a.get_vars(), std::move(-x));
}

template <typename Poly>
RCP<const Poly> pow_mpoly(const Poly &a, unsigned int n)
{
    auto x = a.get_poly();
    return Poly::from_container(a.get_vars(), Poly::container_type::pow(x, n));
}
} // namespace SymEngine

#endif
