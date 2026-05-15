/**
 *  \file fields.h
 *
 **/
#ifndef SYMENGINE_FIELDS_H
#define SYMENGINE_FIELDS_H

#include <symengine/basic.h>
#include <symengine/dict.h>
#include <symengine/polys/upolybase.h>
#include <symengine/polys/uintpoly.h>

namespace SymEngine
{
class GaloisFieldDict
{
public:
    std::vector<integer_class> dict_;
    integer_class modulo_;

public:
    struct DictLess {
        bool operator()(const GaloisFieldDict &a,
                        const GaloisFieldDict &b) const
        {
            if (a.degree() == b.degree())
                return a.dict_ < b.dict_;
            else
                return a.degree() < b.degree();
        }
        bool operator()(const std::pair<GaloisFieldDict, unsigned> &a,
                        const std::pair<GaloisFieldDict, unsigned> &b) const
        {
            if (a.first.degree() == b.first.degree())
                return a.first.dict_ < b.first.dict_;
            else
                return a.first.degree() < b.first.degree();
        }
    };
    GaloisFieldDict() SYMENGINE_NOEXCEPT {}
    ~GaloisFieldDict() SYMENGINE_NOEXCEPT {}
    GaloisFieldDict(GaloisFieldDict &&other) SYMENGINE_NOEXCEPT
        : dict_(std::move(other.dict_)),
          modulo_(std::move(other.modulo_))
    {
    }
    GaloisFieldDict(const int &i, const integer_class &mod);
    GaloisFieldDict(const map_uint_mpz &p, const integer_class &mod);
    GaloisFieldDict(const integer_class &i, const integer_class &mod);

    static GaloisFieldDict from_vec(const std::vector<integer_class> &v,
                                    const integer_class &modulo);

    GaloisFieldDict(const GaloisFieldDict &) = default;
    GaloisFieldDict &operator=(const GaloisFieldDict &) = default;
    void gf_div(const GaloisFieldDict &o, const Ptr<GaloisFieldDict> &quo,
                const Ptr<GaloisFieldDict> &rem) const;

    GaloisFieldDict gf_lshift(const integer_class n) const;
    void gf_rshift(const integer_class n, const Ptr<GaloisFieldDict> &quo,
                   const Ptr<GaloisFieldDict> &rem) const;
    GaloisFieldDict gf_sqr() const;
    GaloisFieldDict gf_pow(const unsigned long n) const;
    void gf_monic(integer_class &res, const Ptr<GaloisFieldDict> &monic) const;
    GaloisFieldDict gf_gcd(const GaloisFieldDict &o) const;
    GaloisFieldDict gf_lcm(const GaloisFieldDict &o) const;
    GaloisFieldDict gf_diff() const;
    integer_class gf_eval(const integer_class &a) const;
    vec_integer_class gf_multi_eval(const vec_integer_class &v) const;

    // Returns whether polynomial is squarefield in `modulo_`
    bool gf_is_sqf() const;
    // Returns the square free decomposition of polynomial's monic
    // representation in `modulo_`
    // A vector of pair is returned where each element is a factor and each
    // pair's first raised to power of second gives the factor.
    std::vector<std::pair<GaloisFieldDict, unsigned>> gf_sqf_list() const;

    // Returns the square free part of the polynomaial in `modulo_`
    GaloisFieldDict gf_sqf_part() const;
    // composition of polynomial g(h) mod (*this)
    GaloisFieldDict gf_compose_mod(const GaloisFieldDict &g,
                                   const GaloisFieldDict &h) const;
    // returns `x**(i * modullo_) % (*this)` for `i` in [0, n)
    // where n = this->degree()
    std::vector<GaloisFieldDict> gf_frobenius_monomial_base() const;
    // computes `f**n % (*this)` in modulo_
    GaloisFieldDict gf_pow_mod(const GaloisFieldDict &f,
                               const unsigned long &n) const;
    // uses Frobenius Map to find g.gf_pow_mod(*this, modulo_)
    // i.e. `(*this)**modulo_ % g`
    GaloisFieldDict
    gf_frobenius_map(const GaloisFieldDict &g,
                     const std::vector<GaloisFieldDict> &b) const;
    std::pair<GaloisFieldDict, GaloisFieldDict>
    gf_trace_map(const GaloisFieldDict &a, const GaloisFieldDict &b,
                 const GaloisFieldDict &c, const unsigned long &n) const;
    GaloisFieldDict _gf_trace_map(const GaloisFieldDict &f,
                                  const unsigned long &n,
                                  const std::vector<GaloisFieldDict> &b) const;
    // For a monic square-free polynomial in modulo_, it returns its distinct
    // degree factorization. Each element's first is a factor and second
    // is used by equal degree factorization. (Zassenhaus's algorithm)
    std::vector<std::pair<GaloisFieldDict, unsigned>> gf_ddf_zassenhaus() const;
    // Computes `f**((modulo_**n - 1) // 2) % *this`
    GaloisFieldDict _gf_pow_pnm1d2(const GaloisFieldDict &f, const unsigned &n,
                                   const std::vector<GaloisFieldDict> &b) const;
    // Generates a random polynomial in `modulo_` of degree `n`.
    GaloisFieldDict gf_random(const unsigned int &n_val,
                              mp_randstate &state) const;

    // Given a monic square-free polynomial and an integer `n`, such that `n`
    // divides `this->degree()`,
    // returns all irreducible factors, each of degree `n`.
    std::set<GaloisFieldDict, DictLess>
    gf_edf_zassenhaus(const unsigned &n) const;
    // For a monic square-free polynomial in modulo_, it returns its distinct
    // degree factorization. Each element's first is a factor and second
    // is used by equal degree factorization. (Shoup's algorithm)
    // Factors a polynomial in field of modulo_
    std::vector<std::pair<GaloisFieldDict, unsigned>> gf_ddf_shoup() const;
    // Equal degree factorization using Shoup's algorithm.
    std::set<GaloisFieldDict, DictLess> gf_edf_shoup(const unsigned &n) const;
    // Factors a square free polynomial in field of modulo_ using Zassenhaus's
    // algorithm.
    // References :
    //     1.) J. von zur Gathen, J. Gerhard, Modern Computer Algebra, 1999
    //     2.) K. Geddes, S. R. Czapor, G. Labahn, Algorithms for Computer
    //     Algebra, 1992
    std::set<GaloisFieldDict, DictLess> gf_zassenhaus() const;
    // Factors a square free polynomial in field of modulo_ using Shoup's
    // algorithm.
    // References :
    //     1.) V. Shoup, A New Polynomial Factorization Algorithm and its
    //     Implementation,1995
    //     2.) E. Kaltofen, V. Shoup, Subquadratic-time Factoring of Polynomials
    //     over Finite Fields, 1998
    //     3.) J. von zur Gathen, V. Shoup, Computing Frobenius Maps and
    //     Factoring Polynomials, 1992
    //     4.) V. Shoup, A Fast Deterministic Algorithm for Factoring
    //     Polynomials over Finite Fields of Small Characteristic, 1991
    std::set<GaloisFieldDict, DictLess> gf_shoup() const;
    std::pair<integer_class,
              std::set<std::pair<GaloisFieldDict, unsigned>, DictLess>>
    gf_factor() const;

    GaloisFieldDict &operator=(GaloisFieldDict &&other) SYMENGINE_NOEXCEPT
    {
        if (this != &other) {
            dict_ = std::move(other.dict_);
            modulo_ = std::move(other.modulo_);
        }
        return down_cast<GaloisFieldDict &>(*this);
    }

    template <typename T>
    friend GaloisFieldDict operator+(const GaloisFieldDict &a, const T &b)
    {
        GaloisFieldDict c = a;
        c += b;
        return c;
    }

    GaloisFieldDict &operator+=(const GaloisFieldDict &other)
    {
        if (modulo_ != other.modulo_)
            throw SymEngineException("Error: field must be same.");
        if (other.dict_.size() == 0)
            return down_cast<GaloisFieldDict &>(*this);
        if (this->dict_.size() == 0) {
            *this = other;
            return down_cast<GaloisFieldDict &>(*this);
        }
        if (other.dict_.size() < this->dict_.size()) {
            for (unsigned int i = 0; i < other.dict_.size(); i++) {
                integer_class temp;
                temp += dict_[i];
                temp += other.dict_[i];
                if (temp != integer_class(0)) {
                    mp_fdiv_r(temp, temp, modulo_);
                }
                dict_[i] = temp;
            }
        } else {
            for (unsigned int i = 0; i < dict_.size(); i++) {
                integer_class temp;
                temp += dict_[i];
                temp += other.dict_[i];
                if (temp != integer_class(0)) {
                    mp_fdiv_r(temp, temp, modulo_);
                }
                dict_[i] = temp;
            }
            if (other.dict_.size() == this->dict_.size())
                gf_istrip();
            else
                dict_.insert(dict_.end(), other.dict_.begin() + dict_.size(),
                             other.dict_.end());
        }
        return down_cast<GaloisFieldDict &>(*this);
    }

    GaloisFieldDict &operator+=(const integer_class &other)
    {
        if (dict_.empty() or other == integer_class(0))
            return down_cast<GaloisFieldDict &>(*this);
        integer_class temp = dict_[0] + other;
        mp_fdiv_r(temp, temp, modulo_);
        dict_[0] = temp;
        if (dict_.size() == 1)
            gf_istrip();
        return down_cast<GaloisFieldDict &>(*this);
    }

    template <typename T>
    friend GaloisFieldDict operator-(const GaloisFieldDict &a, const T &b)
    {
        GaloisFieldDict c = a;
        c -= b;
        return c;
    }
    GaloisFieldDict operator-() const
    {
        GaloisFieldDict o(*this);
        for (auto &a : o.dict_) {
            a *= -1;
            if (a != 0_z)
                a += modulo_;
        }
        return o;
    }

    GaloisFieldDict &negate();

    GaloisFieldDict &operator-=(const integer_class &other)
    {
        return *this += (-1 * other);
    }

    GaloisFieldDict &operator-=(const GaloisFieldDict &other)
    {
        if (modulo_ != other.modulo_)
            throw SymEngineException("Error: field must be same.");
        if (other.dict_.size() == 0)
            return down_cast<GaloisFieldDict &>(*this);
        if (this->dict_.size() == 0) {
            *this = -other;
            return down_cast<GaloisFieldDict &>(*this);
        }
        if (other.dict_.size() < this->dict_.size()) {
            for (unsigned int i = 0; i < other.dict_.size(); i++) {
                integer_class temp;
                temp += dict_[i];
                temp -= other.dict_[i];
                if (temp != integer_class(0)) {
                    mp_fdiv_r(temp, temp, modulo_);
                }
                dict_[i] = temp;
            }
        } else {
            for (unsigned int i = 0; i < dict_.size(); i++) {
                integer_class temp;
                temp += dict_[i];
                temp -= other.dict_[i];
                if (temp != integer_class(0)) {
                    mp_fdiv_r(temp, temp, modulo_);
                }
                dict_[i] = temp;
            }
            if (other.dict_.size() == this->dict_.size())
                gf_istrip();
            else {
                auto orig_size = dict_.size();
                dict_.resize(other.dict_.size());
                for (auto i = orig_size; i < other.dict_.size(); i++) {
                    dict_[i] = -other.dict_[i];
                    if (dict_[i] != 0_z)
                        dict_[i] += modulo_;
                }
            }
        }
        return down_cast<GaloisFieldDict &>(*this);
    }

    static GaloisFieldDict mul(const GaloisFieldDict &a,
                               const GaloisFieldDict &b);

    friend GaloisFieldDict operator*(const GaloisFieldDict &a,
                                     const GaloisFieldDict &b)
    {
        return GaloisFieldDict::mul(a, b);
    }

    GaloisFieldDict &operator*=(const integer_class &other)
    {
        if (dict_.empty())
            return down_cast<GaloisFieldDict &>(*this);

        if (other == integer_class(0)) {
            dict_.clear();
            return down_cast<GaloisFieldDict &>(*this);
        }

        for (auto &arg : dict_) {
            if (arg != integer_class(0)) {
                arg *= other;
                mp_fdiv_r(arg, arg, modulo_);
            }
        }
        gf_istrip();
        return down_cast<GaloisFieldDict &>(*this);
    }

    GaloisFieldDict &operator*=(const GaloisFieldDict &other)
    {
        if (modulo_ != other.modulo_)
            throw SymEngineException("Error: field must be same.");
        if (dict_.empty())
            return down_cast<GaloisFieldDict &>(*this);

        auto o_dict = other.dict_;
        if (o_dict.empty()) {
            dict_.clear();
            return down_cast<GaloisFieldDict &>(*this);
        }

        // ! other is a just constant term
        if (o_dict.size() == 1) {
            for (auto &arg : dict_) {
                if (arg != integer_class(0)) {
                    arg *= o_dict[0];
                    mp_fdiv_r(arg, arg, modulo_);
                }
            }
            gf_istrip();
            return down_cast<GaloisFieldDict &>(*this);
        }
        // mul will return a stripped dict
        GaloisFieldDict res
            = GaloisFieldDict::mul(down_cast<GaloisFieldDict &>(*this), other);
        res.dict_.swap(this->dict_);
        return down_cast<GaloisFieldDict &>(*this);
    }

    template <class T>
    friend GaloisFieldDict operator/(const GaloisFieldDict &a, const T &b)
    {
        GaloisFieldDict c = a;
        c /= b;
        return c;
    }

    GaloisFieldDict &operator/=(const integer_class &other)
    {
        if (other == integer_class(0)) {
            throw DivisionByZeroError("ZeroDivisionError");
        }
        if (dict_.empty())
            return down_cast<GaloisFieldDict &>(*this);
        integer_class inv;
        mp_invert(inv, other, modulo_);
        for (auto &arg : dict_) {
            if (arg != integer_class(0)) {
                arg *= inv;
                mp_fdiv_r(arg, arg, modulo_);
            }
        }
        gf_istrip();
        return down_cast<GaloisFieldDict &>(*this);
    }

    GaloisFieldDict &operator/=(const GaloisFieldDict &other)
    {
        if (modulo_ != other.modulo_)
            throw SymEngineException("Error: field must be same.");
        auto dict_divisor = other.dict_;
        if (dict_divisor.empty()) {
            throw DivisionByZeroError("ZeroDivisionError");
        }
        if (dict_.empty())
            return down_cast<GaloisFieldDict &>(*this);
        integer_class inv;
        mp_invert(inv, *(dict_divisor.rbegin()), modulo_);

        // ! other is a just constant term
        if (dict_divisor.size() == 1) {
            for (auto &iter : dict_) {
                if (iter != 0) {
                    iter *= inv;
                    mp_fdiv_r(iter, iter, modulo_);
                }
            }
            return down_cast<GaloisFieldDict &>(*this);
        }
        std::vector<integer_class> dict_out;
        size_t deg_dividend = this->degree();
        size_t deg_divisor = other.degree();
        if (deg_dividend < deg_divisor) {
            dict_.clear();
            return down_cast<GaloisFieldDict &>(*this);
        }
        dict_out.swap(dict_);
        dict_.resize(deg_dividend - deg_divisor + 1);
        integer_class coeff;
        for (auto riter = deg_dividend; riter >= deg_divisor; --riter) {
            coeff = dict_out[riter];
            auto lb = deg_divisor + riter > deg_dividend
                          ? deg_divisor + riter - deg_dividend
                          : 0;
            auto ub = std::min(riter + 1, deg_divisor);
            for (auto j = lb; j < ub; ++j) {
                mp_addmul(coeff, dict_out[riter - j + deg_divisor],
                          -dict_divisor[j]);
            }
            coeff *= inv;
            mp_fdiv_r(coeff, coeff, modulo_);
            dict_out[riter] = dict_[riter - deg_divisor] = coeff;
        }
        gf_istrip();
        return down_cast<GaloisFieldDict &>(*this);
    }

    template <class T>
    friend GaloisFieldDict operator%(const GaloisFieldDict &a, const T &b)
    {
        GaloisFieldDict c = a;
        c %= b;
        return c;
    }

    GaloisFieldDict &operator%=(const integer_class &other)
    {
        if (other == integer_class(0)) {
            throw DivisionByZeroError("ZeroDivisionError");
        }
        if (dict_.empty())
            return down_cast<GaloisFieldDict &>(*this);
        dict_.clear();
        return down_cast<GaloisFieldDict &>(*this);
    }

    GaloisFieldDict &operator%=(const GaloisFieldDict &other)
    {
        if (modulo_ != other.modulo_)
            throw SymEngineException("Error: field must be same.");
        auto dict_divisor = other.dict_;
        if (dict_divisor.empty()) {
            throw DivisionByZeroError("ZeroDivisionError");
        }
        if (dict_.empty())
            return down_cast<GaloisFieldDict &>(*this);
        integer_class inv;
        mp_invert(inv, *(dict_divisor.rbegin()), modulo_);

        // ! other is a just constant term
        if (dict_divisor.size() == 1) {
            dict_.clear();
            return down_cast<GaloisFieldDict &>(*this);
        }
        std::vector<integer_class> dict_out;
        size_t deg_dividend = this->degree();
        size_t deg_divisor = other.degree();
        if (deg_dividend < deg_divisor) {
            return down_cast<GaloisFieldDict &>(*this);
        }
        dict_out.swap(dict_);
        dict_.resize(deg_divisor);
        integer_class coeff;
        for (auto it = deg_dividend + 1; it-- != 0;) {
            coeff = dict_out[it];
            auto lb = deg_divisor + it > deg_dividend
                          ? deg_divisor + it - deg_dividend
                          : 0;
            auto ub = std::min(it + 1, deg_divisor);
            for (size_t j = lb; j < ub; ++j) {
                mp_addmul(coeff, dict_out[it - j + deg_divisor],
                          -dict_divisor[j]);
            }
            if (it >= deg_divisor) {
                coeff *= inv;
                mp_fdiv_r(coeff, coeff, modulo_);
                dict_out[it] = coeff;
            } else {
                mp_fdiv_r(coeff, coeff, modulo_);
                dict_out[it] = dict_[it] = coeff;
            }
        }
        gf_istrip();
        return down_cast<GaloisFieldDict &>(*this);
    }

    static GaloisFieldDict pow(const GaloisFieldDict &a, unsigned int p)
    {
        return a.gf_pow(p);
    }

    bool operator==(const GaloisFieldDict &other) const
    {
        return dict_ == other.dict_ and modulo_ == other.modulo_;
    }

    bool operator!=(const GaloisFieldDict &other) const
    {
        return not(*this == other);
    }

    size_t size() const
    {
        return dict_.size();
    }

    bool empty() const
    {
        return dict_.empty();
    }

    unsigned degree() const
    {
        if (dict_.empty())
            return 0;
        return numeric_cast<unsigned>(dict_.size()) - 1;
    }

    const std::vector<integer_class> &get_dict() const
    {
        return dict_;
    }

    void gf_istrip();

    bool is_one() const
    {
        if (dict_.size() == 1)
            if (dict_[0] == integer_class(1))
                return true;
        return false;
    }

    integer_class get_coeff(unsigned int x) const
    {
        if (x <= degree())
            return dict_[x];
        return 0_z;
    }
};

class GaloisField : public UIntPolyBase<GaloisFieldDict, GaloisField>
{
public:
    IMPLEMENT_TYPEID(SYMENGINE_GALOISFIELD)

    //! Constructor of GaloisField class
    GaloisField(const RCP<const Basic> &var, GaloisFieldDict &&dict);

    //! \return true if canonical
    bool is_canonical(const GaloisFieldDict &dict) const;
    //! \return size of the hash
    hash_t __hash__() const override;
    int compare(const Basic &o) const override;

    // creates a GaloisField in cannonical form based on the
    // dictionary.
    static RCP<const GaloisField> from_dict(const RCP<const Basic> &var,
                                            GaloisFieldDict &&d);
    static RCP<const GaloisField> from_vec(const RCP<const Basic> &var,
                                           const std::vector<integer_class> &v,
                                           const integer_class &modulo);
    static RCP<const GaloisField> from_uintpoly(const UIntPoly &a,
                                                const integer_class &modulo);

    integer_class eval(const integer_class &x) const override
    {
        return get_poly().gf_eval(x);
    }

    vec_integer_class multieval(const vec_integer_class &v) const
    {
        return get_poly().gf_multi_eval(v);
    }

    typedef vec_integer_class::const_iterator iterator;
    typedef vec_integer_class::const_reverse_iterator reverse_iterator;
    iterator begin() const
    {
        return get_poly().dict_.begin();
    }
    iterator end() const
    {
        return get_poly().dict_.end();
    }
    reverse_iterator obegin() const
    {
        return get_poly().dict_.rbegin();
    }
    reverse_iterator oend() const
    {
        return get_poly().dict_.rend();
    }

    inline integer_class get_coeff(unsigned int x) const override
    {
        return get_poly().get_coeff(x);
    }

    vec_basic get_args() const override;
    inline const std::vector<integer_class> &get_dict() const
    {
        return get_poly().dict_;
    }

    inline int size() const override
    {
        if (get_poly().empty())
            return 0;
        return get_degree() + 1;
    }
};

inline RCP<const GaloisField> gf_poly(RCP<const Basic> i,
                                      GaloisFieldDict &&dict)
{
    return GaloisField::from_dict(i, std::move(dict));
}

inline RCP<const GaloisField> gf_poly(RCP<const Basic> i, map_uint_mpz &&dict,
                                      integer_class modulo_)
{
    GaloisFieldDict wrapper(dict, modulo_);
    return GaloisField::from_dict(i, std::move(wrapper));
}

inline RCP<const GaloisField> pow_upoly(const GaloisField &a, unsigned int p)
{
    auto dict = GaloisField::container_type::pow(a.get_poly(), p);
    return GaloisField::from_container(a.get_var(), std::move(dict));
}
} // namespace SymEngine

#endif
