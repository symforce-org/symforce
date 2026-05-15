#include <symengine/fields.h>
#include <symengine/add.h>
#include <symengine/constants.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/symengine_exception.h>

namespace SymEngine
{
GaloisField::GaloisField(const RCP<const Basic> &var, GaloisFieldDict &&dict)
    : UIntPolyBase(var, std::move(dict))
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(is_canonical(get_poly()))
}

bool GaloisField::is_canonical(const GaloisFieldDict &dict) const
{
    // Check if dictionary contains terms with coeffienct 0
    if (dict.modulo_ <= integer_class(0))
        return false;
    if (not dict.empty())
        if (dict.dict_[dict.dict_.size() - 1] == integer_class(0))
            return false;
    return true;
}

hash_t GaloisField::__hash__() const
{
    hash_t seed = SYMENGINE_GALOISFIELD;

    seed += get_var()->hash();
    for (const auto &it : get_poly().dict_) {
        hash_t temp = SYMENGINE_GALOISFIELD;
        hash_combine<hash_t>(temp, mp_get_si(it));
        seed += temp;
    }
    return seed;
}

int GaloisField::compare(const Basic &o) const
{
    const GaloisField &s = down_cast<const GaloisField &>(o);

    if (get_poly().size() != s.get_poly().size())
        return (get_poly().size() < s.get_poly().size()) ? -1 : 1;

    int cmp = unified_compare(get_var(), s.get_var());
    if (cmp != 0)
        return cmp;

    cmp = unified_compare(get_poly().modulo_, s.get_poly().modulo_);
    if (cmp != 0)
        return cmp;

    return unified_compare(get_poly().dict_, s.get_poly().dict_);
}

RCP<const GaloisField> GaloisField::from_dict(const RCP<const Basic> &var,
                                              GaloisFieldDict &&d)
{
    return make_rcp<const GaloisField>(var, std::move(d));
}

RCP<const GaloisField>
GaloisField::from_vec(const RCP<const Basic> &var,
                      const std::vector<integer_class> &v,
                      const integer_class &modulo)
{
    return make_rcp<const GaloisField>(var,
                                       GaloisFieldDict::from_vec(v, modulo));
}

RCP<const GaloisField> GaloisField::from_uintpoly(const UIntPoly &a,
                                                  const integer_class &modulo)
{
    GaloisFieldDict wrapper(a.get_poly().get_dict(), modulo);
    return GaloisField::from_dict(a.get_var(), std::move(wrapper));
}

vec_basic GaloisField::get_args() const
{
    vec_basic args;
    if (get_poly().dict_.empty())
        args.push_back(zero);
    else {
        for (unsigned i = 0; i < get_poly().dict_.size(); i++) {
            if (get_poly().dict_[i] == integer_class(0))
                continue;
            if (i == 0) {
                args.push_back(integer(get_poly().dict_[i]));
            } else if (i == 1) {
                if (get_poly().dict_[i] == 1) {
                    args.push_back(get_var());
                } else {
                    args.push_back(Mul::from_dict(integer(get_poly().dict_[i]),
                                                  {{get_var(), one}}));
                }
            } else {
                if (get_poly().dict_[i] == 1) {
                    args.push_back(pow(get_var(), integer(i)));
                } else {
                    args.push_back(Mul::from_dict(integer(get_poly().dict_[i]),
                                                  {{get_var(), integer(i)}}));
                }
            }
        }
    }
    return args;
}

GaloisFieldDict::GaloisFieldDict(const int &i, const integer_class &mod)
    : modulo_(mod)
{
    integer_class temp;
    mp_fdiv_r(temp, integer_class(i), modulo_);
    if (temp != integer_class(0))
        dict_.insert(dict_.begin(), temp);
}

GaloisFieldDict::GaloisFieldDict(const map_uint_mpz &p,
                                 const integer_class &mod)
    : modulo_(mod)
{
    if (p.size() != 0) {
        dict_.resize(p.rbegin()->first + 1, integer_class(0));
        for (auto &iter : p) {
            integer_class temp;
            mp_fdiv_r(temp, iter.second, modulo_);
            dict_[iter.first] = temp;
        }
        gf_istrip();
    }
}

GaloisFieldDict::GaloisFieldDict(const integer_class &i,
                                 const integer_class &mod)
    : modulo_(mod)
{
    integer_class temp;
    mp_fdiv_r(temp, i, modulo_);
    if (temp != integer_class(0))
        dict_.insert(dict_.begin(), temp);
}

GaloisFieldDict GaloisFieldDict::from_vec(const std::vector<integer_class> &v,
                                          const integer_class &modulo)
{
    GaloisFieldDict x;
    x.modulo_ = modulo;
    x.dict_.resize(v.size());
    for (unsigned int i = 0; i < v.size(); ++i) {
        integer_class a;
        mp_fdiv_r(a, v[i], modulo);
        x.dict_[i] = a;
    }
    x.gf_istrip();
    return x;
}

GaloisFieldDict &GaloisFieldDict::negate()
{
    for (auto &a : dict_) {
        a *= -1;
        if (a != 0_z)
            a += modulo_;
    }
    return down_cast<GaloisFieldDict &>(*this);
}

void GaloisFieldDict::gf_istrip()
{
    for (auto i = dict_.size(); i-- != 0;) {
        if (dict_[i] == integer_class(0))
            dict_.pop_back();
        else
            break;
    }
}

GaloisFieldDict GaloisFieldDict::mul(const GaloisFieldDict &a,
                                     const GaloisFieldDict &b)
{
    if (a.modulo_ != b.modulo_)
        throw std::runtime_error("Error: field must be same.");
    if (a.get_dict().empty())
        return a;
    if (b.get_dict().empty())
        return b;

    GaloisFieldDict p;
    p.dict_.resize(a.degree() + b.degree() + 1, integer_class(0));
    p.modulo_ = a.modulo_;
    for (unsigned int i = 0; i <= a.degree(); i++)
        for (unsigned int j = 0; j <= b.degree(); j++) {
            auto temp = a.dict_[i];
            temp *= b.dict_[j];
            if (temp != integer_class(0)) {
                auto t = p.dict_[i + j];
                t += temp;
                mp_fdiv_r(t, t, a.modulo_);
                p.dict_[i + j] = t;
            }
        }
    p.gf_istrip();
    return p;
}

void GaloisFieldDict::gf_div(const GaloisFieldDict &o,
                             const Ptr<GaloisFieldDict> &quo,
                             const Ptr<GaloisFieldDict> &rem) const
{
    if (modulo_ != o.modulo_)
        throw SymEngineException("Error: field must be same.");
    if (o.dict_.empty())
        throw DivisionByZeroError("ZeroDivisionError");
    std::vector<integer_class> dict_out;
    if (dict_.empty()) {
        *quo = GaloisFieldDict::from_vec(dict_out, modulo_);
        *rem = GaloisFieldDict::from_vec(dict_, modulo_);
        return;
    }
    auto dict_divisor = o.dict_;
    auto deg_dividend = this->degree();
    auto deg_divisor = o.degree();
    if (deg_dividend < deg_divisor) {
        *quo = GaloisFieldDict::from_vec(dict_out, modulo_);
        *rem = GaloisFieldDict::from_vec(dict_, modulo_);
    } else {
        dict_out = dict_;
        integer_class inv;
        mp_invert(inv, *(dict_divisor.rbegin()), modulo_);
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
            if (it >= deg_divisor)
                coeff *= inv;
            mp_fdiv_r(coeff, coeff, modulo_);
            dict_out[it] = coeff;
        }
        std::vector<integer_class> dict_rem, dict_quo;
        dict_rem.resize(deg_divisor);
        dict_quo.resize(deg_dividend - deg_divisor + 1);
        for (unsigned it = 0; it < dict_out.size(); it++) {
            if (it < deg_divisor)
                dict_rem[it] = dict_out[it];
            else
                dict_quo[it - deg_divisor] = dict_out[it];
        }
        *quo = GaloisFieldDict::from_vec(dict_quo, modulo_);
        *rem = GaloisFieldDict::from_vec(dict_rem, modulo_);
    }
}

GaloisFieldDict GaloisFieldDict::gf_lshift(const integer_class n) const
{
    std::vector<integer_class> dict_out;
    auto to_ret = GaloisFieldDict::from_vec(dict_out, modulo_);
    if (!dict_.empty()) {
        auto n_val = mp_get_ui(n);
        to_ret.dict_.resize(n_val, integer_class(0));
        to_ret.dict_.insert(to_ret.dict_.end(), dict_.begin(), dict_.end());
    }
    return to_ret;
}

void GaloisFieldDict::gf_rshift(const integer_class n,
                                const Ptr<GaloisFieldDict> &quo,
                                const Ptr<GaloisFieldDict> &rem) const
{
    std::vector<integer_class> dict_quo;
    *quo = GaloisFieldDict::from_vec(dict_quo, modulo_);
    auto n_val = mp_get_ui(n);
    if (n_val < dict_.size()) {
        quo->dict_.insert(quo->dict_.end(), dict_.begin() + n_val, dict_.end());
        std::vector<integer_class> dict_rem(dict_.begin(),
                                            dict_.begin() + n_val);
        *rem = GaloisFieldDict::from_vec(dict_rem, modulo_);
    } else {
        *rem = down_cast<const GaloisFieldDict &>(*this);
    }
}

GaloisFieldDict GaloisFieldDict::gf_sqr() const
{
    return (*this * *this);
}

GaloisFieldDict GaloisFieldDict::gf_pow(const unsigned long n) const
{
    if (n == 0) {
        return GaloisFieldDict({integer_class(1)}, modulo_);
    }
    if (n == 1)
        return down_cast<const GaloisFieldDict &>(*this);
    if (n == 2)
        return gf_sqr();
    auto num = n;
    GaloisFieldDict to_sq = down_cast<const GaloisFieldDict &>(*this);
    GaloisFieldDict to_ret = GaloisFieldDict({integer_class(1)}, modulo_);
    while (1) {
        if (num & 1) {
            to_ret *= to_sq;
        }
        num >>= 1;
        if (num == 0)
            return to_ret;
        to_sq = to_sq.gf_sqr();
    }
}

void GaloisFieldDict::gf_monic(integer_class &res,
                               const Ptr<GaloisFieldDict> &monic) const
{
    *monic = down_cast<const GaloisFieldDict &>(*this);
    if (dict_.empty()) {
        res = integer_class(0);
    } else {
        res = *dict_.rbegin();
        if (res != integer_class(1)) {
            integer_class inv, temp;
            mp_invert(inv, res, modulo_);
            for (auto &iter : monic->dict_) {
                temp = inv;
                temp *= iter;
                mp_fdiv_r(iter, temp, modulo_);
            }
        }
    }
}

GaloisFieldDict GaloisFieldDict::gf_gcd(const GaloisFieldDict &o) const
{
    if (modulo_ != o.modulo_)
        throw SymEngineException("Error: field must be same.");
    GaloisFieldDict f = down_cast<const GaloisFieldDict &>(*this);
    GaloisFieldDict g = o;
    GaloisFieldDict temp_out;
    while (not g.dict_.empty()) {
        f %= g; // f, g = f % g, g
        f.dict_.swap(g.dict_);
    }
    integer_class temp_LC;
    f.gf_monic(temp_LC, outArg(f));
    return f;
}

GaloisFieldDict GaloisFieldDict::gf_lcm(const GaloisFieldDict &o) const
{
    if (modulo_ != o.modulo_)
        throw SymEngineException("Error: field must be same.");
    if (dict_.empty())
        return down_cast<const GaloisFieldDict &>(*this);
    if (o.dict_.empty())
        return o;
    GaloisFieldDict out, temp_out;
    out = o * (*this);
    out /= gf_gcd(o);
    integer_class temp_LC;
    out.gf_monic(temp_LC, outArg(out));
    return out;
}

GaloisFieldDict GaloisFieldDict::gf_diff() const
{
    auto df = degree();
    GaloisFieldDict out = GaloisFieldDict({}, modulo_);
    out.dict_.resize(df, integer_class(0));
    for (unsigned i = 1; i <= df; i++) {
        if (dict_[i] != integer_class(0)) {
            out.dict_[i - 1] = i * dict_[i];
            mp_fdiv_r(out.dict_[i - 1], out.dict_[i - 1], modulo_);
        }
    }
    out.gf_istrip();
    return out;
}

integer_class GaloisFieldDict::gf_eval(const integer_class &a) const
{
    integer_class res = 0_z;
    for (auto rit = dict_.rbegin(); rit != dict_.rend(); ++rit) {
        res *= a;
        res += (*rit);
        res %= modulo_;
    }
    return res;
}

vec_integer_class
GaloisFieldDict::gf_multi_eval(const vec_integer_class &v) const
{
    vec_integer_class res(v.size());
    for (unsigned int i = 0; i < v.size(); ++i)
        res[i] = gf_eval(v[i]);
    return res;
}

bool GaloisFieldDict::gf_is_sqf() const
{
    if (dict_.empty())
        return true;
    integer_class LC;
    GaloisFieldDict monic;
    gf_monic(LC, outArg(monic));
    monic = monic.gf_gcd(monic.gf_diff());
    return monic.is_one();
}

std::vector<std::pair<GaloisFieldDict, unsigned>>
GaloisFieldDict::gf_sqf_list() const
{
    std::vector<std::pair<GaloisFieldDict, unsigned>> vec_out;
    if (degree() < 1)
        return vec_out;
    unsigned n = 1;
    // This cast is okay, because the multiplicities are unsigned
    unsigned r = numeric_cast<unsigned>(mp_get_ui(modulo_));
    bool sqf = false;
    integer_class LC;
    GaloisFieldDict f;
    gf_monic(LC, outArg(f));
    while (true) {
        GaloisFieldDict F = f.gf_diff();
        if (not F.dict_.empty()) {
            GaloisFieldDict g = f.gf_gcd(F);
            GaloisFieldDict h = f / g;

            unsigned i = 1;

            while (not h.is_one()) {
                GaloisFieldDict G = h.gf_gcd(g);
                GaloisFieldDict H = h / G;

                if (H.degree() > 0)
                    vec_out.push_back({H, i * n});

                ++i;
                g /= G;
                h = G;
            }
            if (g.is_one())
                sqf = true;
            else
                f = g;
        }
        if (not sqf) {
            auto deg = f.degree();
            auto d = deg / r;
            GaloisFieldDict temp = f;
            for (unsigned int i = 0; i <= d; ++i) {
                f.dict_[d - i] = temp.dict_[deg - i * r];
            }
            n *= r;
            f.dict_.resize(d + 1);
            f.gf_istrip();
        } else
            break;
    }
    return vec_out;
}

GaloisFieldDict GaloisFieldDict::gf_sqf_part() const
{
    auto sqf = gf_sqf_list();
    GaloisFieldDict g = GaloisFieldDict::from_vec({1_z}, modulo_);

    for (auto &f : sqf)
        g *= f.first;

    return g;
}

GaloisFieldDict GaloisFieldDict::gf_compose_mod(const GaloisFieldDict &g,
                                                const GaloisFieldDict &h) const
{
    if (g.modulo_ != h.modulo_)
        throw SymEngineException("Error: field must be same.");
    if (g.modulo_ != modulo_)
        throw SymEngineException("Error: field must be same.");
    if (g.dict_.size() == 0)
        return g;
    GaloisFieldDict out
        = GaloisFieldDict::from_vec({*(g.dict_.rbegin())}, modulo_);
    if (g.dict_.size() >= 2) {
        for (auto i = g.dict_.size() - 2;; --i) {
            out *= h;
            out += g.dict_[i];
            out %= (*this);
            if (i == 0)
                break;
        }
    }
    return out;
}

GaloisFieldDict GaloisFieldDict::gf_pow_mod(const GaloisFieldDict &f,
                                            const unsigned long &n) const
{
    if (modulo_ != f.modulo_)
        throw SymEngineException("Error: field must be same.");
    if (n == 0)
        return GaloisFieldDict::from_vec({1_z}, modulo_);
    GaloisFieldDict in = f;
    if (n == 1) {
        return f % (*this);
    }
    if (n == 2) {
        return f.gf_sqr() % (*this);
    }
    GaloisFieldDict h = GaloisFieldDict::from_vec({1_z}, modulo_);
    auto mod = n;
    while (true) {
        if (mod & 1) {
            h *= in;
            h %= *this;
        }
        mod >>= 1;

        if (mod == 0)
            break;

        in = in.gf_sqr() % *this;
    }
    return h;
}

std::vector<GaloisFieldDict> GaloisFieldDict::gf_frobenius_monomial_base() const
{
    auto n = degree();
    std::vector<GaloisFieldDict> b;
    if (n == 0)
        return b;
    b.resize(n);
    b[0] = GaloisFieldDict::from_vec({1_z}, modulo_);
    GaloisFieldDict temp_out;
    if (mp_get_ui(modulo_) < n) {
        for (unsigned i = 1; i < n; ++i) {
            b[i] = b[i - 1].gf_lshift(modulo_);
            b[i] %= (*this);
        }
    } else if (n > 1) {
        b[1] = gf_pow_mod(GaloisFieldDict::from_vec({0_z, 1_z}, modulo_),
                          mp_get_ui(modulo_));
        for (unsigned i = 2; i < n; ++i) {
            b[i] = b[i - 1] * b[1];
            b[i] %= (*this);
        }
    }
    return b;
}

GaloisFieldDict
GaloisFieldDict::gf_frobenius_map(const GaloisFieldDict &g,
                                  const std::vector<GaloisFieldDict> &b) const
{
    if (modulo_ != g.modulo_)
        throw SymEngineException("Error: field must be same.");
    auto m = g.degree();
    GaloisFieldDict temp_out(*this), out;
    if (this->degree() >= m) {
        temp_out %= g;
    }
    if (temp_out.empty()) {
        return temp_out;
    }
    m = temp_out.degree();
    out = GaloisFieldDict::from_vec({temp_out.dict_[0]}, modulo_);
    for (unsigned i = 1; i <= m; ++i) {
        auto v = b[i];
        v *= temp_out.dict_[i];
        out += v;
    }
    out.gf_istrip();
    return out;
}

std::pair<GaloisFieldDict, GaloisFieldDict> GaloisFieldDict::gf_trace_map(
    const GaloisFieldDict &a, const GaloisFieldDict &b,
    const GaloisFieldDict &c, const unsigned long &n) const
{
    unsigned long n_val(n);
    auto u = this->gf_compose_mod(a, b);
    GaloisFieldDict v(b), U, V;
    if (n_val & 1) {
        U = a + u;
        V = b;
    } else {
        U = a;
        V = c;
    }
    n_val >>= 1;
    while (n_val) {
        u += this->gf_compose_mod(u, v);
        v = gf_compose_mod(v, v);

        if (n_val & 1) {
            auto temp = gf_compose_mod(u, V);
            U += temp;
            V = gf_compose_mod(v, V);
        }
        n_val >>= 1;
    }
    return std::make_pair(gf_compose_mod(a, V), U);
}

GaloisFieldDict
GaloisFieldDict::_gf_trace_map(const GaloisFieldDict &f, const unsigned long &n,
                               const std::vector<GaloisFieldDict> &b) const
{
    GaloisFieldDict x = f % (*this);
    auto h = f;
    auto r = f;
    for (unsigned i = 1; i < n; ++i) {
        h = gf_frobenius_map(h, b);
        r += h;
        r %= (*this);
    }
    return r;
}

std::vector<std::pair<GaloisFieldDict, unsigned>>
GaloisFieldDict::gf_ddf_zassenhaus() const
{
    unsigned i = 1;
    GaloisFieldDict f(*this);
    GaloisFieldDict g = GaloisFieldDict::from_vec({0_z, 1_z}, modulo_);
    GaloisFieldDict to_sub(g);
    std::vector<std::pair<GaloisFieldDict, unsigned>> factors;

    auto b = f.gf_frobenius_monomial_base();
    while (2 * i <= f.degree()) {
        g = g.gf_frobenius_map(f, b);

        GaloisFieldDict h = f.gf_gcd(g - to_sub);

        if (not h.is_one()) {
            factors.push_back({h, i});
            f /= h;
            g %= f;
            b = f.gf_frobenius_monomial_base();
        }
        ++i;
    }
    if (not(f.is_one() || f.empty())) {
        factors.push_back({f, f.degree()});
    }
    return factors;
}

GaloisFieldDict
GaloisFieldDict::_gf_pow_pnm1d2(const GaloisFieldDict &f, const unsigned &n,
                                const std::vector<GaloisFieldDict> &b) const
{
    GaloisFieldDict f_in(f);
    f_in %= *this;
    GaloisFieldDict h, r;
    h = r = f_in;
    for (unsigned i = 1; i < n; ++i) {
        h = h.gf_frobenius_map(*this, b);
        r *= h;
        r %= *this;
    }
    auto res = gf_pow_mod(r, (mp_get_ui(modulo_) - 1) / 2);
    return res;
}

GaloisFieldDict GaloisFieldDict::gf_random(const unsigned int &n_val,
                                           mp_randstate &state) const
{
    std::vector<integer_class> v(n_val + 1);
    for (unsigned i = 0; i < n_val; ++i) {
        state.urandomint(v[i], modulo_);
    }
    v[n_val] = 1_z;
    return GaloisFieldDict::from_vec(v, modulo_);
}

std::set<GaloisFieldDict, GaloisFieldDict::DictLess>
GaloisFieldDict::gf_edf_zassenhaus(const unsigned &n) const
{
    std::set<GaloisFieldDict, DictLess> factors;
    factors.insert(*this);
    if (this->degree() <= n)
        return factors;

    auto N = this->degree() / n;

    std::vector<GaloisFieldDict> b;
    if (modulo_ != 2_z)
        b = this->gf_frobenius_monomial_base();
    mp_randstate state;
    while (factors.size() < N) {
        auto r = gf_random(2 * n - 1, state);
        GaloisFieldDict g;
        if (modulo_ == 2_z) {
            GaloisFieldDict h = r;
            unsigned ub = 1 << (n * N - 1);
            for (unsigned i = 0; i < ub; ++i) {
                r = gf_pow_mod(r, 2);
                h += r;
            }
            g = this->gf_gcd(h);
        } else {
            GaloisFieldDict h = _gf_pow_pnm1d2(r, n, b);
            h -= 1_z;
            g = this->gf_gcd(h);
        }

        if (!g.is_one() and g != (*this)) {
            factors = g.gf_edf_zassenhaus(n);
            auto to_add = ((*this) / g).gf_edf_zassenhaus(n);
            if (not to_add.empty())
                factors.insert(to_add.begin(), to_add.end());
        }
    }
    return factors;
}

std::vector<std::pair<GaloisFieldDict, unsigned>>
GaloisFieldDict::gf_ddf_shoup() const
{
    std::vector<std::pair<GaloisFieldDict, unsigned>> factors;
    if (dict_.empty())
        return factors;
    GaloisFieldDict f(*this);
    auto n = this->degree();
    auto k = static_cast<unsigned>(std::ceil(std::sqrt(n / 2)));
    auto b = gf_frobenius_monomial_base();
    auto x = GaloisFieldDict::from_vec({0_z, 1_z}, modulo_);
    auto h = x.gf_frobenius_map(f, b);

    std::vector<GaloisFieldDict> U;
    U.push_back(x);
    U.push_back(h);
    U.resize(k + 1);
    for (unsigned i = 2; i <= k; ++i)
        U[i] = U[i - 1].gf_frobenius_map(*this, b);
    h = U[k];
    U.resize(k);
    std::vector<GaloisFieldDict> V;
    V.push_back(h);
    V.resize(k);
    for (unsigned i = 1; i + 1 <= k; ++i)
        V[i] = this->gf_compose_mod(V[i - 1], h);
    for (unsigned i = 0; i < V.size(); i++) {
        h = GaloisFieldDict::from_vec({1_z}, modulo_);
        auto j = k - 1;
        GaloisFieldDict g;
        for (auto &u : U) {
            g = V[i] - u;
            h *= g;
            h %= f;
        }
        g = f.gf_gcd(h);
        f /= g;
        for (auto rit = U.rbegin(); rit != U.rend(); ++rit) {
            h = V[i] - (*rit);
            auto F = g.gf_gcd(h);
            if (not F.is_one()) {
                unsigned temp = k * (i + 1) - j;
                factors.push_back({F, temp});
            }
            g /= F;
            --j;
        }
    }
    if (not f.is_one())
        factors.push_back({f, f.degree()});
    return factors;
}

std::set<GaloisFieldDict, GaloisFieldDict::DictLess>
GaloisFieldDict::gf_edf_shoup(const unsigned &n) const
{
    auto N = this->degree();
    std::set<GaloisFieldDict, DictLess> factors;
    if (N <= n) {
        if (N != 0)
            factors.insert(*this);
        return factors;
    }
    auto x = GaloisFieldDict::from_vec({0_z, 1_z}, modulo_);
    mp_randstate state;
    auto r = gf_random(N - 1, state);
    if (modulo_ == 2_z) {
        auto h = gf_pow_mod(x, mp_get_ui(modulo_));
        auto H = gf_trace_map(r, h, x, n - 1).second;
        auto h1 = gf_gcd(H);
        auto h2 = (*this) / h1;
        factors = h1.gf_edf_shoup(n);
        auto temp = h2.gf_edf_shoup(n);
        factors.insert(temp.begin(), temp.end());
    } else {
        auto b = gf_frobenius_monomial_base();
        auto H = _gf_trace_map(r, n, b);
        auto h = gf_pow_mod(H, (mp_get_ui(modulo_) - 1) / 2);
        auto h1 = gf_gcd(h);
        auto h2 = gf_gcd(h - 1_z);
        auto h3 = (*this) / (h1 * h2);
        factors = h1.gf_edf_shoup(n);
        auto temp = h2.gf_edf_shoup(n);
        factors.insert(temp.begin(), temp.end());
        temp = h3.gf_edf_shoup(n);
        factors.insert(temp.begin(), temp.end());
    }
    return factors;
}

std::set<GaloisFieldDict, GaloisFieldDict::DictLess>
GaloisFieldDict::gf_zassenhaus() const
{
    std::set<GaloisFieldDict, DictLess> factors;
    auto temp1 = gf_ddf_zassenhaus();
    for (auto &f : temp1) {
        auto temp2 = f.first.gf_edf_zassenhaus(f.second);
        factors.insert(temp2.begin(), temp2.end());
    }
    return factors;
}

std::set<GaloisFieldDict, GaloisFieldDict::DictLess>
GaloisFieldDict::gf_shoup() const
{
    std::set<GaloisFieldDict, DictLess> factors;
    auto temp1 = gf_ddf_shoup();
    for (auto &f : temp1) {
        auto temp2 = f.first.gf_edf_shoup(f.second);
        factors.insert(temp2.begin(), temp2.end());
    }
    return factors;
}

std::pair<integer_class, std::set<std::pair<GaloisFieldDict, unsigned>,
                                  GaloisFieldDict::DictLess>>
GaloisFieldDict::gf_factor() const
{
    integer_class lc;
    std::set<std::pair<GaloisFieldDict, unsigned>, DictLess> factors;
    GaloisFieldDict monic;
    gf_monic(lc, outArg(monic));
    if (monic.degree() < 1)
        return std::make_pair(lc, factors);
    std::vector<std::pair<GaloisFieldDict, unsigned>> sqf_list
        = monic.gf_sqf_list();
    for (auto a : sqf_list) {
        auto temp = (a.first).gf_zassenhaus();
        for (auto f : temp)
            factors.insert({f, a.second});
    }
    return std::make_pair(lc, factors);
}
} // namespace SymEngine
