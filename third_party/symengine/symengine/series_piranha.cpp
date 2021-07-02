#include <symengine/series_visitor.h>

namespace SymEngine
{

URatPSeriesPiranha::URatPSeriesPiranha(pp_t p, const std::string varname,
                                       const unsigned degree)
    : SeriesBase(std::move(p), varname, degree)
{
    SYMENGINE_ASSIGN_TYPEID()
}
RCP<const URatPSeriesPiranha>
URatPSeriesPiranha::series(const RCP<const Basic> &t, const std::string &x,
                           unsigned int prec)
{
    SeriesVisitor<pp_t, piranha::rational, URatPSeriesPiranha> visitor(pp_t(x),
                                                                       x, prec);
    return visitor.series(t);
}

hash_t URatPSeriesPiranha::__hash__() const
{
    hash_t seed = SYMENGINE_URATPSERIESPIRANHA;
    hash_combine(seed, p_.hash());
    hash_combine(seed, var_);
    hash_combine(seed, degree_);
    return seed;
}

int URatPSeriesPiranha::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<URatPSeriesPiranha>(o))
    const URatPSeriesPiranha &s = down_cast<const URatPSeriesPiranha &>(o);
    if (var_ != s.var_)
        return (var_ < s.var_) ? -1 : 1;
    if (degree_ != s.degree_)
        return (degree_ < s.degree_) ? -1 : 1;
    if (p_ == s.p_)
        return 0;
    return (p_.hash() < s.p_.hash()) ? -1 : 1;
}

piranha::integer URatPSeriesPiranha::convert(const Integer &x)
{
    return piranha::integer(get_mpz_t(x.as_integer_class()));
}

piranha::rational URatPSeriesPiranha::convert(const rational_class &x)
{
    piranha::integer i1(get_mpz_t(get_num(x)));
    piranha::integer i2(get_mpz_t(get_den(x)));
    piranha::rational r(i1);
    r /= i2;
    return r;
}

pp_t URatPSeriesPiranha::var(const std::string &s)
{
    return pp_t(s);
}

piranha::rational URatPSeriesPiranha::convert(const Rational &x)
{
    return convert(x.as_rational_class());
}

piranha::rational URatPSeriesPiranha::convert(const Basic &x)
{
    throw NotImplementedError("Not Implemented");
}

RCP<const Basic> URatPSeriesPiranha::as_basic() const
{
    RCP<const Symbol> x = symbol(var_);
    RCP<const Number> zcoef = integer(0);
    umap_basic_num dict_;
    for (const auto &it : p_) {
        if (it.first != 0) {
            rational_class cl_rat(it.first.get_mpq_view());
            canonicalize(cl_rat);
            RCP<const Number> co_basic = Rational::from_mpq(std::move(cl_rat));
            auto term = SymEngine::mul(
                SymEngine::pow(x, SymEngine::integer(it.second.degree())),
                co_basic);
            if (it.second.degree() == 0)
                zcoef = co_basic;
            Add::coef_dict_add_term(outArg(co_basic), dict_, one, term);
        }
    }
    return std::move(Add::from_dict(zcoef, std::move(dict_)));
}

umap_int_basic URatPSeriesPiranha::as_dict() const
{
    umap_int_basic map;
    for (const auto &it : p_) {
        if (it.first != 0) {
            rational_class cl_rat(it.first.get_mpq_view());
            RCP<const Basic> basic = Rational::from_mpq(std::move(cl_rat));
            map[it.second.degree()] = basic;
        }
    }
    return map;
}

RCP<const Basic> URatPSeriesPiranha::get_coeff(int i) const
{
    rational_class cl_rat(p_.find_cf({i}).get_mpq_view());
    return Rational::from_mpq(std::move(cl_rat));
}

pp_t URatPSeriesPiranha::mul(const pp_t &s, const pp_t &r, unsigned prec)
{
    pp_t::set_auto_truncate_degree(prec - 1);
    pp_t ret = s * r;
    pp_t::unset_auto_truncate_degree();
    return ret;
}

pp_t URatPSeriesPiranha::pow(const pp_t &s, int n, unsigned prec)
{
    pp_t::set_auto_truncate_degree(prec - 1);
    pp_t::clear_pow_cache();
    pp_t ret = s.pow(n);
    pp_t::unset_auto_truncate_degree();
    return ret;
}

unsigned URatPSeriesPiranha::ldegree(const pp_t &s)
{
    return s.ldegree();
}

piranha::rational URatPSeriesPiranha::find_cf(const pp_t &s, const pp_t &var,
                                              unsigned deg)
{
    return s.find_cf({deg});
}

piranha::rational URatPSeriesPiranha::root(piranha::rational &c, unsigned n)
{
    rational_class cl_rat(c.get_mpq_view());
    bool res;
    if (get_den(cl_rat) == 1) {
        // integer constant
        rational_class cl_root;
        res = mp_root(get_num(cl_root), get_num(cl_rat), n);
        if (not res)
            throw SymEngineException("constant term is not an nth power");
        return convert(cl_root);
    } else {
        RCP<const Rational> cterm = make_rcp<const Rational>(std::move(cl_rat));
        RCP<const Number> cout;
        res = cterm->nth_root(outArg(cout), n);
        if (not res)
            throw SymEngineException("constant term is not an nth power");
        return convert(down_cast<const Rational &>(*cout).as_rational_class());
    }
}

pp_t URatPSeriesPiranha::diff(const pp_t &s, const pp_t &var)
{
    return s.partial(var.get_symbol_set()[0].get_name());
}

pp_t URatPSeriesPiranha::integrate(const pp_t &s, const pp_t &var)
{
    return s.integrate(var.get_symbol_set()[0].get_name());
}

pp_t URatPSeriesPiranha::subs(const pp_t &s, const pp_t &var, const pp_t &r,
                              unsigned prec)
{
    pp_t::set_auto_truncate_degree(prec - 1);
    pp_t ret = s.subs(var.get_symbol_set()[0].get_name(), r);
    pp_t::unset_auto_truncate_degree();
    return ret;
}

UPSeriesPiranha::UPSeriesPiranha(p_expr p, const std::string varname,
                                 const unsigned degree)
    : SeriesBase(std::move(p), varname, degree)
{
    SYMENGINE_ASSIGN_TYPEID()
}

RCP<const UPSeriesPiranha> UPSeriesPiranha::series(const RCP<const Basic> &t,
                                                   const std::string &x,
                                                   unsigned int prec)
{
    SeriesVisitor<p_expr, Expression, UPSeriesPiranha> visitor(p_expr(x), x,
                                                               prec);
    return visitor.series(t);
}

hash_t UPSeriesPiranha::__hash__() const
{
    hash_t seed = SYMENGINE_URATPSERIESPIRANHA;
    hash_combine(seed, p_.hash());
    hash_combine(seed, var_);
    hash_combine(seed, degree_);
    return seed;
}

int UPSeriesPiranha::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<UPSeriesPiranha>(o))
    const UPSeriesPiranha &s = down_cast<const UPSeriesPiranha &>(o);
    if (var_ != s.var_)
        return (var_ < s.var_) ? -1 : 1;
    if (degree_ != s.degree_)
        return (degree_ < s.degree_) ? -1 : 1;
    if (p_ == s.p_)
        return 0;
    return (p_.hash() < s.p_.hash()) ? -1 : 1;
}

p_expr UPSeriesPiranha::var(const std::string &s)
{
    return p_expr(s);
}

Expression UPSeriesPiranha::convert(const Basic &x)
{
    return Expression(x.rcp_from_this());
}

RCP<const Basic> UPSeriesPiranha::as_basic() const
{
    RCP<const Symbol> x = symbol(var_);
    umap_basic_num dict_;
    for (const auto &it : p_) {
        if (it.first != 0) {
            auto term = SymEngine::mul(
                it.first.get_basic(),
                SymEngine::pow(x, SymEngine::integer(it.second.degree())));
            RCP<const Number> coef;
            coef = zero;
            Add::coef_dict_add_term(outArg(coef), dict_, one, term);
        }
    }
    return std::move(Add::from_dict(one, std::move(dict_)));
}

umap_int_basic UPSeriesPiranha::as_dict() const
{
    umap_int_basic map;
    for (const auto &it : p_) {
        if (it.first != 0) {
            map[it.second.degree()] = it.first.get_basic();
        }
    }
    return map;
}

RCP<const Basic> UPSeriesPiranha::get_coeff(int i) const
{
    return p_.find_cf({i}).get_basic();
}

p_expr UPSeriesPiranha::mul(const p_expr &s, const p_expr &r, unsigned prec)
{
    p_expr::set_auto_truncate_degree(prec - 1);
    p_expr ret = s * r;
    p_expr::unset_auto_truncate_degree();
    return ret;
}

p_expr UPSeriesPiranha::pow(const p_expr &s, int n, unsigned prec)
{
    p_expr::set_auto_truncate_degree(prec - 1);
    p_expr::clear_pow_cache();
    p_expr ret = s.pow(n);
    p_expr::unset_auto_truncate_degree();
    return ret;
}

unsigned UPSeriesPiranha::ldegree(const p_expr &s)
{
    return s.ldegree();
}

Expression UPSeriesPiranha::find_cf(const p_expr &s, const p_expr &var,
                                    unsigned deg)
{
    return s.find_cf({deg});
}

Expression UPSeriesPiranha::root(Expression &c, unsigned n)
{
    return SymEngine::pow(c, 1 / Expression(n));
}

p_expr UPSeriesPiranha::diff(const p_expr &s, const p_expr &var)
{
    return s.partial(var.get_symbol_set()[0].get_name());
}

p_expr UPSeriesPiranha::integrate(const p_expr &s, const p_expr &var)
{
    return s.integrate(var.get_symbol_set()[0].get_name());
}

p_expr UPSeriesPiranha::subs(const p_expr &s, const p_expr &var,
                             const p_expr &r, unsigned prec)
{
    p_expr::set_auto_truncate_degree(prec - 1);
    p_expr ret = s.subs(var.get_symbol_set()[0].get_name(), r);
    p_expr::unset_auto_truncate_degree();
    return ret;
}

Expression UPSeriesPiranha::sin(const Expression &c)
{
    return SymEngine::sin(c.get_basic());
}

Expression UPSeriesPiranha::cos(const Expression &c)
{
    return SymEngine::cos(c.get_basic());
}

Expression UPSeriesPiranha::tan(const Expression &c)
{
    return SymEngine::tan(c.get_basic());
}

Expression UPSeriesPiranha::asin(const Expression &c)
{
    return SymEngine::asin(c.get_basic());
}

Expression UPSeriesPiranha::acos(const Expression &c)
{
    return SymEngine::acos(c.get_basic());
}

Expression UPSeriesPiranha::atan(const Expression &c)
{
    return SymEngine::atan(c.get_basic());
}

Expression UPSeriesPiranha::sinh(const Expression &c)
{
    return SymEngine::sinh(c.get_basic());
}

Expression UPSeriesPiranha::cosh(const Expression &c)
{
    return SymEngine::cosh(c.get_basic());
}

Expression UPSeriesPiranha::tanh(const Expression &c)
{
    return SymEngine::tanh(c.get_basic());
}

Expression UPSeriesPiranha::asinh(const Expression &c)
{
    return SymEngine::asinh(c.get_basic());
}

Expression UPSeriesPiranha::atanh(const Expression &c)
{
    return SymEngine::atanh(c.get_basic());
}

Expression UPSeriesPiranha::exp(const Expression &c)
{
    return SymEngine::exp(c.get_basic());
}

Expression UPSeriesPiranha::log(const Expression &c)
{
    return SymEngine::log(c.get_basic());
}
}
