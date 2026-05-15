#include <symengine/solve.h>
#include <symengine/polys/basic_conversions.h>
#include <symengine/logic.h>
#include <symengine/mul.h>
#include <symengine/as_real_imag.cpp>

namespace SymEngine
{

RCP<const Set> solve_poly_linear(const vec_basic &coeffs,
                                 const RCP<const Set> &domain)
{
    if (coeffs.size() != 2) {
        throw SymEngineException("Expected a polynomial of degree 1. Try with "
                                 "solve() or solve_poly()");
    }
    auto root = neg(div(coeffs[0], coeffs[1]));
    return set_intersection({domain, finiteset({root})});
}

RCP<const Set> solve_poly_quadratic(const vec_basic &coeffs,
                                    const RCP<const Set> &domain)
{
    if (coeffs.size() != 3) {
        throw SymEngineException("Expected a polynomial of degree 2. Try with "
                                 "solve() or solve_poly()");
    }

    auto a = coeffs[2];
    auto b = div(coeffs[1], a), c = div(coeffs[0], a);
    RCP<const Basic> root1, root2;
    if (eq(*c, *zero)) {
        root1 = neg(b);
        root2 = zero;
    } else if (eq(*b, *zero)) {
        root1 = sqrt(neg(c));
        root2 = neg(root1);
    } else {
        auto discriminant = sub(mul(b, b), mul(integer(4), c));
        auto lterm = div(neg(b), integer(2));
        auto rterm = div(sqrt(discriminant), integer(2));
        root1 = add(lterm, rterm);
        root2 = sub(lterm, rterm);
    }
    return set_intersection({domain, finiteset({root1, root2})});
}

RCP<const Set> solve_poly_cubic(const vec_basic &coeffs,
                                const RCP<const Set> &domain)
{
    if (coeffs.size() != 4) {
        throw SymEngineException("Expected a polynomial of degree 3. Try with "
                                 "solve() or solve_poly()");
    }

    auto a = coeffs[3];
    auto b = div(coeffs[2], a), c = div(coeffs[1], a), d = div(coeffs[0], a);

    // ref :
    // https://en.wikipedia.org/wiki/Cubic_function#General_solution_to_the_cubic_equation_with_real_coefficients
    auto i2 = integer(2), i3 = integer(3), i4 = integer(4), i9 = integer(9),
         i27 = integer(27);

    RCP<const Basic> root1, root2, root3;
    if (eq(*d, *zero)) {
        root1 = zero;
        auto fset = solve_poly_quadratic({c, b, one}, domain);
        SYMENGINE_ASSERT(is_a<FiniteSet>(*fset));
        auto cont = down_cast<const FiniteSet &>(*fset).get_container();
        if (cont.size() == 2) {
            root2 = *cont.begin();
            root3 = *std::next(cont.begin());
        } else {
            root2 = root3 = *cont.begin();
        }
    } else {
        auto delta0 = sub(mul(b, b), mul(i3, c));
        auto delta1
            = add(sub(mul(pow(b, i3), i2), mul({i9, b, c})), mul(i27, d));
        auto delta = div(sub(mul(i4, pow(delta0, i3)), pow(delta1, i2)), i27);
        if (eq(*delta, *zero)) {
            if (eq(*delta0, *zero)) {
                root1 = root2 = root3 = div(neg(b), i3);
            } else {
                root1 = root2
                    = div(sub(mul(i9, d), mul(b, c)), mul(i2, delta0));
                root3 = div(sub(mul({i4, b, c}), add(mul(d, i9), pow(b, i3))),
                            delta0);
            }
        } else {
            auto temp = sqrt(mul(neg(i27), delta));
            auto Cexpr = div(add(delta1, temp), i2);
            if (eq(*Cexpr, *zero)) {
                Cexpr = div(sub(delta1, temp), i2);
            }
            auto C = pow(Cexpr, div(one, i3));
            root1 = neg(div(add(b, add(C, div(delta0, C))), i3));
            auto coef = div(mul(I, sqrt(i3)), i2);
            temp = neg(div(one, i2));
            auto cbrt1 = add(temp, coef);
            auto cbrt2 = sub(temp, coef);
            root2 = neg(div(
                add(b, add(mul(cbrt1, C), div(delta0, mul(cbrt1, C)))), i3));
            root3 = neg(div(
                add(b, add(mul(cbrt2, C), div(delta0, mul(cbrt2, C)))), i3));
        }
    }
    return set_intersection({domain, finiteset({root1, root2, root3})});
}

RCP<const Set> solve_poly_quartic(const vec_basic &coeffs,
                                  const RCP<const Set> &domain)
{
    if (coeffs.size() != 5) {
        throw SymEngineException("Expected a polynomial of degree 4. Try with "
                                 "solve() or solve_poly()");
    }

    auto i2 = integer(2), i3 = integer(3), i4 = integer(4), i8 = integer(8),
         i16 = integer(16), i64 = integer(64), i256 = integer(256);

    // ref : http://mathforum.org/dr.math/faq/faq.cubic.equations.html
    auto lc = coeffs[4];
    auto a = div(coeffs[3], lc), b = div(coeffs[2], lc), c = div(coeffs[1], lc),
         d = div(coeffs[0], lc);
    set_basic roots;

    if (eq(*d, *zero)) {
        vec_basic newcoeffs(4);
        newcoeffs[0] = c, newcoeffs[1] = b, newcoeffs[2] = a,
        newcoeffs[3] = one;
        auto rcubic = solve_poly_cubic(newcoeffs, domain);
        SYMENGINE_ASSERT(is_a<FiniteSet>(*rcubic));
        roots = down_cast<const FiniteSet &>(*rcubic).get_container();
        roots.insert(zero);
    } else {
        // substitute x = y-a/4 to get equation of the form y**4 + e*y**2 + f*y
        // + g = 0
        auto sqa = mul(a, a);
        auto cba = mul(sqa, a);
        auto aby4 = div(a, i4);
        auto e = sub(b, div(mul(i3, sqa), i8));
        auto ff = sub(add(c, div(cba, i8)), div(mul(a, b), i2));
        auto g = sub(add(d, div(mul(sqa, b), i16)),
                     add(div(mul(a, c), i4), div(mul({i3, cba, a}), i256)));

        // two special cases
        if (eq(*g, *zero)) {
            vec_basic newcoeffs(4);
            newcoeffs[0] = ff, newcoeffs[1] = e, newcoeffs[2] = zero,
            newcoeffs[3] = one;
            auto rcubic = solve_poly_cubic(newcoeffs, domain);
            SYMENGINE_ASSERT(is_a<FiniteSet>(*rcubic));
            auto rtemp = down_cast<const FiniteSet &>(*rcubic).get_container();
            SYMENGINE_ASSERT(rtemp.size() > 0 and rtemp.size() <= 3);
            for (auto &r : rtemp) {
                roots.insert(sub(r, aby4));
            }
            roots.insert(neg(aby4));
        } else if (eq(*ff, *zero)) {
            vec_basic newcoeffs(3);
            newcoeffs[0] = g, newcoeffs[1] = e, newcoeffs[2] = one;
            auto rquad = solve_poly_quadratic(newcoeffs, domain);
            SYMENGINE_ASSERT(is_a<FiniteSet>(*rquad));
            auto rtemp = down_cast<const FiniteSet &>(*rquad).get_container();
            SYMENGINE_ASSERT(rtemp.size() > 0 and rtemp.size() <= 2);
            for (auto &r : rtemp) {
                auto sqrtr = sqrt(r);
                roots.insert(sub(sqrtr, aby4));
                roots.insert(sub(neg(sqrtr), aby4));
            }
        } else {
            // Leonhard Euler's method
            vec_basic newcoeffs(4);
            newcoeffs[0] = neg(div(mul(ff, ff), i64)),
            newcoeffs[1] = div(sub(mul(e, e), mul(i4, g)), i16),
            newcoeffs[2] = div(e, i2);
            newcoeffs[3] = one;

            auto rcubic = solve_poly_cubic(newcoeffs);
            SYMENGINE_ASSERT(is_a<FiniteSet>(*rcubic));
            roots = down_cast<const FiniteSet &>(*rcubic).get_container();
            SYMENGINE_ASSERT(roots.size() > 0 and roots.size() <= 3);
            auto p = sqrt(*roots.begin());
            auto q = p;
            if (roots.size() > 1) {
                q = sqrt(*std::next(roots.begin()));
            }
            auto r = div(neg(ff), mul({i8, p, q}));
            roots.clear();
            roots.insert(add({p, q, r, neg(aby4)}));
            roots.insert(add({p, neg(q), neg(r), neg(aby4)}));
            roots.insert(add({neg(p), q, neg(r), neg(aby4)}));
            roots.insert(add({neg(p), neg(q), r, neg(aby4)}));
        }
    }
    return set_intersection({domain, finiteset(roots)});
}

RCP<const Set> solve_poly_heuristics(const vec_basic &coeffs,
                                     const RCP<const Set> &domain)
{
    auto degree = coeffs.size() - 1;
    switch (degree) {
        case 0: {
            if (eq(*coeffs[0], *zero)) {
                return domain;
            } else {
                return emptyset();
            }
        }
        case 1:
            return solve_poly_linear(coeffs, domain);
        case 2:
            return solve_poly_quadratic(coeffs, domain);
        case 3:
            return solve_poly_cubic(coeffs, domain);
        case 4:
            return solve_poly_quartic(coeffs, domain);
        default:
            throw SymEngineException(
                "expected a polynomial of order between 0 to 4");
    }
}

inline RCP<const Basic> get_coeff_basic(const integer_class &i)
{
    return integer(i);
}

inline RCP<const Basic> get_coeff_basic(const Expression &i)
{
    return i.get_basic();
}

template <typename Poly>
inline vec_basic extract_coeffs(const RCP<const Poly> &f)
{
    int degree = f->get_degree();
    vec_basic coeffs;
    for (int i = 0; i <= degree; i++)
        coeffs.push_back(get_coeff_basic(f->get_coeff(i)));
    return coeffs;
}

RCP<const Set> solve_poly(const RCP<const Basic> &f,
                          const RCP<const Symbol> &sym,
                          const RCP<const Set> &domain)
{

#if defined(HAVE_SYMENGINE_FLINT) && __FLINT_RELEASE > 20502
    try {
        auto poly = from_basic<UIntPolyFlint>(f, sym);
        auto fac = factors(*poly);
        set_set solns;
        for (const auto &elem : fac) {
            auto uip = UIntPoly::from_poly(*elem.first);
            auto degree = uip->get_poly().degree();
            if (degree <= 4) {
                solns.insert(
                    solve_poly_heuristics(extract_coeffs(uip), domain));
            } else {
                solns.insert(
                    conditionset(sym, logical_and({Eq(uip->as_symbolic(), zero),
                                                   domain->contains(sym)})));
            }
        }
        return SymEngine::set_union(solns);
    } catch (SymEngineException &x) {
        // Try next
    }
#endif
    RCP<const Basic> gen = rcp_static_cast<const Basic>(sym);
    auto uexp = from_basic<UExprPoly>(f, gen);
    auto degree = uexp->get_degree();
    if (degree <= 4) {
        return solve_poly_heuristics(extract_coeffs(uexp), domain);
    } else {
        return conditionset(sym,
                            logical_and({Eq(f, zero), domain->contains(sym)}));
    }
}

RCP<const Set> solve_rational(const RCP<const Basic> &f,
                              const RCP<const Symbol> &sym,
                              const RCP<const Set> &domain)
{
    RCP<const Basic> num, den;
    as_numer_denom(f, outArg(num), outArg(den));
    if (has_symbol(*den, *sym)) {
        auto numsoln = solve(num, sym, domain);
        auto densoln = solve(den, sym, domain);
        return set_complement(numsoln, densoln);
    }
    return solve_poly(num, sym, domain);
}

/* Helper Visitors for solve_trig */

class IsALinearArgTrigVisitor
    : public BaseVisitor<IsALinearArgTrigVisitor, LocalStopVisitor>
{
protected:
    Ptr<const Symbol> x_;
    bool is_;

public:
    IsALinearArgTrigVisitor(Ptr<const Symbol> x) : x_(x) {}

    bool apply(const Basic &b)
    {
        stop_ = false;
        is_ = true;
        preorder_traversal_local_stop(b, *this);
        return is_;
    }

    bool apply(const RCP<const Basic> &b)
    {
        return apply(*b);
    }

    void bvisit(const Basic &x)
    {
        local_stop_ = false;
    }

    void bvisit(const Symbol &x)
    {
        if (x_->__eq__(x)) {
            is_ = 0;
            stop_ = true;
        }
    }

    template <typename T,
              typename
              = enable_if_t<std::is_base_of<TrigFunction, T>::value
                            or std::is_base_of<HyperbolicFunction, T>::value>>
    void bvisit(const T &x)
    {
        is_ = (from_basic<UExprPoly>(x.get_args()[0], (*x_).rcp_from_this())
                   ->get_degree()
               <= 1);
        if (not is_)
            stop_ = true;
        local_stop_ = true;
    }
};

bool is_a_LinearArgTrigEquation(const Basic &b, const Symbol &x)
{
    IsALinearArgTrigVisitor v(ptrFromRef(x));
    return v.apply(b);
}

class InvertComplexVisitor : public BaseVisitor<InvertComplexVisitor>
{
protected:
    RCP<const Set> result_;
    RCP<const Set> gY_;
    RCP<const Dummy> nD_;
    RCP<const Symbol> sym_;
    RCP<const Set> domain_;

public:
    InvertComplexVisitor(RCP<const Set> gY, RCP<const Dummy> nD,
                         RCP<const Symbol> sym, RCP<const Set> domain)
        : gY_(gY), nD_(nD), sym_(sym), domain_(domain)
    {
    }

    void bvisit(const Basic &x)
    {
        result_ = gY_;
    }

    void bvisit(const Add &x)
    {
        vec_basic f1X, f2X;
        for (auto &elem : x.get_args()) {
            if (has_symbol(*elem, *sym_)) {
                f1X.push_back(elem);
            } else {
                f2X.push_back(elem);
            }
        }
        auto depX = add(f1X), indepX = add(f2X);
        if (not eq(*indepX, *zero)) {
            gY_ = imageset(nD_, sub(nD_, indepX), gY_);
            result_ = apply(*depX);
        } else {
            result_ = gY_;
        }
    }

    void bvisit(const Mul &x)
    {
        vec_basic f1X, f2X;
        for (auto &elem : x.get_args()) {
            if (has_symbol(*elem, *sym_)) {
                f1X.push_back(elem);
            } else {
                f2X.push_back(elem);
            }
        }
        auto depX = mul(f1X), indepX = mul(f2X);
        if (not eq(*indepX, *one)) {
            if (eq(*indepX, *NegInf) or eq(*indepX, *Inf)
                or eq(*indepX, *ComplexInf)) {
                result_ = emptyset();
            } else {
                gY_ = imageset(nD_, div(nD_, indepX), gY_);
                result_ = apply(*depX);
            }
        } else {
            result_ = gY_;
        }
    }

    void bvisit(const Pow &x)
    {
        if (eq(*x.get_base(), *E) and is_a<FiniteSet>(*gY_)) {
            set_set inv;
            for (const auto &elem :
                 down_cast<const FiniteSet &>(*gY_).get_container()) {
                if (eq(*elem, *zero))
                    continue;
                RCP<const Basic> re, im;
                as_real_imag(elem, outArg(re), outArg(im));
                auto logabs = log(add(mul(re, re), mul(im, im)));
                auto logarg = atan2(im, re);
                inv.insert(imageset(
                    nD_,
                    add(mul(add(mul({integer(2), nD_, pi}), logarg), I),
                        div(logabs, integer(2))),
                    interval(NegInf, Inf, true,
                             true))); // TODO : replace interval(-oo,oo) with
                // Set of Integers once Class for Range is implemented.
            }
            gY_ = set_union(inv);
            apply(*x.get_exp());
            return;
        }
        result_ = gY_;
    }

    RCP<const Set> apply(const Basic &b)
    {
        result_ = gY_;
        b.accept(*this);
        return set_intersection({domain_, result_});
    }
};

RCP<const Set> invertComplex(const RCP<const Basic> &fX,
                             const RCP<const Set> &gY,
                             const RCP<const Symbol> &sym,
                             const RCP<const Dummy> &nD,
                             const RCP<const Set> &domain)
{
    InvertComplexVisitor v(gY, nD, sym, domain);
    return v.apply(*fX);
}

RCP<const Set> solve_trig(const RCP<const Basic> &f,
                          const RCP<const Symbol> &sym,
                          const RCP<const Set> &domain)
{
    // TODO : first simplify f using `fu`.
    auto exp_f = rewrite_as_exp(f);
    RCP<const Basic> num, den;
    as_numer_denom(exp_f, outArg(num), outArg(den));

    auto xD = dummy("x");
    map_basic_basic d;
    auto temp = exp(mul(I, sym));
    d[temp] = xD;
    num = expand(num), den = expand(den);
    num = num->subs(d);
    den = den->subs(d);

    if (has_symbol(*num, *sym) or has_symbol(*den, *sym)) {
        return conditionset(sym, logical_and({Eq(f, zero)}));
    }

    auto soln = set_complement(solve(num, xD), solve(den, xD));
    if (eq(*soln, *emptyset()))
        return emptyset();
    else if (is_a<FiniteSet>(*soln)) {
        set_set res;
        auto nD
            = dummy("n"); // use the same dummy for finding every solution set.
        auto n = symbol(
            "n"); // replaces the above dummy in final set of solutions.
        map_basic_basic d;
        d[nD] = n;
        for (const auto &elem :
             down_cast<const FiniteSet &>(*soln).get_container()) {
            res.insert(
                invertComplex(exp(mul(I, sym)), finiteset({elem}), sym, nD));
        }
        auto ans = set_union(res)->subs(d);
        if (not is_a_Set(*ans))
            throw SymEngineException("Expected an object of type Set");
        return set_intersection({rcp_static_cast<const Set>(ans), domain});
    }
    return conditionset(sym, logical_and({Eq(f, zero), domain->contains(sym)}));
}

RCP<const Set> solve(const RCP<const Basic> &f, const RCP<const Symbol> &sym,
                     const RCP<const Set> &domain)
{
    if (eq(*f, *boolTrue))
        return domain;
    if (eq(*f, *boolFalse))
        return emptyset();
    if (is_a<Equality>(*f)) {
        return solve(sub(down_cast<const Relational &>(*f).get_arg1(),
                         down_cast<const Relational &>(*f).get_arg2()),
                     sym, domain);
    } else if (is_a<Unequality>(*f)) {
        auto soln = solve(sub(down_cast<const Relational &>(*f).get_arg1(),
                              down_cast<const Relational &>(*f).get_arg2()),
                          sym, domain);
        return set_complement(domain, soln);
    } else if (is_a_Relational(*f)) {
        // Solving inequalities is not implemented yet.
        return conditionset(sym, logical_and({rcp_static_cast<const Boolean>(f),
                                              domain->contains(sym)}));
    }

    if (is_a_Number(*f)) {
        if (eq(*f, *zero)) {
            return domain;
        } else {
            return emptyset();
        }
    }

    if (not has_symbol(*f, *sym))
        return emptyset();

    if (is_a_LinearArgTrigEquation(*f, *sym)) {
        return solve_trig(f, sym, domain);
    }

    if (is_a<Mul>(*f)) {
        auto args = f->get_args();
        set_set solns;
        for (auto &a : args) {
            solns.insert(solve(a, sym, domain));
        }
        return SymEngine::set_union(solns);
    }

    return solve_rational(f, sym, domain);
}

vec_basic linsolve_helper(const DenseMatrix &A, const DenseMatrix &b)
{
    DenseMatrix res(A.nrows(), 1);
    fraction_free_gauss_jordan_solve(A, b, res);
    vec_basic fs;
    for (unsigned i = 0; i < res.nrows(); i++) {
        fs.push_back(res.get(i, 0));
    }
    return fs;
}

vec_basic linsolve(const DenseMatrix &system, const vec_sym &syms)
{
    DenseMatrix A(system.nrows(), system.ncols() - 1), b(system.nrows(), 1);
    system.submatrix(A, 0, 0, system.nrows() - 1, system.ncols() - 2);
    system.submatrix(b, 0, system.ncols() - 1, system.nrows() - 1,
                     system.ncols() - 1);
    return linsolve_helper(A, b);
}

vec_basic linsolve(const vec_basic &system, const vec_sym &syms)
{
    auto mat = linear_eqns_to_matrix(system, syms);
    DenseMatrix A = mat.first, b = mat.second;
    return linsolve_helper(A, b);
}

set_basic get_set_from_vec(const vec_sym &syms)
{
    set_basic sb;
    for (auto &s : syms)
        sb.insert(s);
    return sb;
}

std::pair<DenseMatrix, DenseMatrix>
linear_eqns_to_matrix(const vec_basic &equations, const vec_sym &syms)
{
    auto size = numeric_cast<unsigned int>(syms.size());
    DenseMatrix A(numeric_cast<unsigned int>(equations.size()), size);
    zeros(A);
    vec_basic bvec;

    int row = 0;
    auto gens = get_set_from_vec(syms);
    umap_basic_uint index_of_sym;
    for (unsigned int i = 0; i < size; i++) {
        index_of_sym[syms[i]] = i;
    }
    for (const auto &eqn : equations) {
        auto neqn = eqn;
        if (is_a<Equality>(*eqn)) {
            neqn = sub(down_cast<const Equality &>(*eqn).get_arg2(),
                       down_cast<const Equality &>(*eqn).get_arg1());
        }

        auto mpoly = from_basic<MExprPoly>(neqn, gens);
        RCP<const Basic> rem = zero;
        for (const auto &p : mpoly->get_poly().dict_) {
            RCP<const Basic> res = (p.second.get_basic());
            int whichvar = 0, non_zero = 0;
            RCP<const Basic> cursim;
            for (auto &sym : gens) {
                if (0 != p.first[whichvar]) {
                    non_zero++;
                    cursim = sym;
                    if (p.first[whichvar] != 1 or non_zero == 2) {
                        throw SymEngineException("Expected a linear equation.");
                    }
                }
                whichvar++;
            }
            if (not non_zero) {
                rem = res;
            } else {
                A.set(row, index_of_sym[cursim], res);
            }
        }
        bvec.push_back(neg(rem));
        ++row;
    }
    return std::make_pair(
        A, DenseMatrix(numeric_cast<unsigned int>(equations.size()), 1, bvec));
}
} // namespace SymEngine
