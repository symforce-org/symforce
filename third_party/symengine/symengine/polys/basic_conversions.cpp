#include <symengine/visitor.h>
#include <symengine/polys/basic_conversions.h>

namespace SymEngine
{
// all throughout Number refers to either a Rational or an Integer only
umap_basic_num _find_gens_poly_pow(const RCP<const Basic> &x,
                                   const RCP<const Basic> &base);

class PolyGeneratorVisitor : public BaseVisitor<PolyGeneratorVisitor>
{
private:
    // the generators are pow(it.first, it.second)
    // those which are not Pow are stored as (x, one)
    // numbers must always be positive and of the form (1/d, d belongs to N)
    umap_basic_num gen_set;

public:
    umap_basic_num apply(const Basic &b)
    {
        b.accept(*this);
        return std::move(gen_set);
    }

    // adds curr to gen_set, or updates already existing gen
    void add_to_gen_set(const RCP<const Basic> &base,
                        const RCP<const Number> &exp)
    {
        auto it = gen_set.find(base);
        if (it == gen_set.end()) {
            gen_set[base] = exp;
            return;
        }

        if (is_a<const Rational>(*exp)) {
            RCP<const Integer> den
                = down_cast<const Rational &>(*exp).get_den();
            if (is_a<const Rational>(*it->second))
                gen_set[base] = divnum(
                    one,
                    lcm(*den,
                        *down_cast<const Rational &>(*it->second).get_den()));
            else
                gen_set[base] = divnum(one, den);
        }
    }

    void bvisit(const Pow &x)
    {
        if (is_a<const Integer>(*x.get_exp())) {
            if (down_cast<const Integer &>(*x.get_exp()).is_positive()) {
                x.get_base()->accept(*this);
            } else {
                add_to_gen_set(pow(x.get_base(), minus_one), one);
            }

        } else if (is_a<const Rational>(*x.get_exp())) {

            RCP<const Basic> base = x.get_base();
            RCP<const Rational> r
                = rcp_static_cast<const Rational>(x.get_exp());
            if (r->is_negative())
                base = pow(base, minus_one);
            add_to_gen_set(base, divnum(one, r->get_den()));

        } else {
            umap_basic_num pow_pairs
                = _find_gens_poly_pow(x.get_exp(), x.get_base());
            for (auto it : pow_pairs)
                add_to_gen_set(pow(x.get_base(), it.first), it.second);
        }
    }

    void bvisit(const Add &x)
    {
        for (auto it : x.get_dict())
            it.first->accept(*this);
    }

    void bvisit(const Mul &x)
    {
        for (auto it : x.get_dict())
            it.first->accept(*this);
    }

    void bvisit(const Number &x)
    {
        // intentionally blank
    }

    void bvisit(const Basic &x)
    {
        add_to_gen_set(x.rcp_from_this(), one);
    }
};

class PolyGeneratorVisitorPow : public BaseVisitor<PolyGeneratorVisitorPow>
{
private:
    // the generators are mul(it.first, it.second) not Pow
    // the_base is the base of the Pow (whose exp we are currently dealing)
    // numbers must always be positive and of the form (1/d, d belongs to N)
    umap_basic_num gen_set;
    RCP<const Basic> the_base;

public:
    umap_basic_num apply(const Basic &b, const RCP<const Basic> &base)
    {
        the_base = base;
        b.accept(*this);
        return std::move(gen_set);
    }

    void bvisit(const Add &x)
    {
        if (not x.get_coef()->is_zero())
            x.get_coef()->accept(*this);

        for (auto it : x.get_dict()) {
            RCP<const Number> mulx = one, divx = one;

            if (it.second->is_negative())
                mulx = minus_one;

            if (is_a<const Rational>(*it.second))
                divx = down_cast<const Rational &>(*it.second).get_den();

            gen_set[mul(mulx, it.first)] = divnum(one, divx);
        }
    }

    void bvisit(const Mul &x)
    {
        // won't handle cases like 2**((x+1)(x+2))
        // needs `expand` to have been called
        RCP<const Number> mulx = one, divx = one;

        if (x.get_coef()->is_negative())
            mulx = minus_one;

        if (is_a<const Rational>(*x.get_coef()))
            divx = down_cast<const Rational &>(*x.get_coef()).get_den();

        auto dict = x.get_dict();
        gen_set[Mul::from_dict(mulx, std::move(dict))] = divnum(one, divx);
    }

    void bvisit(const Number &x)
    {
        if (not is_a_Number(*pow(the_base, x.rcp_from_this()))) {
            if (x.is_positive())
                gen_set[one] = x.rcp_from_this_cast<const Number>();
            else
                gen_set[minus_one]
                    = mulnum(x.rcp_from_this_cast<const Number>(), minus_one);
        }
    }

    void bvisit(const Basic &x)
    {
        gen_set[x.rcp_from_this()] = one;
    }
};

umap_basic_num _find_gens_poly(const RCP<const Basic> &x)
{
    PolyGeneratorVisitor v;
    return v.apply(*x);
}

umap_basic_num _find_gens_poly_pow(const RCP<const Basic> &x,
                                   const RCP<const Basic> &base)
{
    PolyGeneratorVisitorPow v;
    return v.apply(*x, base);
}

} // namespace SymEngine
