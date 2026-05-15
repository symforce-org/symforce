#include <symengine/visitor.h>

namespace SymEngine
{

UExprPoly::UExprPoly(const RCP<const Basic> &var, UExprDict &&dict)
    : USymEnginePoly(
        var, std::move(dict)){SYMENGINE_ASSIGN_TYPEID()
                                  SYMENGINE_ASSERT(is_canonical(get_poly()))}

      hash_t UExprPoly::__hash__() const
{
    hash_t seed = SYMENGINE_UEXPRPOLY;

    seed += get_var()->hash();
    for (const auto &it : get_poly().dict_) {
        hash_t temp = SYMENGINE_UEXPRPOLY;
        hash_combine<unsigned int>(temp, it.first);
        hash_combine<Basic>(temp, *(it.second.get_basic()));
        seed += temp;
    }
    return seed;
}

Expression UExprPoly::max_coef() const
{
    Expression curr = get_poly().get_dict().begin()->second;
    for (const auto &it : get_poly().get_dict())
        if (curr.get_basic()->__cmp__(*it.second.get_basic()))
            curr = it.second;
    return curr;
}

Expression UExprPoly::eval(const Expression &x) const
{
    Expression ans = 0;
    for (const auto &p : get_poly().get_dict()) {
        Expression temp;
        temp = pow(x, Expression(p.first));
        ans += p.second * temp;
    }
    return ans;
}

bool UExprPoly::is_zero() const
{
    return get_poly().empty();
}

bool UExprPoly::is_one() const
{
    return get_poly().size() == 1 and get_poly().get_dict().begin()->second == 1
           and get_poly().get_dict().begin()->first == 0;
}

bool UExprPoly::is_minus_one() const
{
    return get_poly().size() == 1
           and get_poly().get_dict().begin()->second == -1
           and get_poly().get_dict().begin()->first == 0;
}

bool UExprPoly::is_integer() const
{
    if (get_poly().empty())
        return true;
    return get_poly().size() == 1 and get_poly().get_dict().begin()->first == 0;
}

bool UExprPoly::is_symbol() const
{
    return get_poly().size() == 1 and get_poly().get_dict().begin()->first == 1
           and get_poly().get_dict().begin()->second == 1;
}

bool UExprPoly::is_mul() const
{
    return get_poly().size() == 1 and get_poly().get_dict().begin()->first != 0
           and get_poly().get_dict().begin()->second != 1
           and get_poly().get_dict().begin()->second != 0;
}

bool UExprPoly::is_pow() const
{
    return get_poly().size() == 1 and get_poly().get_dict().begin()->second == 1
           and get_poly().get_dict().begin()->first != 1
           and get_poly().get_dict().begin()->first != 0;
}

} // namespace SymEngine
