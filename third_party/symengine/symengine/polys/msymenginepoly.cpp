#include <symengine/polys/msymenginepoly.h>

namespace SymEngine
{

RCP<const Basic> MIntPoly::as_symbolic() const
{
    vec_basic args;
    for (const auto &p : get_poly().dict_) {
        RCP<const Basic> res = integer(p.second);
        int whichvar = 0;
        for (auto sym : get_vars()) {
            if (0 != p.first[whichvar])
                res = SymEngine::mul(res, pow(sym, integer(p.first[whichvar])));
            whichvar++;
        }
        args.push_back(res);
    }
    return SymEngine::add(args);
}

hash_t MIntPoly::__hash__() const
{
    hash_t seed = SYMENGINE_MINTPOLY;
    for (auto var : get_vars())
        hash_combine<std::string>(seed, var->__str__());

    for (auto &p : get_poly().dict_) {
        hash_t t = vec_hash<vec_uint>()(p.first);
        hash_combine<hash_t>(t, mp_get_si(p.second));
        seed ^= t;
    }
    return seed;
}

integer_class MIntPoly::eval(
    std::map<RCP<const Basic>, integer_class, RCPBasicKeyLess> &vals) const
{
    // TODO : handle missing values
    integer_class ans(0), temp, term;
    for (auto bucket : get_poly().dict_) {
        term = bucket.second;
        unsigned int whichvar = 0;
        for (auto sym : get_vars()) {
            mp_pow_ui(temp, vals.find(sym)->second, bucket.first[whichvar]);
            term *= temp;
            whichvar++;
        }
        ans += term;
    }
    return ans;
}

RCP<const Basic> MExprPoly::as_symbolic() const
{
    vec_basic args;
    for (const auto &p : get_poly().dict_) {
        RCP<const Basic> res = (p.second.get_basic());
        int whichvar = 0;
        for (auto sym : get_vars()) {
            if (0 != p.first[whichvar])
                res = SymEngine::mul(res, pow(sym, integer(p.first[whichvar])));
            whichvar++;
        }
        args.push_back(res);
    }
    return SymEngine::add(args);
}

hash_t MExprPoly::__hash__() const
{
    hash_t seed = SYMENGINE_MEXPRPOLY;
    for (auto var : get_vars())
        hash_combine<std::string>(seed, var->__str__());

    for (auto &p : get_poly().dict_) {
        hash_t t = vec_hash<vec_int>()(p.first);
        hash_combine<Basic>(t, *(p.second.get_basic()));
        seed ^= t;
    }
    return seed;
}

Expression MExprPoly::eval(
    std::map<RCP<const Basic>, Expression, RCPBasicKeyLess> &vals) const
{
    // TODO : handle missing values
    Expression ans(0);
    for (auto bucket : get_poly().dict_) {
        Expression term = bucket.second;
        unsigned int whichvar = 0;
        for (auto sym : get_vars()) {
            term *= pow(vals.find(sym)->second, bucket.first[whichvar]);
            whichvar++;
        }
        ans += term;
    }
    return ans;
}

unsigned int reconcile(vec_uint &v1, vec_uint &v2, set_basic &s,
                       const set_basic &s1, const set_basic &s2)
{
    auto i = s1.begin();
    auto j = s2.begin();
    unsigned int pos = 0;

    // Performs a merge of s1 and s2, and builds up v1 and v2 as translators
    // v1[i] and v2[i] is the position of the ith symbol in the new set

    // set union
    s = s1;
    s.insert(s2.begin(), s2.end());

    for (auto &it : s) {
        if (i != s1.end() and eq(*it, **i)) {
            v1.push_back(pos);
            i++;
        }
        if (j != s2.end() and eq(*it, **j)) {
            v2.push_back(pos);
            j++;
        }
        pos++;
    }
    // return size of the new symbol set
    return pos;
}

} // namespace SymEngine
