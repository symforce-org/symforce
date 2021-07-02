#include <symengine/parser/parser.h>
#include <symengine/parser/parser.tab.hh>
#include <symengine/real_double.h>
#include <symengine/real_mpfr.h>

namespace SymEngine
{

RCP<const Basic> parse(const std::string &s, bool convert_xor)
{
    // This is expensive:
    Parser p;
    // If you need to parse multiple strings, initialize Parser first, then
    // call Parser::parse() repeatedly.
    return p.parse(s, convert_xor);
}

RCP<const Basic> Parser::parse(const std::string &input, bool convert_xor)
{
    inp = input;
    if (convert_xor) {
        std::replace(inp.begin(), inp.end(), '^', '@');
    }
    m_tokenizer.set_string(inp);
    if (yyparse(*this) == 0)
        return this->res;
    throw ParseError("Parsing Unsuccessful");
}

// reference :
// http://stackoverflow.com/questions/30393285/stdfunction-fails-to-distinguish-overloaded-functions
typedef RCP<const Basic> (*single_arg_func)(const RCP<const Basic> &);
typedef RCP<const Basic> (*double_arg_func)(const RCP<const Basic> &,
                                            const RCP<const Basic> &);
typedef RCP<const Boolean> (*single_arg_boolean_func)(const RCP<const Basic> &);
typedef RCP<const Boolean> (*double_arg_boolean_func)(const RCP<const Basic> &,
                                                      const RCP<const Basic> &);

// cast overloaded functions below to single_arg, double_arg before they can
// be used in the map
single_arg_func single_casted_log = log;
single_arg_func single_casted_zeta = zeta;

double_arg_func double_casted_log = log;
double_arg_func double_casted_zeta = zeta;

single_arg_boolean_func single_casted_Eq = Eq;
double_arg_boolean_func double_casted_Eq = Eq;

RCP<const Basic> Parser::functionify(const std::string &name, vec_basic &params)
{
    const static std::map<const std::string,
                          const std::function<RCP<const Basic>(
                              const RCP<const Basic> &)>>
        single_arg_functions = {
            {"sin", sin},
            {"cos", cos},
            {"tan", tan},
            {"cot", cot},
            {"csc", csc},
            {"sec", sec},

            {"asin", asin},
            {"acos", acos},
            {"atan", atan},
            {"asec", asec},
            {"acsc", acsc},
            {"acot", acot},

            {"sinh", sinh},
            {"cosh", cosh},
            {"tanh", tanh},
            {"coth", coth},
            {"sech", sech},
            {"csch", csch},

            {"asinh", asinh},
            {"acosh", acosh},
            {"atanh", atanh},
            {"asech", asech},
            {"acoth", acoth},
            {"acsch", acsch},

            {"gamma", gamma},
            {"sqrt", sqrt},
            {"abs", abs},
            {"exp", exp},
            {"erf", erf},
            {"erfc", erfc},
            {"loggamma", loggamma},
            {"lambertw", lambertw},
            {"dirichlet_eta", dirichlet_eta},
            {"ln", single_casted_log},
            {"log", single_casted_log},
            {"zeta", single_casted_zeta},
        };
    const static std::map<const std::string,
                          const std::function<RCP<const Basic>(
                              const RCP<const Basic> &,
                              const RCP<const Basic> &)>>
        double_arg_functions = {
            {"pow", (double_arg_func)pow},
            {"beta", beta},
            {"log", double_casted_log},
            {"zeta", double_casted_zeta},
            {"lowergamma", lowergamma},
            {"uppergamma", uppergamma},
            {"polygamma", polygamma},
            {"kronecker_delta", kronecker_delta},
            {"atan2", atan2},
        };

    const static std::map<const std::string,
                          const std::function<RCP<const Basic>(vec_basic &)>>
        multi_arg_functions = {
            {"max", max}, {"min", min}, {"levi_civita", levi_civita},
        };

    const static std::map<const std::string,
                          const std::function<RCP<const Boolean>(
                              const RCP<const Basic> &)>>
        single_arg_boolean_functions = {
            {"Eq", single_casted_Eq},
        };
    const static std::map<const std::string,
                          const std::function<RCP<const Boolean>(
                              const RCP<const Boolean> &)>>
        single_arg_boolean_boolean_functions = {
            {"Not", logical_not},
        };

    const static std::map<const std::string,
                          const std::function<RCP<const Boolean>(
                              const RCP<const Basic> &,
                              const RCP<const Basic> &)>>
        double_arg_boolean_functions = {
            {"Eq", double_casted_Eq},
            {"Ne", Ne},
            {"Ge", Ge},
            {"Gt", Gt},
            {"Le", Le},
            {"Lt", Lt},
        };

    const static std::map<const std::string,
                          const std::function<RCP<const Boolean>(
                              vec_boolean &)>>
        multi_arg_vec_boolean_functions = {
            {"Xor", logical_xor}, {"Xnor", logical_xnor},
        };

    const static std::map<const std::string,
                          const std::function<RCP<const Boolean>(
                              set_boolean &)>>
        multi_arg_set_boolean_functions = {
            {"And", logical_and},
            {"Or", logical_or},
            {"Nand", logical_nand},
            {"Nor", logical_nor},
        };

    if (params.size() == 1) {
        auto it1 = single_arg_functions.find(name);
        if (it1 != single_arg_functions.end()) {
            return it1->second(params[0]);
        }
        auto it2 = single_arg_boolean_functions.find(name);
        if (it2 != single_arg_boolean_functions.end()) {
            return it2->second(params[0]);
        }
        auto it3 = single_arg_boolean_boolean_functions.find(name);
        if (it3 != single_arg_boolean_boolean_functions.end()) {
            return it3->second(rcp_static_cast<const Boolean>(params[0]));
        }
    }

    if (params.size() == 2) {
        auto it1 = double_arg_functions.find(name);
        if (it1 != double_arg_functions.end()) {
            return it1->second(params[0], params[1]);
        }
        auto it2 = double_arg_boolean_functions.find(name);
        if (it2 != double_arg_boolean_functions.end()) {
            return it2->second(params[0], params[1]);
        }
    }

    auto it1 = multi_arg_functions.find(name);
    if (it1 != multi_arg_functions.end()) {
        return it1->second(params);
    }

    auto it2 = multi_arg_vec_boolean_functions.find(name);
    if (it2 != multi_arg_vec_boolean_functions.end()) {
        vec_boolean p;
        for (auto &v : params) {
            p.push_back(rcp_static_cast<const Boolean>(v));
        }
        return it2->second(p);
    }

    auto it3 = multi_arg_set_boolean_functions.find(name);
    if (it3 != multi_arg_set_boolean_functions.end()) {
        set_boolean s;
        for (auto &v : params) {
            s.insert(rcp_static_cast<const Boolean>(v));
        }
        return it3->second(s);
    }

    return function_symbol(name, params);
}

RCP<const Basic> Parser::parse_identifier(const std::string &expr)
{
    const static std::map<const std::string, const RCP<const Basic>>
        parser_constants = {{"e", E},
                            {"E", E},
                            {"EulerGamma", EulerGamma},
                            {"Catalan", Catalan},
                            {"GoldenRatio", GoldenRatio},
                            {"pi", pi},
                            {"I", I},
                            {"oo", Inf},
                            {"inf", Inf},
                            {"zoo", ComplexInf},
                            {"nan", Nan}};

    auto c = parser_constants.find(expr);
    if (c != parser_constants.end()) {
        return c->second;
    } else {
        return symbol(expr);
    }
}

RCP<const Basic> Parser::parse_numeric(const std::string &expr)
{
    const char *startptr = expr.c_str();
    char *lendptr;
    // if the expr is numeric, it's either a float or an integer
    errno = 0;
    long l = std::strtol(startptr, &lendptr, 0);

    // Number is a long;
    if (expr.find_first_of('.') == std::string::npos
        && lendptr == startptr + expr.length()) {
        if (errno != ERANGE) {
            // No overflow in l
            return integer(l);
        } else {
            return integer(integer_class(expr));
        }
    } else {
#ifdef HAVE_SYMENGINE_MPFR
        unsigned digits = 0;
        for (size_t i = 0; i < expr.length(); ++i) {
            if (expr[i] == '.' or expr[i] == '-')
                continue;
            if (expr[i] == 'E' or expr[i] == 'e')
                break;
            if (digits != 0 or expr[i] != '0') {
                ++digits;
            }
        }
        if (digits <= 15) {
            char *endptr = 0;
            double d = std::strtod(startptr, &endptr);
            return real_double(d);
        } else {
            // mpmath.libmp.libmpf.dps_to_prec
            long prec = std::max(
                long(1), std::lround((digits + 1) * 3.3219280948873626));
            return real_mpfr(mpfr_class(expr, prec));
        }
#else
        char *endptr = 0;
        double d = std::strtod(startptr, &endptr);
        return real_double(d);
#endif
    }
}

std::tuple<RCP<const Basic>, RCP<const Basic>>
Parser::parse_implicit_mul(const std::string &expr)
{
    const char *startptr = expr.c_str();
    char *endptr = 0;
    std::strtod(startptr, &endptr);

    RCP<const Basic> num = one, sym;

    // Numerical part of the result of e.g. "100x";
    size_t length = endptr - startptr;
    std::string lexpr = std::string(startptr, length);
    num = parse_numeric(lexpr);

    // get the rest of the string
    lexpr = std::string(startptr + length, expr.length() - length);
    if (lexpr.length() == 0) {
        sym = one;
    } else {
        sym = parse_identifier(lexpr);
    }
    return std::make_tuple(num, sym);
}

} // namespace SymEngine
