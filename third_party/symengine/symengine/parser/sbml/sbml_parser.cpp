#include <symengine/parser.h>
#include <symengine/parser/sbml/sbml_parser.h>
#include <symengine/parser/sbml/sbml_parser.tab.hh>
#include <symengine/parser/sbml/sbml_tokenizer.h>
#include <symengine/real_double.h>
#include <symengine/real_mpfr.h>
#include <symengine/functions.h>
#include <algorithm>
#include <cctype>

using SymEngine::ceiling;
using SymEngine::floor;
using SymEngine::gamma;
using SymEngine::logical_xor;
using SymEngine::Lt;
using SymEngine::piecewise;
using SymEngine::pow;
using SymEngine::zero;

namespace SymEngine
{

RCP<const Basic>
parse_sbml(const std::string &s,
           const std::map<const std::string, const RCP<const Basic>> &constants)
{
    SbmlParser p(constants);
    return p.parse(s);
}

RCP<const Basic> SbmlParser::parse(const std::string &input)
{
    inp = input;
    m_tokenizer->set_string(inp);
    sbml::parser p(*this);
    if (p() == 0)
        return this->res;
    throw ParseError("Parsing Unsuccessful");
}

static std::string lowercase(const std::string &str)
{
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return lower;
}

// reference :
// http://stackoverflow.com/questions/30393285/stdfunction-fails-to-distinguish-overloaded-functions
typedef RCP<const Basic> (*single_arg_func)(const RCP<const Basic> &);
typedef RCP<const Boolean> (*double_arg_boolean_func)(const RCP<const Basic> &,
                                                      const RCP<const Basic> &);
typedef RCP<const Basic> (*vector_arg_func)(const vec_basic &);

static RCP<const Basic> sqr(const RCP<const Basic> &x)
{
    return pow(x, integer(2));
}

static RCP<const Basic> sbml_log(const RCP<const Basic> &b,
                                 const RCP<const Basic> &x)
{
    return log(x, b);
}

static RCP<const Basic> log10(const RCP<const Basic> &x)
{
    return log(x, integer(10));
}

static RCP<const Basic> root(const RCP<const Basic> &b,
                             const RCP<const Basic> &x)
{
    return pow(x, div(integer(1), b));
}

static RCP<const Basic> fact(const RCP<const Basic> &x)
{
    return gamma(add(x, integer(1)));
}

static const std::map<const std::string, const std::function<RCP<const Basic>(
                                             const RCP<const Basic> &)>> &
init_sbml_parser_single_arg_functions()
{
    static const std::map<
        const std::string,
        const std::function<RCP<const Basic>(const RCP<const Basic> &)>>
        functions = {{"sin", sin},
                     {"cos", cos},
                     {"tan", tan},
                     {"cot", cot},
                     {"csc", csc},
                     {"sec", sec},

                     {"asin", asin},
                     {"arcsin", asin},
                     {"acos", acos},
                     {"arccos", acos},
                     {"atan", atan},
                     {"arctan", atan},
                     {"asec", asec},
                     {"arcsec", asec},
                     {"acsc", acsc},
                     {"arccsc", acsc},
                     {"acot", acot},
                     {"arccot", acot},

                     {"sinh", sinh},
                     {"cosh", cosh},
                     {"tanh", tanh},
                     {"coth", coth},
                     {"sech", sech},
                     {"csch", csch},

                     {"asinh", asinh},
                     {"arcsinh", asinh},
                     {"acosh", acosh},
                     {"arccosh", acosh},
                     {"atanh", atanh},
                     {"arctanh", atanh},
                     {"asech", asech},
                     {"arcsech", asech},
                     {"acoth", acoth},
                     {"arccoth", acoth},
                     {"acsch", acsch},
                     {"arccsch", acsch},

                     {"sqrt", sqrt},
                     {"abs", abs},
                     {"exp", exp},
                     {"floor", floor},
                     {"ceil", ceiling},
                     {"ceiling", ceiling},
                     {"ln", (single_arg_func)log},
                     {"log", log10},
                     {"log10", log10},
                     {"factorial", fact},
                     {"root", sqrt},
                     {"sqr", sqr}};
    return functions;
}

static RCP<const Boolean> vec_ge(const vec_basic &x)
{
    set_boolean b;
    for (std::size_t i = 0; i + 1 < x.size(); ++i) {
        b.insert(Ge(x[i], x[i + 1]));
    }
    return logical_and(b);
}

static RCP<const Boolean> vec_gt(const vec_basic &x)
{
    set_boolean b;
    for (std::size_t i = 0; i + 1 < x.size(); ++i) {
        b.insert(Gt(x[i], x[i + 1]));
    }
    return logical_and(b);
}

static RCP<const Boolean> vec_le(const vec_basic &x)
{
    set_boolean b;
    for (std::size_t i = 0; i + 1 < x.size(); ++i) {
        b.insert(Le(x[i], x[i + 1]));
    }
    return logical_and(b);
}

static RCP<const Boolean> vec_lt(const vec_basic &x)
{
    set_boolean b;
    for (std::size_t i = 0; i + 1 < x.size(); ++i) {
        b.insert(Lt(x[i], x[i + 1]));
    }
    return logical_and(b);
}

RCP<const Basic> SbmlParser::modulo(const RCP<const Basic> &a,
                                    const RCP<const Basic> &b)
{
    return piecewise({{sub(a, mul(b, ceiling(div(a, b)))),
                       logical_xor({Lt(a, zero), Lt(b, zero)})},
                      {sub(a, mul(b, floor(div(a, b)))), boolTrue}});
}

RCP<const Basic> SbmlParser::functionify(const std::string &name)
{
    const static std::map<const std::string, const RCP<const Basic>>
        zero_arg_functions = {{"plus", integer(0)},
                              {"times", integer(1)},
                              {"and", boolTrue},
                              {"or", boolFalse},
                              {"xor", boolFalse}};
    std::string lname = lowercase(name);
    auto l = zero_arg_functions.find(lname);
    if (l != zero_arg_functions.end()) {
        return l->second;
    }
    return function_symbol(name, vec_basic{});
}

RCP<const Basic> SbmlParser::functionify(const std::string &name,
                                         vec_basic &params)
{
    const static std::map<
        const std::string,
        const std::function<RCP<const Basic>(const RCP<const Basic> &,
                                             const RCP<const Basic> &)>>
        double_arg_functions = {
            {"minus", sub}, {"divide", div}, {"pow", pow},
            {"power", pow}, {"root", root},  {"log", sbml_log},
        };

    const static std::map<const std::string,
                          const std::function<RCP<const Basic>(vec_basic &)>>
        multi_arg_functions = {
            {"max", max},
            {"min", min},
            {"plus", (vector_arg_func)add},
            {"times", (vector_arg_func)mul},
        };

    const static std::map<
        const std::string,
        const std::function<RCP<const Boolean>(const RCP<const Boolean> &)>>
        single_arg_boolean_boolean_functions = {
            {"not", logical_not},
        };

    const static std::map<
        const std::string,
        const std::function<RCP<const Boolean>(const RCP<const Basic> &,
                                               const RCP<const Basic> &)>>
        double_arg_boolean_functions
        = {{"eq", (double_arg_boolean_func)Eq}, {"neq", Ne}};

    const static std::map<const std::string,
                          const std::function<RCP<const Boolean>(vec_basic &)>>
        multi_arg_vec_basic_boolean_functions
        = {{"geq", vec_ge}, {"gt", vec_gt}, {"leq", vec_le}, {"lt", vec_lt}};

    const static std::map<
        const std::string,
        const std::function<RCP<const Boolean>(vec_boolean &)>>
        multi_arg_vec_boolean_functions = {{"xor", logical_xor}};

    const static std::map<
        const std::string,
        const std::function<RCP<const Boolean>(set_boolean &)>>
        multi_arg_set_boolean_functions = {
            {"and", logical_and},
            {"or", logical_or},
        };

    std::string lname = lowercase(name);
    if (params.size() == 1) {
        const auto &single_arg_functions_
            = init_sbml_parser_single_arg_functions();
        auto it1 = single_arg_functions_.find(lname);
        if (it1 != single_arg_functions_.end()) {
            return it1->second(params[0]);
        }
        auto it3 = single_arg_boolean_boolean_functions.find(lname);
        if (it3 != single_arg_boolean_boolean_functions.end()) {
            return it3->second(rcp_static_cast<const Boolean>(params[0]));
        }
    }

    if (params.size() == 2) {
        auto it1 = double_arg_functions.find(lname);
        if (it1 != double_arg_functions.end()) {
            return it1->second(params[0], params[1]);
        }
        auto it2 = double_arg_boolean_functions.find(lname);
        if (it2 != double_arg_boolean_functions.end()) {
            return it2->second(params[0], params[1]);
        }
    }

    auto it1 = multi_arg_functions.find(lname);
    if (it1 != multi_arg_functions.end()) {
        return it1->second(params);
    }

    if (lname == "piecewise") {
        PiecewiseVec v;
        for (std::size_t i = 0; i < params.size() / 2; ++i) {
            // piecewise has pairs of {value, bool condition}
            v.push_back({params[2 * i],
                         rcp_static_cast<const Boolean>(params[2 * i + 1])});
        }
        if (params.size() % 2 == 1) {
            // piecewise can also have a default value
            v.push_back({params.back(), boolTrue});
        }
        return piecewise(std::move(v));
    }

    auto it2 = multi_arg_vec_boolean_functions.find(lname);
    if (it2 != multi_arg_vec_boolean_functions.end()) {
        vec_boolean p;
        for (auto &v : params) {
            p.push_back(rcp_static_cast<const Boolean>(v));
        }
        return it2->second(p);
    }

    auto it3 = multi_arg_set_boolean_functions.find(lname);
    if (it3 != multi_arg_set_boolean_functions.end()) {
        set_boolean s;
        for (auto &v : params) {
            s.insert(rcp_static_cast<const Boolean>(v));
        }
        return it3->second(s);
    }

    auto it4 = multi_arg_vec_basic_boolean_functions.find(lname);
    if (it4 != multi_arg_vec_basic_boolean_functions.end()) {
        if (params.size() < 2) {
            throw ParseError("Parsing Unsuccessful: '" + it4->first
                             + "' must have at least 2 arguments");
        }
        return it4->second(params);
    }

    return function_symbol(name, params);
}

RCP<const Basic> SbmlParser::parse_identifier(const std::string &expr)
{
    const static std::map<const std::string, const RCP<const Basic>>
        parser_constants = {{"pi", pi},
                            {"exponentiale", E},
                            {"avogadro", symbol("avogadro")},
                            {"time", symbol("time")},
                            {"inf", Inf},
                            {"infinity", Inf},
                            {"nan", Nan},
                            {"notanumber", Nan},
                            {"true", boolTrue},
                            {"false", boolFalse}};

    auto it_l = local_parser_constants.find(expr);
    if (it_l != local_parser_constants.end()) {
        return it_l->second;
    }
    std::string lexpr = lowercase(expr);
    auto it_c = parser_constants.find(lexpr);
    if (it_c != parser_constants.end()) {
        return it_c->second;
    } else {
        return symbol(expr);
    }
}

SbmlParser::SbmlParser(
    const std::map<const std::string, const RCP<const Basic>> &parser_constants)
    : Parser(parser_constants), m_tokenizer{new SbmlTokenizer()}
{
}

SbmlParser::~SbmlParser() = default;

} // namespace SymEngine
