#include "catch.hpp"
#include <symengine/basic.h>
#include <symengine/visitor.h>
#include <symengine/eval_double.h>
#include <symengine/derivative.h>
#include <symengine/symengine_exception.h>
#include <cstring>

using SymEngine::Basic;
using SymEngine::Add;
using SymEngine::Mul;
using SymEngine::Symbol;
using SymEngine::symbol;
using SymEngine::FunctionSymbol;
using SymEngine::umap_basic_num;
using SymEngine::map_basic_num;
using SymEngine::map_basic_basic;
using SymEngine::umap_basic_basic;
using SymEngine::map_uint_mpz;
using SymEngine::unified_compare;
using SymEngine::map_int_Expr;
using SymEngine::multiset_basic;
using SymEngine::vec_basic;
using SymEngine::set_basic;
using SymEngine::Integer;
using SymEngine::integer;
using SymEngine::Rational;
using SymEngine::real_double;
using SymEngine::one;
using SymEngine::zero;
using SymEngine::Number;
using SymEngine::pow;
using SymEngine::RCP;
using SymEngine::make_rcp;
using SymEngine::print_stack_on_segfault;
using SymEngine::Complex;
using SymEngine::complex_double;
using SymEngine::has_symbol;
using SymEngine::coeff;
using SymEngine::is_a;
using SymEngine::rcp_static_cast;
using SymEngine::set_basic;
using SymEngine::free_symbols;
using SymEngine::function_symbol;
using SymEngine::rational_class;
using SymEngine::pi;
using SymEngine::diff;
using SymEngine::sdiff;
using SymEngine::down_cast;
using SymEngine::NotImplementedError;
using SymEngine::ComplexInf;
using SymEngine::Nan;
using SymEngine::EulerGamma;
using SymEngine::atoms;
using SymEngine::tribool;

using namespace SymEngine::literals;

TEST_CASE("Test version", "[basic]")
{
    REQUIRE(std::strcmp(SymEngine::get_version(), SYMENGINE_VERSION) == 0);
}

TEST_CASE("Symbol hash: Basic", "[basic]")
{
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> x2 = symbol("x");
    RCP<const Symbol> y = symbol("y");

    REQUIRE(x->__eq__(*x));
    REQUIRE(x->__eq__(*x2));
    REQUIRE(not(x->__eq__(*y)));
    REQUIRE(x->__neq__(*y));

    std::hash<Basic> hash_fn;
    // Hashes of x and x2 must be the same:
    REQUIRE(hash_fn(*x) == hash_fn(*x2));

    // This checks that the hash of the Symbols are ordered:
    REQUIRE(hash_fn(*x) < hash_fn(*y));
}

TEST_CASE("Symbol dict: Basic", "[basic]")
{
    umap_basic_num ubn;
    map_basic_num mbn;
    umap_basic_basic ubb;
    map_basic_basic mbb;
    vec_basic vb;
    set_basic sb;
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> x2 = symbol("x");
    RCP<const Basic> y = symbol("y");
    RCP<const Number> i2 = integer(2);
    RCP<const Number> i3 = integer(3);
    bool p = (x != x2);
    REQUIRE(p);           // The instances are different...
    REQUIRE(eq(*x, *x2)); // ...but equal in the SymPy sense

    std::stringstream buffer;
    buffer << ubn;
    REQUIRE(buffer.str() == "{}");
    insert(ubn, x, i2);
    buffer.str("");
    buffer << ubn;
    REQUIRE(buffer.str() == "{x: 2}");
    insert(ubn, y, i3);
    buffer.str("");
    buffer << mbn;
    REQUIRE(buffer.str() == "{}");
    insert(mbn, x, i2);
    insert(mbn, y, i3);
    buffer.str("");
    buffer << mbb;
    REQUIRE(buffer.str() == "{}");
    insert(mbb, x, i2);
    insert(mbb, y, i3);
    buffer.str("");
    buffer << ubb;
    REQUIRE(buffer.str() == "{}");
    insert(ubb, x, i2);
    insert(ubb, y, i3);
    buffer.str("");
    buffer << vb;
    REQUIRE(buffer.str() == "{}");
    vb.push_back(x);
    vb.push_back(i3);
    buffer.str("");
    buffer << vb;
    REQUIRE(buffer.str() == "{x, 3}");
    REQUIRE(unified_eq(vb, {x, i3}));
    REQUIRE(unified_compare(vb, {x, i3}) == 0);
    REQUIRE(not unified_eq(vb, {i3, x}));
    REQUIRE(vec_basic_eq_perm(vb, {i3, x}));
    REQUIRE(not unified_eq(vb, {i3}));
    REQUIRE(not vec_basic_eq_perm(vb, {i3}));
    REQUIRE(unified_compare(vb, {i3}) == 1);
    buffer.str("");
    buffer << sb;
    REQUIRE(buffer.str() == "{}");
    sb.insert(i2);
    sb.insert(y);

    auto check_map_str = [](std::string to_chk, std::vector<std::string> key,
                            std::vector<std::string> val) {
        if (key.size() != val.size())
            return false;
        for (unsigned i = 0; i < key.size(); i++) {
            if (to_chk.find(key[i] + std::string(": " + val[i]))
                == std::string::npos)
                return false;
        }
        return true;
    };

    buffer.str("");
    buffer << ubn;
    REQUIRE(check_map_str(buffer.str(), {"x", "y"}, {"2", "3"}));
    buffer.str("");
    buffer << mbn;
    REQUIRE(check_map_str(buffer.str(), {"x", "y"}, {"2", "3"}));
    buffer.str("");
    buffer << mbb;
    REQUIRE(check_map_str(buffer.str(), {"x", "y"}, {"2", "3"}));
    buffer.str("");
    buffer << ubb;
    REQUIRE(check_map_str(buffer.str(), {"x", "y"}, {"2", "3"}));
    buffer.str("");
    buffer << ubb;
    REQUIRE(check_map_str(buffer.str(), {"x", "y"}, {"2", "3"}));
    buffer.str("");
    buffer << vb;
    bool check_vec_str;
    check_vec_str = buffer.str() == "{x, 3}";
    REQUIRE(check_vec_str);
    buffer.str("");
    buffer << sb;
    check_vec_str = buffer.str() == "{2, y}";
    REQUIRE(check_vec_str);

    map_uint_mpz a = {{0, 1_z}, {1, 2_z}, {2, 1_z}};
    map_uint_mpz b = {{0, 1_z}, {2, 1_z}, {1, 2_z}};
    REQUIRE(unified_compare(a, b) == 0);
    b = {{0, 1_z}, {2, 1_z}};
    REQUIRE(unified_compare(a, b) == 1);
    b = {{0, 1_z}, {3, 1_z}, {1, 2_z}};
    REQUIRE(unified_compare(a, b) == -1);
    b = {{0, 1_z}, {3, 1_z}, {1, 2_z}};
    REQUIRE(unified_compare(a, b) == -1);
    b = {{0, 1_z}, {1, 1_z}, {2, 3_z}};
    REQUIRE(unified_compare(a, b) == 1);

    map_int_Expr adict = {{0, 1}, {1, 2}, {2, x}};
    map_int_Expr bdict = {{0, 1}, {1, 2}, {2, x}};
    REQUIRE(unified_compare(adict, bdict) == 0);
    bdict = {{0, 1}, {1, 1}, {2, x}};
    REQUIRE(unified_compare(adict, bdict) != 0);
    REQUIRE(unified_compare(adict, bdict) == -unified_compare(bdict, adict));
    adict = {{0, 1}, {1, 1}, {3, x}};
    REQUIRE(unified_compare(adict, bdict) != 0);
    REQUIRE(unified_compare(adict, bdict) == -unified_compare(bdict, adict));
    bdict = {{0, 1}, {1, 3}};
    REQUIRE(unified_compare(adict, bdict) == 1);
    buffer.str("");
    buffer << bdict;
    REQUIRE(check_map_str(buffer.str(), {"0", "1"}, {"1", "3"}));

    multiset_basic msba, msbb;
    msba.insert(x);
    msba.insert(y);
    msba.insert(i2);
    msbb.insert(y);
    msbb.insert(i2);
    REQUIRE(not unified_eq(msba, msbb));
    REQUIRE(unified_compare(msba, msbb) == 1);
    msbb.insert(x);
    REQUIRE(unified_eq(msba, msbb));
    REQUIRE(unified_compare(msba, msbb) == 0);
    msbb.insert(i3);
    REQUIRE(not unified_eq(msba, msbb));
    REQUIRE(unified_compare(msba, msbb) == -1);
    REQUIRE(not unified_eq(msba, {x, y, i3}));
    REQUIRE(unified_compare(msba, {x, y, i3}) == -1);
}

TEST_CASE("Add: basic", "[basic]")
{
    umap_basic_num m, m2;
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    insert(m, x, integer(2));
    insert(m, y, integer(3));

    m2 = m;
    RCP<const Add> a = make_rcp<const Add>(zero, std::move(m2));
    insert(m, x, integer(-2));
    RCP<const Add> b = make_rcp<const Add>(zero, std::move(m));
    std::cout << *a << std::endl;
    std::cout << *b << std::endl;

    RCP<const Basic> r = add(add(x, y), add(y, x));
    std::cout << *r << std::endl;

    r = add(x, x);
    std::cout << *r << std::endl;

    r = add(add(x, x), y);
    std::cout << *r << std::endl;
    std::cout << "----------------------" << std::endl;

    REQUIRE(vec_basic_eq_perm(r->get_args(), {mul(integer(2), x), y}));
    REQUIRE(not vec_basic_eq_perm(r->get_args(), {mul(integer(3), x), y}));

    RCP<const Add> ar = rcp_static_cast<const Add>(r);
    REQUIRE(eq(*ar->get_coef(), *zero));
    const umap_basic_num &addmap = ar->get_dict();
    auto search = addmap.find(x);
    REQUIRE(search != addmap.end());
    REQUIRE(eq(*search->second, *integer(2)));
    search = addmap.find(y);
    REQUIRE(search != addmap.end());
    REQUIRE(eq(*search->second, *integer(1)));

    RCP<const Basic> term1, term2;
    RCP<const Add> a1 = rcp_static_cast<const Add>(add(r, r));
    a1->as_two_terms(outArg(term1), outArg(term2));
    RCP<const Add> a2 = rcp_static_cast<const Add>(add(term1, term2));
    REQUIRE(eq(*a1, *a2));

    r = add(mul(integer(5), x), integer(5));
    ar = rcp_static_cast<const Add>(r);
    REQUIRE(eq(*ar->get_coef(), *integer(5)));
    REQUIRE(vec_basic_eq_perm(r->get_args(), {mul(integer(5), x), integer(5)}));

    r = add(add(mul(mul(integer(2), x), y), integer(5)), pow(x, integer(2)));
    REQUIRE(vec_basic_eq_perm(
        r->get_args(),
        {integer(5), mul(mul(integer(2), x), y), pow(x, integer(2))}));
    std::cout << *r << std::endl;
}

TEST_CASE("Integer: Basic", "[basic]")
{
    RCP<const Integer> i = integer(5);
    RCP<const Integer> j = integer(6);
    RCP<const Basic> r;
    std::cout << *i << std::endl;
    std::cout << *j << std::endl;

    RCP<const Number> k = addnum(i, j);
    std::cout << *k << std::endl;
    REQUIRE(eq(*k, *integer(11)));
    REQUIRE(neq(*k, *integer(12)));

    k = subnum(i, j);
    std::cout << *k << std::endl;
    REQUIRE(eq(*k, *integer(-1)));
    REQUIRE(neq(*k, *integer(12)));

    k = mulnum(i, j);
    std::cout << *k << std::endl;
    REQUIRE(eq(*k, *integer(30)));
    REQUIRE(neq(*k, *integer(12)));

    k = divnum(i, j);
    REQUIRE(eq(*k, *Rational::from_two_ints(*integer(5), *integer(6))));
    std::cout << *k << std::endl;

    k = divnum(i, zero);
    REQUIRE(eq(*k, *ComplexInf));

    k = divnum(zero, i);
    REQUIRE(eq(*k, *zero));

    k = divnum(zero, zero);
    REQUIRE(eq(*k, *Nan));

    k = pownum(i, j);
    REQUIRE(eq(*k, *integer(15625)));
    std::cout << *k << std::endl;

    k = pownum(i, j->neg());
    REQUIRE(eq(*k, *Rational::from_two_ints(*integer(1), *integer(15625))));
    std::cout << *k << std::endl;

    k = i->neg();
    std::cout << *k << std::endl;
    REQUIRE(eq(*k, *integer(-5)));
    REQUIRE(neq(*k, *integer(12)));

    REQUIRE(not i->is_complex());

    i = integer(0);
    j = integer(0);
    r = i->div(*j);
    REQUIRE(eq(*r, *Nan));
}

TEST_CASE("Rational: Basic", "[basic]")
{
    RCP<const Number> r1, r2, r3;
    RCP<const Rational> r;
    rational_class a, b;

    r1 = Rational::from_two_ints(*integer(5), *integer(6));
    std::cout << *r1 << std::endl;
    REQUIRE(eq(*r1, *Rational::from_two_ints(*integer(5), *integer(6))));
    REQUIRE(neq(*r1, *Rational::from_two_ints(*integer(5), *integer(7))));

    r1 = Rational::from_two_ints(*integer(2), *integer(4));
    r2 = Rational::from_two_ints(*integer(1), *integer(2));
    REQUIRE(eq(*r1, *r2));

    r1 = Rational::from_two_ints(*integer(-2), *integer(3));
    r2 = Rational::from_two_ints(*integer(2), *integer(-3));
    REQUIRE(eq(*r1, *r2));

    r1 = Rational::from_two_ints(*integer(4), *integer(2));
    r2 = integer(2);
    REQUIRE(eq(*r1, *r2));

    r1 = Rational::from_two_ints(*integer(2), *integer(3));
    r2 = Rational::from_two_ints(*integer(5), *integer(7));
    r3 = Rational::from_two_ints(*integer(10), *integer(21));
    REQUIRE(eq(*mulnum(r1, r2), *r3));

    r1 = Rational::from_two_ints(*integer(2), *integer(3));
    r2 = Rational::from_two_ints(*integer(1), *integer(2));
    r3 = Rational::from_two_ints(*integer(1), *integer(3));
    REQUIRE(eq(*mulnum(r1, r2), *r3));

    r1 = Rational::from_two_ints(*integer(2), *integer(3));
    r2 = Rational::from_two_ints(*integer(9), *integer(2));
    r3 = integer(3);
    REQUIRE(eq(*mulnum(r1, r2), *r3));

    r1 = Rational::from_two_ints(*integer(1), *integer(2));
    r2 = integer(1);
    REQUIRE(eq(*addnum(r1, r1), *r2));

    r1 = Rational::from_two_ints(*integer(1), *integer(2));
    r2 = Rational::from_two_ints(*integer(1), *integer(3));
    r3 = Rational::from_two_ints(*integer(1), *integer(6));
    REQUIRE(eq(*subnum(r1, r2), *r3));

    r1 = Rational::from_two_ints(*integer(1), *integer(6));
    r2 = Rational::from_two_ints(*integer(1), *integer(3));
    r3 = Rational::from_two_ints(*integer(1), *integer(2));
    REQUIRE(eq(*divnum(r1, r2), *r3));

    r1 = Rational::from_two_ints(*integer(2), *integer(3));
    r2 = integer(2);
    r3 = Rational::from_two_ints(*integer(4), *integer(9));
    REQUIRE(eq(*pownum(r1, r2), *r3));

    r1 = Rational::from_two_ints(*integer(2), *integer(3));
    r2 = integer(-2);
    r3 = Rational::from_two_ints(*integer(9), *integer(4));
    REQUIRE(eq(*pownum(r1, r2), *r3));

    r1 = Rational::from_two_ints(*integer(2), *integer(3));
    r2 = integer(3);
    r3 = Rational::from_two_ints(*integer(8), *integer(27));
    REQUIRE(eq(*pownum(r1, r2), *r3));

    r1 = Rational::from_two_ints(*integer(2), *integer(3));
    r2 = integer(-3);
    r3 = Rational::from_two_ints(*integer(27), *integer(8));
    REQUIRE(eq(*pownum(r1, r2), *r3));

    r1 = Rational::from_two_ints(*integer(2), *integer(3));
    r2 = integer(3);
    r3 = integer(2);
    REQUIRE(eq(*mulnum(r1, r2), *r3));
    REQUIRE(eq(*mulnum(r2, r1), *r3));

    r1 = Rational::from_two_ints(*integer(2), *integer(3));
    r2 = integer(3);
    r3 = Rational::from_two_ints(*integer(11), *integer(3));
    REQUIRE(eq(*addnum(r1, r2), *r3));
    REQUIRE(eq(*addnum(r2, r1), *r3));

    r1 = Rational::from_two_ints(*integer(2), *integer(3));
    r2 = integer(3);
    r3 = Rational::from_two_ints(*integer(-7), *integer(3));
    REQUIRE(eq(*subnum(r1, r2), *r3));
    r3 = Rational::from_two_ints(*integer(7), *integer(3));
    REQUIRE(eq(*subnum(r2, r1), *r3));

    r1 = Rational::from_two_ints(*integer(2), *integer(3));
    r2 = integer(3);
    r3 = Rational::from_two_ints(*integer(2), *integer(9));
    REQUIRE(eq(*divnum(r1, r2), *r3));
    r3 = Rational::from_two_ints(*integer(9), *integer(2));
    REQUIRE(eq(*divnum(r2, r1), *r3));

    r1 = Rational::from_two_ints(*integer(2), *integer(3));
    r2 = zero;
    REQUIRE(eq(*divnum(r1, r2), *ComplexInf));

    r1 = Rational::from_two_ints(*integer(2), *zero);
    REQUIRE(eq(*r1, *ComplexInf));

    r1 = Rational::from_two_ints(2, 0);
    REQUIRE(eq(*r1, *ComplexInf));

    r1 = Rational::from_two_ints(*zero, *zero);
    REQUIRE(eq(*r1, *Nan));
    r1 = Rational::from_two_ints(0, 0);
    REQUIRE(eq(*r1, *Nan));

    r1 = Rational::from_two_ints(*integer(3), *integer(5));
    REQUIRE(is_a<Rational>(*r1));
    r = rcp_static_cast<const Rational>(r1);
    a = rational_class(3, 5);
    b = r->as_rational_class();
    REQUIRE(a == b);

    r1 = Rational::from_two_ints(*integer(0), *integer(0));
    r2 = r1->div(*integer(0));
    REQUIRE(eq(*r2, *Nan));
    r1 = Rational::from_two_ints(*integer(0), *integer(0));
    r2 = r1->div(*Rational::from_two_ints(*integer(0), *integer(0)));
    REQUIRE(eq(*r2, *Nan));
    r2 = integer(0)->div(*r1);
    REQUIRE(eq(*r2, *Nan));
}

TEST_CASE("Mul: Basic", "[basic]")
{
    map_basic_basic m, m2;
    RCP<const Basic> x = symbol("x");
    RCP<const Basic> y = symbol("y");
    insert(m, x, integer(2));
    insert(m, y, integer(3));

    m2 = m;
    RCP<const Mul> a = make_rcp<const Mul>(one, std::move(m2));
    insert(m, x, integer(-2));
    RCP<const Mul> b = make_rcp<const Mul>(one, std::move(m));
    std::cout << *a << std::endl;
    std::cout << *b << std::endl;

    RCP<const Basic> r = mul(mul(x, y), mul(y, x));
    std::cout << *r << std::endl;

    REQUIRE(vec_basic_eq_perm(r->get_args(),
                              {pow(x, integer(2)), pow(y, integer(2))}));

    r = mul(mul(pow(x, integer(3)), integer(2)), y);
    REQUIRE(
        vec_basic_eq_perm(r->get_args(), {integer(2), pow(x, integer(3)), y}));

    r = add(x, x);
    REQUIRE(vec_basic_eq_perm(r->get_args(), {x, integer(2)}));

    r = sub(x, x);
    REQUIRE(unified_eq(r->get_args(), {}));

    r = mul(x, x);
    REQUIRE(unified_eq(r->get_args(), {x, integer(2)}));

    r = div(x, x);
    REQUIRE(unified_eq(r->get_args(), {}));

    REQUIRE(eq(*div(integer(1), zero), *ComplexInf));

    r = mul(mul(mul(x, y), mul(x, integer(2))), integer(3));
    RCP<const Mul> mr = rcp_static_cast<const Mul>(r);
    REQUIRE(eq(*mr->get_coef(), *integer(6)));
    const map_basic_basic &mulmap = mr->get_dict();
    auto search = mulmap.find(x);
    REQUIRE(search != mulmap.end());
    REQUIRE(eq(*search->second, *integer(2)));
    search = mulmap.find(y);
    REQUIRE(search != mulmap.end());
    REQUIRE(eq(*search->second, *integer(1)));
}

TEST_CASE("Diff: Basic", "[basic]")
{
    RCP<const Basic> r1, r2;
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> i3 = integer(3);
    RCP<const Basic> i5 = integer(5);
    RCP<const Basic> i10 = integer(10);
    r1 = integer(5);
    r2 = r1->diff(x);
    REQUIRE(eq(*r2, *zero));

    r1 = Rational::from_two_ints(*integer(2), *integer(3));
    r2 = r1->diff(x);
    REQUIRE(eq(*r2, *zero));

    r1 = pow(x, i3)->diff(x);
    r2 = mul(i3, pow(x, i2));
    REQUIRE(eq(*r1, *r2));

    r1 = pow(add(x, y), i2)->diff(x);
    r2 = mul(i2, add(x, y));
    REQUIRE(eq(*r1, *r2));

    r1 = add(add(i2, mul(i3, x)), mul(i5, pow(x, i2)));
    REQUIRE(eq(*r1->diff(x), *add(i3, mul(i10, x))));
    REQUIRE(eq(*r1->diff(x)->diff(x), *i10));

    r1 = add(mul(mul(pow(x, y), pow(y, x)), i2), one)->diff(x);
    r2 = add(mul(i2, mul(pow(x, y), mul(pow(y, x), log(y)))),
             mul(i2, mul(pow(x, y), mul(pow(y, x), div(y, x)))));
    REQUIRE(eq(*r1, *r2));

    r1 = sdiff(add(pow(x, i2), x), pow(x, i2));
    r2 = one;
    REQUIRE(eq(*r1, *r2));

    r1 = sdiff(add(pow(x, i2), x), x);
    r2 = diff(add(pow(x, i2), x), x);
    REQUIRE(eq(*r1, *r2));

    r1 = diff(mul(x, add(one, x)), x);
    r2 = add(one, mul(i2, x));
    REQUIRE(eq(*r1, *r2));

    // Test that this doesn't segfault
    r1 = pow(sqrt(div(x, i3)), real_double(2.0));
    r1 = diff(r1, x);
}

TEST_CASE("compare: Basic", "[basic]")
{
    RCP<const Basic> r1, r2;
    RCP<const Symbol> x = symbol("x");
    RCP<const Symbol> y = symbol("y");
    RCP<const Symbol> z = symbol("z");
    RCP<const Basic> i2 = integer(2);
    RCP<const Basic> im2 = integer(-2);
    RCP<const Basic> i3 = integer(3);
    CHECK(x->compare(*x) == 0);
    CHECK(x->compare(*y) == -1);
    CHECK(x->compare(*z) == -1);
    CHECK(y->compare(*x) == 1);
    CHECK(y->compare(*z) == -1);
    CHECK(z->compare(*x) == 1);
    CHECK(z->compare(*y) == 1);

    CHECK(i2->compare(*i2) == 0);
    CHECK(i2->compare(*i3) == -1);
    CHECK(i3->compare(*i2) == 1);

    r1 = mul(x, y);
    r2 = mul(x, y);
    CHECK(r1->compare(*r2) == 0);
    CHECK(r2->compare(*r1) == 0);

    r1 = mul(x, y);
    r2 = mul(x, z);
    //    CHECK(r1->compare(*r2) == -1);
    //    CHECK(r2->compare(*r1) == 1);

    r1 = mul(y, x);
    r2 = mul(x, z);
    //    CHECK(r1->compare(*r2) == -1);
    //    CHECK(r2->compare(*r1) == 1);

    r1 = mul(y, x);
    r2 = mul(x, z);
    //    CHECK(r1->compare(*r2) == -1);
    //    CHECK(r2->compare(*r1) == 1);

    r1 = mul(mul(y, x), z);
    r2 = mul(x, z);
    CHECK(r1->compare(*r2) == 1);
    CHECK(r2->compare(*r1) == -1);

    r1 = add(add(y, x), z);
    r2 = add(x, z);
    CHECK(r1->compare(*r2) == 1);
    CHECK(r2->compare(*r1) == -1);

    r1 = pow(x, z);
    r2 = pow(y, x);
    CHECK(r1->compare(*r2) == -1);
    CHECK(r2->compare(*r1) == 1);

    r1 = pow(x, z);
    r2 = pow(x, x);
    CHECK(r1->compare(*r2) == 1);
    CHECK(r2->compare(*r1) == -1);

    r1 = add(add(x, y), z);
    r2 = add(x, y);
    CHECK(r1->compare(*r2) == 1);
    CHECK(r2->compare(*r1) == -1);

    r1 = add(add(x, y), i2);
    r2 = add(x, y);
    CHECK(r1->compare(*r2) == 1);
    CHECK(r2->compare(*r1) == -1);

    r1 = add(add(x, y), im2);
    r2 = add(x, y);
    CHECK(r1->compare(*r2) == -1);
    CHECK(r2->compare(*r1) == 1);

    r1 = add(x, y);
    r2 = add(x, z);
    //    CHECK(r1->compare(*r2) == -1);
    //    CHECK(r2->compare(*r1) == 1);

    r1 = add(x, y);
    r2 = add(x, y);
    CHECK(r1->compare(*r2) == 0);
    CHECK(r2->compare(*r1) == 0);

    r1 = add(add(x, y), z);
    r2 = add(add(x, z), y);
    CHECK(r1->compare(*r2) == 0);
    CHECK(r2->compare(*r1) == 0);

    r1 = sin(x);
    r2 = sin(y);
    CHECK(r1->compare(*r2) == -1);
    CHECK(r2->compare(*r1) == 1);
    CHECK(r1->compare(*r1) == 0);

    r1 = real_double(1.0);
    r2 = real_double(2.0);
    CHECK(r1->compare(*r2) == -1);
    CHECK(r2->compare(*r1) == 1);
    CHECK(r1->compare(*r1) == 0);

    // These are specific to the order in the declaration of enum TypeID,
    // so we just make sure that if x < y, then y > x.
    r1 = add(x, z);
    r2 = mul(x, y);
    int cmp = r1->__cmp__(*r2);
    CHECK(cmp != 0);
    CHECK(r2->__cmp__(*r1) == -cmp);

    r1 = mul(x, pow(z, x));
    r2 = mul(x, y);
    cmp = r1->__cmp__(*r2);
    CHECK(cmp != 0);
    CHECK(r2->__cmp__(*r1) == -cmp);

    r1 = mul(x, pow(z, x));
    r2 = mul(x, z);
    cmp = r1->__cmp__(*r2);
    CHECK(cmp != 0);
    CHECK(r2->__cmp__(*r1) == -cmp);

    r1 = pow(z, x);
    r2 = pow(z, pow(x, y));
    cmp = r1->__cmp__(*r2);
    CHECK(cmp != 0);
    CHECK(r2->__cmp__(*r1) == -cmp);

    r1 = div(mul(x, y), i2);
    r2 = mul(x, y);
    cmp = r1->__cmp__(*r2);
    CHECK(cmp != 0);
    CHECK(r2->__cmp__(*r1) == -cmp);

    r1 = add(x, pow(z, x));
    r2 = add(x, y);
    cmp = r1->__cmp__(*r2);
    CHECK(cmp != 0);
    CHECK(r2->__cmp__(*r1) == -cmp);

    r1 = log(log(x));
    r2 = log(x);
    CHECK(r1->__cmp__(*r2) != 0);
    CHECK(r1->__cmp__(*r1) == 0);

    CHECK_THROWS_AS(r2->expand_as_exp(), NotImplementedError &);

    r1 = pi;
    r2 = EulerGamma;
    CHECK(r1->__cmp__(*r2) > 0);
    CHECK(r1->__cmp__(*r1) == 0);
}

TEST_CASE("Complex: Basic", "[basic]")
{
    RCP<const Number> r1, r2, r3, c1, c2, c3;
    RCP<const Complex> c;
    RCP<const Basic> s;
    r1 = Rational::from_two_ints(*integer(2), *integer(4));
    r2 = Rational::from_two_ints(*integer(5), *integer(7));
    r3 = Rational::from_two_ints(*integer(-5), *integer(7));

    c1 = Complex::from_two_nums(*r1, *r2);
    c2 = Complex::from_two_nums(*r1, *r3);

    REQUIRE(c1->is_complex());

    // Basic check for equality in Complex::from_two_nums and
    // Complex::from_two_rats
    REQUIRE(eq(*c1, *Complex::from_two_rats(down_cast<const Rational &>(*r1),
                                            down_cast<const Rational &>(*r2))));
    REQUIRE(
        neq(*c2, *Complex::from_two_rats(down_cast<const Rational &>(*r1),
                                         down_cast<const Rational &>(*r2))));

    // Checks for complex addition
    // Final result is int
    REQUIRE(eq(*addnum(c1, c2), *one));
    // Final result is complex
    r2 = Rational::from_two_ints(*integer(1), *integer(1));
    r3 = Rational::from_two_ints(*integer(10), *integer(7));
    c3 = Complex::from_two_nums(*r2, *r3);
    REQUIRE(eq(*addnum(c1, c1), *c3));
    // Final result is rational
    r1 = Rational::from_two_ints(*integer(1), *integer(4));
    r2 = Rational::from_two_ints(*integer(5), *integer(7));
    r3 = Rational::from_two_ints(*integer(-5), *integer(7));
    c1 = Complex::from_two_nums(*r1, *r2);
    c2 = Complex::from_two_nums(*r1, *r3);
    REQUIRE(eq(*addnum(c1, c2), *div(one, integer(2))));

    // Checks for complex subtraction
    r1 = Rational::from_two_ints(*integer(2), *integer(4));
    r2 = Rational::from_two_ints(*integer(5), *integer(7));
    r3 = Rational::from_two_ints(*integer(-5), *integer(7));

    c1 = Complex::from_two_nums(*r1, *r2);
    c2 = Complex::from_two_nums(*r1, *r3);
    // Final result is int
    REQUIRE(eq(*subnum(c1, c1), *zero));

    // Final result is rational
    r3 = Rational::from_two_ints(*integer(1), *integer(3));
    c1 = Complex::from_two_nums(*r1, *r2);
    c2 = Complex::from_two_nums(*r3, *r2);
    REQUIRE(eq(*subnum(c1, c2), *div(one, integer(6))));

    // Final result is complex
    r2 = Rational::from_two_ints(*integer(1), *integer(6));
    c1 = Complex::from_two_nums(*r1, *r1);
    c2 = Complex::from_two_nums(*r3, *r3);
    c3 = Complex::from_two_nums(*r2, *r2);
    REQUIRE(eq(*subnum(c1, c2), *c3));

    // Checks for complex multiplication
    r1 = Rational::from_two_ints(*integer(2), *integer(1));
    r2 = Rational::from_two_ints(*integer(1), *integer(1));
    r3 = Rational::from_two_ints(*integer(-1), *integer(1));
    // Final result is int
    c1 = Complex::from_two_nums(*r1, *r2);
    c2 = Complex::from_two_nums(*r1, *r3);
    REQUIRE(eq(*mulnum(c1, c2), *integer(5)));

    // Final result is rational
    r1 = Rational::from_two_ints(*integer(1), *integer(2));
    c1 = Complex::from_two_nums(*r1, *r2);
    c2 = Complex::from_two_nums(*r1, *r3);
    REQUIRE(eq(*mulnum(c1, c2), *div(integer(5), integer(4))));

    // Final result is complex
    c1 = Complex::from_two_nums(*r2, *r2);
    c2 = Complex::from_two_nums(*r3, *r3);
    c3 = Complex::from_two_nums(*(integer(0)), *(integer(-2)));
    REQUIRE(eq(*mulnum(c1, c2), *c3));

    // Check for complex division
    // Final result is complex
    c1 = Complex::from_two_nums(*r2, *r2);
    c2 = Complex::from_two_nums(*r2, *r3);
    c3 = Complex::from_two_nums(*(integer(0)), *(integer(1)));
    REQUIRE(eq(*divnum(c1, c2), *c3));

    // Final result is integer
    c1 = Complex::from_two_nums(*r2, *r2);
    c2 = Complex::from_two_nums(*r2, *r2);
    REQUIRE(eq(*divnum(c1, c2), *integer(1)));

    // Final result is rational
    r3 = Rational::from_two_ints(*integer(2), *integer(1));
    c1 = Complex::from_two_nums(*r2, *r2);
    c2 = Complex::from_two_nums(*r3, *r3);
    REQUIRE(eq(*divnum(c1, c2), *div(integer(1), integer(2))));

    r1 = Rational::from_two_ints(*integer(1), *integer(2));
    r2 = Rational::from_two_ints(*integer(3), *integer(4));
    c1 = Complex::from_two_nums(*r1, *r2);

    r1 = Rational::from_two_ints(*integer(5), *integer(6));
    r2 = Rational::from_two_ints(*integer(7), *integer(8));
    c2 = Complex::from_two_nums(*r1, *r2);

    r1 = Rational::from_two_ints(*integer(618), *integer(841));
    r2 = Rational::from_two_ints(*integer(108), *integer(841));
    c3 = Complex::from_two_nums(*r1, *r2);
    REQUIRE(eq(*divnum(c1, c2), *c3));

    r1 = Rational::from_two_ints(*integer(-23), *integer(96));
    r2 = Rational::from_two_ints(*integer(17), *integer(16));
    c3 = Complex::from_two_nums(*r1, *r2);
    REQUIRE(eq(*mulnum(c1, c2), *c3));

    r1 = Rational::from_two_ints(*integer(4), *integer(3));
    r2 = Rational::from_two_ints(*integer(13), *integer(8));
    c3 = Complex::from_two_nums(*r1, *r2);
    REQUIRE(eq(*addnum(c1, c2), *c3));

    r1 = Rational::from_two_ints(*integer(-1), *integer(3));
    r2 = Rational::from_two_ints(*integer(-1), *integer(8));
    c3 = Complex::from_two_nums(*r1, *r2);
    REQUIRE(eq(*subnum(c1, c2), *c3));

    REQUIRE(is_a<Complex>(*c3));
    c = rcp_static_cast<const Complex>(c3);
    REQUIRE(eq(*c->real_part(), *r1));
    REQUIRE(eq(*c->imaginary_part(), *r2));

    // Explicit division by zero checks
    REQUIRE(eq(*divnum(c1, integer(0)), *ComplexInf));

    r3 = Rational::from_two_ints(*integer(0), *integer(1));
    REQUIRE(eq(*divnum(c1, r3), *ComplexInf));

    c2 = Complex::from_two_nums(*r3, *r3);
    REQUIRE(eq(*divnum(c1, c2), *ComplexInf));

    c2 = divnum(c2, integer(0));
    REQUIRE(eq(*c2, *Nan));
    c2 = divnum(c2, r3);
    REQUIRE(eq(*c2, *Nan));
    c2 = divnum(c2, c2);
    REQUIRE(eq(*c2, *Nan));

    c2 = Complex::from_two_nums(*integer(0), *integer(0));
    s = c2->div(*integer(0));
    REQUIRE(eq(*s, *Nan));
    s = c2->div(*Rational::from_two_ints(*integer(0), *integer(0)));
    REQUIRE(eq(*s, *Nan));
    s = c2->div(*Complex::from_two_nums(*integer(0), *integer(0)));
    REQUIRE(eq(*s, *Nan));
    s = integer(0)->div(*c2);
    REQUIRE(eq(*s, *Nan));

    c1 = Complex::from_two_nums(*integer(2), *integer(5));
    c2 = Complex::from_two_nums(*integer(1), *integer(5));
    s = c1->sub(*Rational::from_two_ints(5, 5));
    REQUIRE(eq(*s, *c2));

    c1 = Complex::from_two_nums(*integer(2), *integer(5));
    c2 = Complex::from_two_nums(*integer(1), *integer(5));
    s = c1->sub(*integer(1));
    REQUIRE(eq(*s, *c2));

    c1 = integer(1);
    c2 = Complex::from_two_nums(*integer(-1), *integer(-5));
    s = c1->sub(*Complex::from_two_nums(*integer(2), *integer(5)));
    REQUIRE(eq(*s, *c2));

    c1 = Complex::from_two_nums(*integer(2), *integer(5));
    c2 = Rational::from_two_ints(5, 5);
    c3 = Complex::from_two_nums(*integer(-1), *integer(-5));
    s = c2->sub(*c1);
    REQUIRE(eq(*s, *c3));

    c1 = Complex::from_two_nums(*integer(2), *integer(5));
    c2 = Rational::from_two_ints(5, 5);
    c3 = Complex::from_two_nums(*integer(2), *integer(5));
    s = c1->div(*c2);
    REQUIRE(eq(*s, *c3));

    c1 = Complex::from_two_nums(*integer(2), *integer(5));
    c2 = Rational::from_two_ints(4, 5);
    CHECK_THROWS_AS(c2->div(*c1), NotImplementedError &);

    c1 = Complex::from_two_nums(*integer(2), *integer(5));
    c2 = integer(3);
    CHECK_THROWS_AS(c2->pow(*c1), NotImplementedError &);
}

TEST_CASE("has_symbol: Basic", "[basic]")
{
    RCP<const Basic> r1;
    RCP<const Symbol> x, y, z;
    x = symbol("x");
    y = symbol("y");
    z = symbol("z");
    r1 = add(x, pow(y, integer(2)));
    REQUIRE(has_symbol(*r1, *x));
    REQUIRE(has_symbol(*r1, *y));
    REQUIRE(not has_symbol(*r1, *z));

    r1 = sin(add(x, pow(y, integer(2))));
    REQUIRE(has_symbol(*r1, *x));
    REQUIRE(has_symbol(*r1, *y));
    REQUIRE(not has_symbol(*r1, *z));
}

TEST_CASE("coeff: Basic", "[basic]")
{
    RCP<const Basic> r1, r2, r3, r4, r5, r6;
    RCP<const Symbol> x, y, z;
    RCP<const Basic> f1, f2;
    x = symbol("x");
    y = symbol("y");
    z = symbol("z");
    f1 = function_symbol("f", x);
    f2 = function_symbol("f", y);
    r1 = add(x, pow(y, integer(2)));
    r2 = add(add(mul(integer(2), z), pow(x, integer(3))), pow(y, integer(2)));
    r3 = add(add(add(add(r2, mul(x, z)), f1), f2), mul(f1, integer(3)));
    r4 = mul(pow(x, integer(2)), y);
    r5 = expand(pow(add(x, y), 3));
    r6 = add(add(add(x, sin(x)), mul(x, sin(x))), y);
    REQUIRE(eq(*coeff(*x, *x, *integer(1)), *integer(1)));
    REQUIRE(eq(*coeff(*x, *x, *integer(0)), *integer(0)));

    REQUIRE(eq(*coeff(*r1, *x, *integer(1)), *integer(1)));
    REQUIRE(eq(*coeff(*r1, *x, *integer(1)), *integer(1)));
    REQUIRE(eq(*coeff(*r1, *y, *integer(0)), *x));
    REQUIRE(eq(*coeff(*r1, *y, *integer(1)), *integer(0)));
    REQUIRE(eq(*coeff(*r1, *y, *integer(2)), *integer(1)));
    REQUIRE(eq(*coeff(*r1, *z, *integer(2)), *integer(0)));

    REQUIRE(eq(*coeff(*r2, *y, *integer(0)),
               *add(mul(integer(2), z), pow(x, integer(3)))));
    REQUIRE(eq(*coeff(*r2, *y, *integer(1)), *integer(0)));
    REQUIRE(eq(*coeff(*r2, *y, *integer(2)), *integer(1)));
    REQUIRE(eq(*coeff(*r2, *z, *integer(0)),
               *add(pow(x, integer(3)), pow(y, integer(2)))));
    REQUIRE(eq(*coeff(*r2, *z, *integer(1)), *integer(2)));
    REQUIRE(eq(*coeff(*r2, *z, *integer(2)), *integer(0)));
    REQUIRE(eq(*coeff(*r2, *x, *integer(2)), *integer(0)));
    REQUIRE(eq(*coeff(*r2, *x, *integer(3)), *integer(1)));
    REQUIRE(eq(*coeff(*r2, *x, *integer(0)),
               *add(mul(integer(2), z), pow(y, integer(2)))));

    REQUIRE(eq(*coeff(*r3, *z, *integer(1)), *add(x, integer(2))));
    REQUIRE(eq(*coeff(*r3, *f1, *integer(1)), *integer(4)));
    REQUIRE(eq(*coeff(*r3, *f1, *integer(0)), *add(add(r2, mul(x, z)), f2)));
    REQUIRE(eq(*coeff(*r3, *f2, *integer(1)), *integer(1)));

    REQUIRE(eq(*coeff(*r4, *x, *integer(0)), *integer(0)));
    REQUIRE(eq(*coeff(*r4, *x, *integer(1)), *integer(0)));
    REQUIRE(eq(*coeff(*r4, *x, *integer(2)), *y));

    REQUIRE(eq(*coeff(*r5, *x, *integer(3)), *integer(1)));
    REQUIRE(eq(*coeff(*r5, *x, *integer(2)), *mul(integer(3), y)));
    REQUIRE(
        eq(*coeff(*r5, *x, *integer(1)), *mul(integer(3), pow(y, integer(2)))));
    REQUIRE(eq(*coeff(*r5, *x, *integer(0)), *pow(y, integer(3))));

    REQUIRE(eq(*coeff(*r6, *x, *integer(0)), *y));
    REQUIRE(eq(*coeff(*add(r6, one), *x, *integer(0)), *add(y, one)));
}

TEST_CASE("free_symbols: Basic", "[basic]")
{
    RCP<const Basic> r1;
    RCP<const Symbol> x, y, z;
    x = symbol("x");
    y = symbol("y");
    z = symbol("z");
    r1 = add(x, add(z, pow(y, x)));

    set_basic s = free_symbols(*r1);
    REQUIRE(s.size() == 3);
    REQUIRE(s.count(x) == 1);
    REQUIRE(s.count(y) == 1);
    REQUIRE(s.count(z) == 1);
    s.clear();

    r1 = function_symbol("f", mul(x, integer(2)))->diff(x);
    s = free_symbols(*r1);
    REQUIRE(s.size() == 1);
    REQUIRE(s.count(x) == 1);

    r1 = mul(x, integer(2));
    s = free_symbols(*r1);
    REQUIRE(s.size() == 1);
    REQUIRE(s.count(x) == 1);
}

TEST_CASE("function_symbols: Basic", "[basic]")
{
    RCP<const Basic> r1, f1, f2;
    RCP<const Symbol> x, y, z;
    x = symbol("x");
    y = symbol("y");
    z = symbol("z");
    f1 = function_symbol("f", x);
    f2 = function_symbol("g", {y, z});

    r1 = add(x, add(z, pow(y, x)));
    set_basic s = function_symbols(*r1);
    REQUIRE(s.size() == 0);
    s.clear();

    r1 = f1;
    s = function_symbols(*r1);
    REQUIRE(s.size() == 1);
    REQUIRE(s.count(f1) == 1);
    s.clear();

    r1 = cos(add(f1, f2));
    s = function_symbols(*r1);
    REQUIRE(s.size() == 2);
    REQUIRE(s.count(f1) == 1);
    REQUIRE(s.count(f2) == 1);
}

TEST_CASE("atoms: Basic", "[basic]")
{
    RCP<const Basic> r1, r2, r3;
    RCP<const Symbol> x, y;
    x = symbol("x");
    y = symbol("y");

    r1 = function_symbol("f", mul(x, integer(2)));
    set_basic s = atoms<FunctionSymbol>(*r1);
    REQUIRE(s.size() == 1);

    s = atoms<FunctionSymbol, Symbol>(*r1);
    REQUIRE(s.size() == 2);

    s = atoms<FunctionSymbol, Symbol, Mul>(*r1);
    REQUIRE(s.size() == 3);

    s = atoms<Number>(*r1);
    REQUIRE(s.size() == 1);

    r2 = function_symbol("g", add(r1, y));
    s = atoms<FunctionSymbol>(*r2);
    REQUIRE(s.size() == 2);

    r3 = add(r1, add(r2, x));
    map_basic_basic d;
    d[x] = r1;
    r3 = r3->subs(d);
    s = atoms<FunctionSymbol>(*r3);
    set_basic t({r1, r1->subs(d), r2->subs(d)});
    REQUIRE(unified_eq(s, t));

    r3 = r2->diff(x);
    s = atoms<FunctionSymbol>(*r3);
    REQUIRE(s.size() == 3);
}

TEST_CASE("args: Basic", "[basic]")
{
    RCP<const Basic> r1;
    RCP<const Symbol> x, y;
    x = symbol("x");
    y = symbol("y");

    r1 = add(x, pow(y, x));
    REQUIRE(vec_basic_eq_perm(r1->get_args(), {x, pow(y, x)}));

    r1 = pi;
    REQUIRE(r1->get_args().size() == 0);

    r1 = log(pi);
    REQUIRE(vec_basic_eq_perm(r1->get_args(), {pi}));
}

TEST_CASE("tribool", "[basic]")
{
    REQUIRE(!is_true(tribool::indeterminate));
    REQUIRE(!is_true(tribool::trifalse));
    REQUIRE(is_true(tribool::tritrue));

    REQUIRE(!is_false(tribool::indeterminate));
    REQUIRE(is_false(tribool::trifalse));
    REQUIRE(!is_false(tribool::tritrue));

    REQUIRE(is_indeterminate(tribool::indeterminate));
    REQUIRE(!is_indeterminate(tribool::trifalse));
    REQUIRE(!is_indeterminate(tribool::tritrue));

    REQUIRE(is_false(and_tribool(tribool::trifalse, tribool::trifalse)));
    REQUIRE(is_false(and_tribool(tribool::trifalse, tribool::indeterminate)));
    REQUIRE(is_false(and_tribool(tribool::trifalse, tribool::tritrue)));
    REQUIRE(is_false(and_tribool(tribool::indeterminate, tribool::trifalse)));
    REQUIRE(is_false(and_tribool(tribool::tritrue, tribool::trifalse)));
    REQUIRE(is_indeterminate(
        and_tribool(tribool::indeterminate, tribool::indeterminate)));
    REQUIRE(is_indeterminate(
        and_tribool(tribool::tritrue, tribool::indeterminate)));
    REQUIRE(is_indeterminate(
        and_tribool(tribool::indeterminate, tribool::tritrue)));
    REQUIRE(is_true(and_tribool(tribool::tritrue, tribool::tritrue)));

    REQUIRE(is_true(not_tribool(tribool::trifalse)));
    REQUIRE(is_false(not_tribool(tribool::tritrue)));
    REQUIRE(is_indeterminate(not_tribool(tribool::indeterminate)));

    REQUIRE(is_false(andwk_tribool(tribool::trifalse, tribool::trifalse)));
    REQUIRE(is_indeterminate(
        andwk_tribool(tribool::trifalse, tribool::indeterminate)));
    REQUIRE(is_false(andwk_tribool(tribool::trifalse, tribool::tritrue)));
    REQUIRE(is_indeterminate(
        andwk_tribool(tribool::indeterminate, tribool::trifalse)));
    REQUIRE(is_false(andwk_tribool(tribool::tritrue, tribool::trifalse)));
    REQUIRE(is_indeterminate(
        andwk_tribool(tribool::indeterminate, tribool::indeterminate)));
    REQUIRE(is_indeterminate(
        andwk_tribool(tribool::tritrue, tribool::indeterminate)));
    REQUIRE(is_indeterminate(
        andwk_tribool(tribool::indeterminate, tribool::tritrue)));
    REQUIRE(is_true(andwk_tribool(tribool::tritrue, tribool::tritrue)));
}
