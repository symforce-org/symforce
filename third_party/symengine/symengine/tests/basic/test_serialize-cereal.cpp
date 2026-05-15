#include "catch.hpp"

#include <symengine/basic.h>
#include <symengine/serialize-cereal.h>
#include <cereal/archives/binary.hpp>

using std::string;

using SymEngine::Basic;
using SymEngine::complex_double;
using SymEngine::Integer;
using SymEngine::is_a;
using SymEngine::Number;
using SymEngine::RCP;
using SymEngine::Symbol;

namespace se = SymEngine;

template <typename T>
string dumps(RCP<const T> obj)
{
    std::ostringstream oss;
    cereal::BinaryOutputArchive{oss}(obj);
    return oss.str();
}

template <typename T>
RCP<const T> loads(string sobj)
{
    RCP<const T> obj;
    std::istringstream iss(sobj);
    cereal::BinaryInputArchive{iss}(obj);
    return obj;
}

void check_string_serialization_roundtrip(RCP<const Basic> basic1)
{
    RCP<const Basic> basic2 = loads<Basic>(dumps<Basic>(basic1));
    REQUIRE(eq(*basic1, *basic2));
}

TEST_CASE("Test serialization using cereal", "[serialize-cereal]")
{
    RCP<const Symbol> symb_x_ori = se::symbol("x");
    string s_symb_x = dumps<Symbol>(symb_x_ori);
    RCP<const Symbol> symb_x_des = loads<Symbol>(s_symb_x);
    REQUIRE(symb_x_ori->__eq__(*symb_x_des));
    RCP<const Basic> basic_x_des = loads<Basic>(s_symb_x);
    REQUIRE(is_a<Symbol>(*basic_x_des));
    REQUIRE(!is_a<Integer>(*basic_x_des));

    // Symbol
    check_string_serialization_roundtrip(se::symbol("y"));
    // Add
    check_string_serialization_roundtrip(
        se::add(se::symbol("y"), se::integer(3)));
    // Pow
    check_string_serialization_roundtrip(
        se::pow(se::symbol("y"), se::integer(2)));
    check_string_serialization_roundtrip(se::reals());
    check_string_serialization_roundtrip(
        complex_double(std::complex<double>(4, 5)));
}
