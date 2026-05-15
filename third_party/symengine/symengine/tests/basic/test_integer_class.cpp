#include "catch.hpp"

#include <symengine/mp_class.h>
#include <symengine/basic.h>
#include <symengine/integer.h>
#include <symengine/rational.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/complex.h>
#include <symengine/complex_double.h>
#include <symengine/constants.h>
#include <iostream>

using std::cout;
using std::endl;

using SymEngine::integer_class;
using SymEngine::mp_abs;
using SymEngine::mp_get_ui;
using SymEngine::mp_sign;
using SymEngine::rational_class;

using namespace SymEngine;

TEST_CASE("operators: integer_class", "[integer_class]")
{
    integer_class i, j;
    rational_class r, p;

    // shift operators
    // Shift operators of negatives are unnecessary and
    // undefined behaviour for C++ standard
    /*REQUIRE((integer_class(-1024) << 3) == -8192);
    REQUIRE((integer_class(-1024) >> 3) == -128);
    REQUIRE((integer_class(-768) << 5) == -24576);
    REQUIRE((integer_class(-768) >> 5) == -24);
    REQUIRE((integer_class(-500) << 3) == -4000);
    REQUIRE((integer_class(-500) >> 3) == -63);
    REQUIRE((integer_class(-5) << 1) == -10);
    REQUIRE((integer_class(-5) >> 1) == -3);
    REQUIRE((integer_class(-4) << 0) == -4);
    REQUIRE((integer_class(-4) >> 0) == -4);
    REQUIRE((integer_class(-2) << 10) == -2048);
    REQUIRE((integer_class(-2) >> 10) == -1);
    REQUIRE((integer_class(-1) << 4) == -16);
    REQUIRE((integer_class(-1) >> 4) == -1);*/
    REQUIRE((integer_class(0) << 5) == 0);
    REQUIRE((integer_class(0) >> 5) == 0);
    REQUIRE((integer_class(1) << 2) == 4);
    REQUIRE((integer_class(1) >> 2) == 0);
    REQUIRE((integer_class(2) << 4) == 32);
    REQUIRE((integer_class(2) >> 4) == 0);
    REQUIRE((integer_class(4) << 1) == 8);
    REQUIRE((integer_class(4) >> 1) == 2);
    REQUIRE((integer_class(5) << 6) == 320);
    REQUIRE((integer_class(5) >> 6) == 0);
    REQUIRE((integer_class(500) << 1) == 1000);
    REQUIRE((integer_class(500) >> 1) == 250);
    REQUIRE((integer_class(768) << 2) == 3072);
    REQUIRE((integer_class(768) >> 2) == 192);
    REQUIRE((integer_class(1024) << 3) == 8192);
    REQUIRE((integer_class(1024) >> 3) == 128);

    // bitwise operators
    i = 9;                  // i == 1001
    j = 12;                 // j == 1100
    REQUIRE((i & j) == 8);  // bitwise and
    REQUIRE((i | j) == 13); // bitwise or
    REQUIRE((i ^ j) == 5);  // bitwise exclusive or

    // modulus operator (truncated)
    REQUIRE((integer_class(-1) % integer_class(1)) == 0);
    REQUIRE((integer_class(0) % integer_class(1)) == 0);
    REQUIRE((integer_class(1) % integer_class(1)) == 0);
    REQUIRE((integer_class(-1) % integer_class(2)) == -1);
    REQUIRE((integer_class(0) % integer_class(2)) == 0);
    REQUIRE((integer_class(1) % integer_class(2)) == 1);
    REQUIRE((integer_class(-1) % integer_class(-1)) == 0);
    REQUIRE((integer_class(0) % integer_class(-1)) == 0);
    REQUIRE((integer_class(1) % integer_class(-1)) == 0);
    REQUIRE((integer_class(-1) % integer_class(-2)) == -1);
    REQUIRE((integer_class(0) % integer_class(-2)) == 0);
    REQUIRE((integer_class(1) % integer_class(-2)) == 1);
    REQUIRE((integer_class(2) % integer_class(1)) == 0);
    REQUIRE((integer_class(2) % integer_class(-1)) == 0);
    REQUIRE((integer_class(-2) % integer_class(1)) == 0);
    REQUIRE((integer_class(-2) % integer_class(-1)) == 0);
    REQUIRE((integer_class(-8) % integer_class(3)) == -2);
    REQUIRE((integer_class(-8) % integer_class(-3)) == -2);
    REQUIRE((integer_class(8) % integer_class(-3)) == 2);
    REQUIRE((integer_class(8) % integer_class(-3)) == 2);
    REQUIRE((integer_class(8) % integer_class(3)) == 2);
    REQUIRE((integer_class(100) % integer_class(17)) == 15);

    // compound modulus operator
    j = -9;
    i = 7;
    j %= i;
    REQUIRE(j == -2);

    // move constructor and move assignment
    integer_class n = 5;
    integer_class m(std::move(n));
    i = std::move(m);

    // construction of rational_class
    // r = rational_class(integer_class(2),3);  fails!
    r = rational_class(integer_class(2), integer_class(3));

    // truncated division
    j = integer_class(12) / integer_class(5);
    REQUIRE(j == 2);
    j = integer_class(-12) / integer_class(5);
    REQUIRE(j == -2);
}

TEST_CASE("powers and roots: integer_class", "[integer_class]")
{
    integer_class res, i;

    // mp_pow_ui
    mp_pow_ui(res, -1, 0);
    REQUIRE(res == 1);
    mp_pow_ui(res, 0, 0);
    REQUIRE(res == 1); // gmp doc says 0**0 == 1
    mp_pow_ui(res, 1, 0);
    REQUIRE(res == 1);
    mp_pow_ui(res, -1, 1);
    REQUIRE(res == -1);
    mp_pow_ui(res, 0, 1);
    REQUIRE(res == 0);
    mp_pow_ui(res, 1, 1);
    REQUIRE(res == 1);
    mp_pow_ui(res, -1, 2);
    REQUIRE(res == 1);
    mp_pow_ui(res, 0, 2);
    REQUIRE(res == 0);
    mp_pow_ui(res, 1, 2);
    REQUIRE(res == 1);
    mp_pow_ui(res, -2, 0);
    REQUIRE(res == 1);
    mp_pow_ui(res, -1, 3);
    REQUIRE(res == -1);
    mp_pow_ui(res, -2, 1);
    REQUIRE(res == -2);
    mp_pow_ui(res, -2, 2);
    REQUIRE(res == 4);
    mp_pow_ui(res, -2, 3);
    REQUIRE(res == -8);
    mp_pow_ui(res, 2, 1);
    REQUIRE(res == 2);
    mp_pow_ui(res, 2, 3);
    REQUIRE(res == 8);

    // mp_powm
    mp_powm(res, -1, -1, -1);
    REQUIRE(res == 0);
    mp_powm(res, -1, -1, 1);
    REQUIRE(res == 0);
    mp_powm(res, -1, 0, -1);
    REQUIRE(res == 0);
    mp_powm(res, -1, 0, 1);
    REQUIRE(res == 0);
    mp_powm(res, -1, 1, -1);
    REQUIRE(res == 0);
    mp_powm(res, -1, 1, 1);
    REQUIRE(res == 0);
    mp_powm(res, 0, -1, -1);
    REQUIRE(res == 0);
    mp_powm(res, 0, -1, 1);
    REQUIRE(res == 0);
    mp_powm(res, 0, 0, -1);
    REQUIRE(res == 0);
    mp_powm(res, 0, 0, 1);
    REQUIRE(res == 0);
    mp_powm(res, 0, 1, -1);
    REQUIRE(res == 0);
    mp_powm(res, 0, 1, 1);
    REQUIRE(res == 0);
    mp_powm(res, 1, -1, -1);
    REQUIRE(res == 0);
    mp_powm(res, 1, -1, 1);
    REQUIRE(res == 0);
    mp_powm(res, 1, 0, -1);
    REQUIRE(res == 0);
    mp_powm(res, 1, 0, 1);
    REQUIRE(res == 0);
    mp_powm(res, 1, 1, -1);
    REQUIRE(res == 0);
    mp_powm(res, 1, 1, 1);
    REQUIRE(res == 0);
    mp_powm(res, -2, 3, 2);
    REQUIRE(res == 0);
    mp_powm(res, 3, -1, 5);
    REQUIRE(res == 2);
    mp_powm(res, -3, -1, 5);
    REQUIRE(res == 3);
    mp_powm(res, 3, 2, -5);
    REQUIRE(res == 4);
    mp_powm(res, 4, -2, 9);
    REQUIRE(res == 4);
    mp_powm(res, -4, 2, 9);
    REQUIRE(res == 7);
    mp_powm(res, 4, 2, 6);
    REQUIRE(res == 4);
    mp_powm(res, 9, -2, 4);
    REQUIRE(res == 1);
    mp_powm(res, 10, 2, 4);
    REQUIRE(res == 0);
    mp_powm(res, -9, -2, 4);
    REQUIRE(res == 1);
    mp_powm(res, -12, -3, 35);
    REQUIRE(res == 8);
    mp_powm(res, -3, -9, 10);
    REQUIRE(res == 3);
    mp_powm(res, -11, -6, 10);
    REQUIRE(res == 1);
    mp_powm(res, -4, 1, 5);
    REQUIRE(res == 1);
    mp_powm(res, -3, 2, 7);
    REQUIRE(res == 2);
    mp_powm(res, -3, 3, 7);
    REQUIRE(res == 1);
    mp_powm(res, -3, 1, 7);
    REQUIRE(res == 4);

    // mp_sqrt
    REQUIRE(mp_sqrt(0) == 0);
    REQUIRE(mp_sqrt(1) == 1);
    REQUIRE(mp_sqrt(2) == 1);
    REQUIRE(mp_sqrt(3) == 1);
    REQUIRE(mp_sqrt(4) == 2);
    REQUIRE(mp_sqrt(5) == 2);
    REQUIRE(mp_sqrt(9) == 3);
    REQUIRE(mp_sqrt(10) == 3);
    REQUIRE(mp_sqrt(24) == 4);
    REQUIRE(mp_sqrt(25) == 5);
    REQUIRE(mp_sqrt(26) == 5);
    REQUIRE(mp_sqrt(35) == 5);
    REQUIRE(mp_sqrt(36) == 6);
    REQUIRE(mp_sqrt(37) == 6);
    REQUIRE(mp_sqrt(288) == 16);
    REQUIRE(mp_sqrt(289) == 17);
    REQUIRE(mp_sqrt(290) == 17);
    REQUIRE(mp_sqrt(14640) == 120);
    REQUIRE(mp_sqrt(14641) == 121);
    REQUIRE(mp_sqrt(14642) == 121);

    // mp_root
    mp_root(res, 64, 3);
    REQUIRE(res == 4);
    mp_root(res, integer_class(1234567890123456789), 3);
    REQUIRE(res == 1072765);
    bool is_perfect_root = mp_root(res, -64, 3);
    REQUIRE(is_perfect_root);
    REQUIRE(res == -4);

    // mp_rootrem
    integer_class rem;
    mp_rootrem(res, rem, 50, 2);
    REQUIRE(res == 7);
    REQUIRE(rem == 1);

    // mp_perfect_power_p
    REQUIRE(mp_perfect_power_p(integer_class(1024)));
    REQUIRE(!mp_perfect_power_p(integer_class(1025)));
    mp_pow_ui(res, integer_class(6), 7);
    REQUIRE(mp_perfect_power_p(res));
    REQUIRE(mp_perfect_power_p(integer_class(9)));
    REQUIRE(mp_perfect_power_p(integer_class(1)));
    REQUIRE(mp_perfect_power_p(27));
    REQUIRE(mp_perfect_power_p(-27));
    REQUIRE(mp_perfect_power_p(-64));
    REQUIRE(mp_perfect_power_p(-32));

    // mp_perfect_square_p
    REQUIRE(mp_perfect_square_p(49));
    REQUIRE(!mp_perfect_square_p(50));
}

TEST_CASE("factors and multiples: integer_class", "[integer_class]")
{
    integer_class quo, rem, res, gcd;

    // mp_fdiv_qr
    mp_fdiv_qr(quo, rem, -1, 1);
    REQUIRE(quo == -1);
    REQUIRE(rem == 0);
    mp_fdiv_qr(quo, rem, 0, 1);
    REQUIRE(quo == 0);
    REQUIRE(rem == 0);
    mp_fdiv_qr(quo, rem, 1, 1);
    REQUIRE(quo == 1);
    REQUIRE(rem == 0);
    mp_fdiv_qr(quo, rem, -1, 2);
    REQUIRE(quo == -1);
    REQUIRE(rem == 1);
    mp_fdiv_qr(quo, rem, 0, 2);
    REQUIRE(quo == 0);
    REQUIRE(rem == 0);
    mp_fdiv_qr(quo, rem, 1, 2);
    REQUIRE(quo == 0);
    REQUIRE(rem == 1);
    mp_fdiv_qr(quo, rem, -1, -1);
    REQUIRE(quo == 1);
    REQUIRE(rem == 0);
    mp_fdiv_qr(quo, rem, 0, -1);
    REQUIRE(quo == 0);
    REQUIRE(rem == 0);
    mp_fdiv_qr(quo, rem, 1, -1);
    REQUIRE(quo == -1);
    REQUIRE(rem == 0);
    mp_fdiv_qr(quo, rem, -1, -2);
    REQUIRE(quo == 0);
    REQUIRE(rem == -1);
    mp_fdiv_qr(quo, rem, 0, -2);
    REQUIRE(quo == 0);
    REQUIRE(rem == 0);
    mp_fdiv_qr(quo, rem, 1, -2);
    REQUIRE(quo == -1);
    REQUIRE(rem == -1);
    mp_fdiv_qr(quo, rem, 2, 1);
    REQUIRE(quo == 2);
    REQUIRE(rem == 0);
    mp_fdiv_qr(quo, rem, 2, -1);
    REQUIRE(quo == -2);
    REQUIRE(rem == 0);
    mp_fdiv_qr(quo, rem, -2, 1);
    REQUIRE(quo == -2);
    REQUIRE(rem == 0);
    mp_fdiv_qr(quo, rem, -2, -1);
    REQUIRE(quo == 2);
    REQUIRE(rem == 0);
    mp_fdiv_qr(quo, rem, -8, 3);
    REQUIRE(quo == -3);
    REQUIRE(rem == 1);
    mp_fdiv_qr(quo, rem, -8, -3);
    REQUIRE(quo == 2);
    REQUIRE(rem == -2);
    mp_fdiv_qr(quo, rem, 8, -3);
    REQUIRE(quo == -3);
    REQUIRE(rem == -1);
    mp_fdiv_qr(quo, rem, 8, -3);
    REQUIRE(quo == -3);
    REQUIRE(rem == -1);
    mp_fdiv_qr(quo, rem, 8, 3);
    REQUIRE(quo == 2);
    REQUIRE(rem == 2);
    mp_fdiv_qr(quo, rem, -100, -17);
    REQUIRE(quo == 5);
    REQUIRE(rem == -15);
    mp_fdiv_qr(quo, rem, -100, 17);
    REQUIRE(quo == -6);
    REQUIRE(rem == 2);
    mp_fdiv_qr(quo, rem, 100, -17);
    REQUIRE(quo == -6);
    REQUIRE(rem == -2);
    mp_fdiv_qr(quo, rem, 100, 17);
    REQUIRE(quo == 5);
    REQUIRE(rem == 15);
    mp_fdiv_qr(quo, rem, 369, 12);
    REQUIRE(quo == 30);
    REQUIRE(rem == 9);
    mp_fdiv_qr(quo, rem, 123456789, 12345);
    REQUIRE(quo == 10000);
    REQUIRE(rem == 6789);

    // mp_fdiv_r
    mp_fdiv_r(rem, 15, 7);
    REQUIRE(rem == 1);
    mp_fdiv_r(rem, -15, 7);
    REQUIRE(rem == 6);
    mp_fdiv_r(rem, 15, -7);
    REQUIRE(rem == -6);
    mp_fdiv_r(rem, -15, -7);
    REQUIRE(rem == -1);
    mp_fdiv_r(rem, -1, 5);
    REQUIRE(rem == 4);

    mp_fdiv_q(quo, 15, 7);
    REQUIRE(quo == 2);
    mp_fdiv_q(quo, -15, 7);
    REQUIRE(quo == -3);
    mp_fdiv_q(quo, 15, -7);
    REQUIRE(quo == -3);
    mp_fdiv_q(quo, -15, -7);
    REQUIRE(quo == 2);

    mp_tdiv_qr(quo, rem, 15, 7);
    REQUIRE(quo == 2);
    REQUIRE(rem == 1);
    mp_tdiv_qr(quo, rem, -15, 7);
    REQUIRE(quo == -2);
    REQUIRE(rem == -1);
    mp_tdiv_qr(quo, rem, 15, -7);
    REQUIRE(quo == -2);
    REQUIRE(rem == 1);
    mp_tdiv_qr(quo, rem, -15, -7);
    REQUIRE(quo == 2);
    REQUIRE(rem == -1);

    // mp_divexact
    mp_divexact(res, -2, -1);
    REQUIRE(res == 2);
    mp_divexact(res, -1, -1);
    REQUIRE(res == 1);
    mp_divexact(res, 1, -1);
    REQUIRE(res == -1);
    mp_divexact(res, 2, -1);
    REQUIRE(res == -2);
    mp_divexact(res, -2, 1);
    REQUIRE(res == -2);
    mp_divexact(res, -1, 1);
    REQUIRE(res == -1);
    mp_divexact(res, 1, 1);
    REQUIRE(res == 1);
    mp_divexact(res, 2, 1);
    REQUIRE(res == 2);
    mp_divexact(res, -6, -3);
    REQUIRE(res == 2);
    mp_divexact(res, -6, -2);
    REQUIRE(res == 3);
    mp_divexact(res, -6, 2);
    REQUIRE(res == -3);
    mp_divexact(res, -6, 3);
    REQUIRE(res == -2);
    mp_divexact(res, 109187, -227);
    REQUIRE(res == -481);
    mp_divexact(res, -109187, 227);
    REQUIRE(res == -481);

    // mp_gcd
    mp_gcd(gcd, -1, -1);
    REQUIRE(gcd == 1);
    mp_gcd(gcd, -1, 0);
    REQUIRE(gcd == 1);
    mp_gcd(gcd, -1, 1);
    REQUIRE(gcd == 1);
    mp_gcd(gcd, 0, -1);
    REQUIRE(gcd == 1);
    mp_gcd(gcd, 0, 0);
    REQUIRE(gcd == 0);
    mp_gcd(gcd, 0, 1);
    REQUIRE(gcd == 1);
    mp_gcd(gcd, 1, -1);
    REQUIRE(gcd == 1);
    mp_gcd(gcd, 1, 0);
    REQUIRE(gcd == 1);
    mp_gcd(gcd, 1, -1);
    REQUIRE(gcd == 1);
    mp_gcd(gcd, 2, -1);
    REQUIRE(gcd == 1);
    mp_gcd(gcd, 2, 0);
    REQUIRE(gcd == 2);
    mp_gcd(gcd, 2, 1);
    REQUIRE(gcd == 1);
    mp_gcd(gcd, 2, 2);
    REQUIRE(gcd == 2);
    mp_gcd(gcd, -1, 2);
    REQUIRE(gcd == 1);
    mp_gcd(gcd, 0, 2);
    REQUIRE(gcd == 2);
    mp_gcd(gcd, 1, 2);
    REQUIRE(gcd == 1);
    mp_gcd(gcd, 2, 2);
    REQUIRE(gcd == 2);
    mp_gcd(gcd, 2, 6);
    REQUIRE(gcd == 2);
    mp_gcd(gcd, 6, 2);
    REQUIRE(gcd == 2);
    mp_gcd(gcd, -2, 6);
    REQUIRE(gcd == 2);
    mp_gcd(gcd, 6, -2);
    REQUIRE(gcd == 2);
    mp_gcd(gcd, 2, -6);
    REQUIRE(gcd == 2);
    mp_gcd(gcd, -6, 2);
    REQUIRE(gcd == 2);
    mp_gcd(gcd, -2, -6);
    REQUIRE(gcd == 2);
    mp_gcd(gcd, -6, -2);
    REQUIRE(gcd == 2);
    mp_gcd(gcd, 5, 6);
    REQUIRE(gcd == 1);
    mp_gcd(gcd, 6, 5);
    REQUIRE(gcd == 1);
    mp_gcd(gcd, -5, 6);
    REQUIRE(gcd == 1);
    mp_gcd(gcd, 6, -5);
    REQUIRE(gcd == 1);
    mp_gcd(gcd, 5, -6);
    REQUIRE(gcd == 1);
    mp_gcd(gcd, -6, 5);
    REQUIRE(gcd == 1);
    mp_gcd(gcd, -5, -6);
    REQUIRE(gcd == 1);
    mp_gcd(gcd, -6, -5);
    REQUIRE(gcd == 1);
    mp_gcd(gcd, 8, 12);
    REQUIRE(gcd == 4);
    mp_gcd(gcd, 12, 8);
    REQUIRE(gcd == 4);
    mp_gcd(gcd, -8, 12);
    REQUIRE(gcd == 4);
    mp_gcd(gcd, 12, -8);
    REQUIRE(gcd == 4);
    mp_gcd(gcd, 8, -12);
    REQUIRE(gcd == 4);
    mp_gcd(gcd, -12, 8);
    REQUIRE(gcd == 4);
    mp_gcd(gcd, -8, -12);
    REQUIRE(gcd == 4);
    mp_gcd(gcd, -12, -8);
    REQUIRE(gcd == 4);

    integer_class s, t;
    mp_gcdext(gcd, s, t, -1, -1);
    REQUIRE(gcd == 1);
    REQUIRE(s == 0);
    REQUIRE(t == -1);
    mp_gcdext(gcd, s, t, -1, 0);
    REQUIRE(gcd == 1);
    REQUIRE(s == -1);
    REQUIRE(t == 0);
    mp_gcdext(gcd, s, t, -1, 1);
    REQUIRE(gcd == 1);
    REQUIRE(s == 0);
    REQUIRE(t == 1);
    mp_gcdext(gcd, s, t, 0, -1);
    REQUIRE(gcd == 1);
    REQUIRE(s == 0);
    REQUIRE(t == -1);
    // mp_gcdext(gcd, s, t, 0, 0);
    // REQUIRE(gcd == 0); REQUIRE(s == 0); REQUIRE(t == 0); //fails for boostmp
    // but not gmp
    mp_gcdext(gcd, s, t, 0, 1);
    REQUIRE(gcd == 1);
    REQUIRE(s == 0);
    REQUIRE(t == 1);
    mp_gcdext(gcd, s, t, 1, -1);
    REQUIRE(gcd == 1);
    REQUIRE(s == 0);
    REQUIRE(t == -1);
    mp_gcdext(gcd, s, t, 1, 0);
    REQUIRE(gcd == 1);
    REQUIRE(s == 1);
    REQUIRE(t == 0);
    mp_gcdext(gcd, s, t, 1, -1);
    REQUIRE(gcd == 1);
    REQUIRE(s == 0);
    REQUIRE(t == -1);
    mp_gcdext(gcd, s, t, 2, -1);
    REQUIRE(gcd == 1);
    REQUIRE(s == 0);
    REQUIRE(t == -1);
    mp_gcdext(gcd, s, t, 2, 0);
    REQUIRE(gcd == 2);
    REQUIRE(s == 1);
    REQUIRE(t == 0);
    mp_gcdext(gcd, s, t, 2, 1);
    REQUIRE(gcd == 1);
    REQUIRE(s == 0);
    REQUIRE(t == 1);
    mp_gcdext(gcd, s, t, 2, 2);
    REQUIRE(gcd == 2);
    REQUIRE(s == 0);
    REQUIRE(t == 1);
    mp_gcdext(gcd, s, t, -1, 2);
    REQUIRE(gcd == 1);
    REQUIRE(s == -1);
    REQUIRE(t == 0);
    mp_gcdext(gcd, s, t, 0, 2);
    REQUIRE(gcd == 2);
    REQUIRE(s == 0);
    REQUIRE(t == 1);
    mp_gcdext(gcd, s, t, 1, 2);
    REQUIRE(gcd == 1);
    REQUIRE(s == 1);
    REQUIRE(t == 0);
    mp_gcdext(gcd, s, t, 2, 2);
    REQUIRE(gcd == 2);
    REQUIRE(s == 0);
    REQUIRE(t == 1);
    mp_gcdext(gcd, s, t, 2, 6);
    REQUIRE(gcd == 2);
    REQUIRE(s == 1);
    REQUIRE(t == 0);
    mp_gcdext(gcd, s, t, 6, 2);
    REQUIRE(gcd == 2);
    REQUIRE(s == 0);
    REQUIRE(t == 1);
    mp_gcdext(gcd, s, t, -2, 6);
    REQUIRE(gcd == 2);
    REQUIRE(s == -1);
    REQUIRE(t == 0);
    mp_gcdext(gcd, s, t, 6, -2);
    REQUIRE(gcd == 2);
    REQUIRE(s == 0);
    REQUIRE(t == -1);
    mp_gcdext(gcd, s, t, 2, -6);
    REQUIRE(gcd == 2);
    REQUIRE(s == 1);
    REQUIRE(t == 0);
    mp_gcdext(gcd, s, t, -6, 2);
    REQUIRE(gcd == 2);
    REQUIRE(s == 0);
    REQUIRE(t == 1);
    mp_gcdext(gcd, s, t, -2, -6);
    REQUIRE(gcd == 2);
    REQUIRE(s == -1);
    REQUIRE(t == 0);
    mp_gcdext(gcd, s, t, -6, -2);
    REQUIRE(gcd == 2);
    REQUIRE(s == 0);
    REQUIRE(t == -1);
    mp_gcdext(gcd, s, t, 5, 6);
    REQUIRE(gcd == 1);
    REQUIRE(s == -1);
    REQUIRE(t == 1);
    mp_gcdext(gcd, s, t, 6, 5);
    REQUIRE(gcd == 1);
    REQUIRE(s == 1);
    REQUIRE(t == -1);
    mp_gcdext(gcd, s, t, -5, 6);
    REQUIRE(gcd == 1);
    REQUIRE(s == 1);
    REQUIRE(t == 1);
    mp_gcdext(gcd, s, t, 6, -5);
    REQUIRE(gcd == 1);
    REQUIRE(s == 1);
    REQUIRE(t == 1);
    mp_gcdext(gcd, s, t, 5, -6);
    REQUIRE(gcd == 1);
    REQUIRE(s == -1);
    REQUIRE(t == -1);
    mp_gcdext(gcd, s, t, -6, 5);
    REQUIRE(gcd == 1);
    REQUIRE(s == -1);
    REQUIRE(t == -1);
    mp_gcdext(gcd, s, t, -5, -6);
    REQUIRE(gcd == 1);
    REQUIRE(s == 1);
    REQUIRE(t == -1);
    mp_gcdext(gcd, s, t, -6, -5);
    REQUIRE(gcd == 1);
    REQUIRE(s == -1);
    REQUIRE(t == 1);
    mp_gcdext(gcd, s, t, 8, 12);
    REQUIRE(gcd == 4);
    REQUIRE(s == -1);
    REQUIRE(t == 1);
    mp_gcdext(gcd, s, t, 12, 8);
    REQUIRE(gcd == 4);
    REQUIRE(s == 1);
    REQUIRE(t == -1);
    mp_gcdext(gcd, s, t, -8, 12);
    REQUIRE(gcd == 4);
    REQUIRE(s == 1);
    REQUIRE(t == 1);
    mp_gcdext(gcd, s, t, 12, -8);
    REQUIRE(gcd == 4);
    REQUIRE(s == 1);
    REQUIRE(t == 1);
    mp_gcdext(gcd, s, t, 8, -12);
    REQUIRE(gcd == 4);
    REQUIRE(s == -1);
    REQUIRE(t == -1);
    mp_gcdext(gcd, s, t, -12, 8);
    REQUIRE(gcd == 4);
    REQUIRE(s == -1);
    REQUIRE(t == -1);
    mp_gcdext(gcd, s, t, -8, -12);
    REQUIRE(gcd == 4);
    REQUIRE(s == 1);
    REQUIRE(t == -1);
    mp_gcdext(gcd, s, t, -12, -8);
    REQUIRE(gcd == 4);
    REQUIRE(s == -1);
    REQUIRE(t == 1);

    // mp_invert
    // behavior undefined when m == 0, so don't check those cases
    bool is_invertible;
    is_invertible = mp_invert(res, -1, -1);
    REQUIRE(is_invertible == 1);
    REQUIRE(res == 0);
    is_invertible = mp_invert(res, -1, 1);
    REQUIRE(is_invertible == 1);
    REQUIRE(res == 0);
    is_invertible = mp_invert(res, -1, 1);
    REQUIRE(is_invertible == 1);
    REQUIRE(res == 0);
    is_invertible = mp_invert(res, 2, 1);
    REQUIRE(is_invertible == 1);
    REQUIRE(res == 0);
    is_invertible = mp_invert(res, -1, 2);
    REQUIRE(is_invertible == 1);
    REQUIRE(res == 1);
    is_invertible = mp_invert(res, 3, 4);
    REQUIRE(is_invertible == 1);
    REQUIRE(res == 3);
    is_invertible = mp_invert(res, 2, 4);
    REQUIRE(is_invertible == 0);
    is_invertible = mp_invert(res, -14, 15);
    REQUIRE(is_invertible == 1);
    REQUIRE(res == 1);
    is_invertible = mp_invert(res, 39, -10);
    REQUIRE(is_invertible == 1);
    REQUIRE(res == 9);
    is_invertible = mp_invert(res, -10, 77);
    REQUIRE(is_invertible == 1);
    REQUIRE(res == 23);
    is_invertible = mp_invert(res, -77, -10);
    REQUIRE(is_invertible == 1);
    REQUIRE(res == 7);

    // mp_lcm
    mp_lcm(res, 18, 24);
    REQUIRE(res == 72);
}

TEST_CASE("primes: integer_class", "[integer_class]")
{
    // mp_probab_prime_p
    REQUIRE(mp_probab_prime_p(690089, 25));  // 690089 is prime
    REQUIRE(!mp_probab_prime_p(636751, 25)); // 636751 == 857*743

    // mp_nextprime
    integer_class res;
    mp_nextprime(res, integer_class(6000));
    REQUIRE(res == 6007);
}

TEST_CASE("fibonacci and lucas: integer_class", "[integer_class]")
{
    integer_class res, res2;
    mp_fib_ui(res, 6);
    REQUIRE(res == 8);
    mp_fib2_ui(res, res2, 9);
    REQUIRE(res == 34);
    REQUIRE(res2 == 21);

    mp_lucnum_ui(res, 0);
    REQUIRE(res == 2);
    mp_lucnum_ui(res, 1);
    REQUIRE(res == 1);
    mp_lucnum_ui(res, 6);
    REQUIRE(res == 18);
    mp_lucnum2_ui(res, res2, 8);
    REQUIRE(res == 47);
    REQUIRE(res2 == 29);
}

TEST_CASE("legendre: integer_class", "[integer_class]")
{
    integer_class prime(13619591); // 13619591 is a prime
    integer_class i = prime * integer_class(34714) + integer_class(81);
    REQUIRE(mp_legendre(i, prime) == 1);

    // mp_jacobi
    REQUIRE(mp_jacobi(i, prime) == 1);
    REQUIRE(mp_jacobi(30, 59) == -1);
    REQUIRE(mp_jacobi(30, 57) == 0);
    REQUIRE(mp_jacobi(17, 41) == -1);
    REQUIRE(mp_jacobi(18, 27) == 0);
    REQUIRE(mp_jacobi(1, 1) == 1);
    REQUIRE(mp_jacobi(28, 25) == 1);
    REQUIRE(mp_jacobi(13, 45) == -1);
    REQUIRE(mp_jacobi(13, 45) == -1);
    REQUIRE(mp_jacobi(24, 33) == 0);
    REQUIRE(mp_jacobi(20, 39) == 1);
    REQUIRE(mp_jacobi(12, 51) == 0);
    REQUIRE(mp_jacobi(30, 7) == 1);
    REQUIRE(mp_jacobi(27, 11) == 1);
    REQUIRE(mp_jacobi(14, 3) == -1);
    REQUIRE(mp_jacobi(-58, 3) == -1);
    REQUIRE(mp_jacobi(-58, 9) == 1);
    REQUIRE(mp_jacobi(-20, 11) == -1);
    REQUIRE(mp_jacobi(-1, 1) == 1);
    REQUIRE(mp_jacobi(0, 17) == 0);
    REQUIRE(mp_jacobi(1, 1) == 1);
    REQUIRE(mp_jacobi(1, 15) == 1);
    REQUIRE(mp_jacobi(5, 15) == 0);
    REQUIRE(mp_jacobi(36, 25) == 1);
    REQUIRE(mp_jacobi(46, 35) == 1);

    // mp_kronecker
    REQUIRE(mp_kronecker(-58, -58) == 0);
    REQUIRE(mp_kronecker(-58, -43) == 1);
    REQUIRE(mp_kronecker(-20, -35) == 0);
    REQUIRE(mp_kronecker(-1, -32) == -1);
    REQUIRE(mp_kronecker(0, -1) == 1);
    REQUIRE(mp_kronecker(1, 1) == 1);
    REQUIRE(mp_kronecker(1, 3) == 1);
    REQUIRE(mp_kronecker(5, 18) == -1);
    REQUIRE(mp_kronecker(36, 50) == 0);
    REQUIRE(mp_kronecker(46, 79) == 1);
}

TEST_CASE("misc: integer_class", "[integer_class]")
{
    // mp_abs
    REQUIRE(mp_abs(integer_class(2)) == 2);
    REQUIRE(mp_abs(integer_class(-2)) == 2);
    REQUIRE(mp_abs(integer_class(0)) == 0);

    // mp_sign
    REQUIRE(mp_sign(integer_class(-2)) < 0);
    REQUIRE(mp_sign(integer_class(2)) > 0);
    REQUIRE(mp_sign(integer_class(0)) == 0);

    // mp_get_ui
    REQUIRE(mp_get_ui(integer_class("-18446744073709551616")) == 0u);
    REQUIRE(mp_get_ui(integer_class("-18446744073709551615"))
            == 18446744073709551615u);
    REQUIRE(mp_get_ui(integer_class("-9223372036854775809"))
            == 9223372036854775809u);
    REQUIRE(mp_get_ui(integer_class("-9223372036854775808"))
            == 9223372036854775808u);
    REQUIRE(mp_get_ui(integer_class("-2")) == 2u);
    REQUIRE(mp_get_ui(integer_class("-1")) == 1u);
    REQUIRE(mp_get_ui(integer_class("0")) == 0u);
    REQUIRE(mp_get_ui(integer_class("1")) == 1u);
    REQUIRE(mp_get_ui(integer_class("2")) == 2u);
    REQUIRE(mp_get_ui(integer_class("9223372036854775807"))
            == 9223372036854775807u);
    REQUIRE(mp_get_ui(integer_class("-9223372036854775807"))
            == 9223372036854775807u);
    REQUIRE(mp_get_ui(integer_class("18446744073709551615"))
            == 18446744073709551615u);
    REQUIRE(mp_get_ui(integer_class("18446744073709551615"))
            == 18446744073709551615u);
    REQUIRE(mp_get_ui(integer_class("18446744073709551616")) == 0u);

    // mp_scan1
    REQUIRE(mp_scan1(LONG_MIN) == 63);
    REQUIRE(mp_scan1(-1024) == 10);
    REQUIRE(mp_scan1(-768) == 8);
    REQUIRE(mp_scan1(-500) == 2);
    REQUIRE(mp_scan1(-5) == 0);
    REQUIRE(mp_scan1(-4) == 2);
    REQUIRE(mp_scan1(-2) == 1);
    REQUIRE(mp_scan1(-1) == 0);
    REQUIRE(mp_scan1(0) == ULONG_MAX);
    REQUIRE(mp_scan1(1) == 0);
    REQUIRE(mp_scan1(2) == 1);
    REQUIRE(mp_scan1(4) == 2);
    REQUIRE(mp_scan1(5) == 0);
    REQUIRE(mp_scan1(500) == 2);
    REQUIRE(mp_scan1(768) == 8);
    REQUIRE(mp_scan1(1024) == 10);
    REQUIRE(mp_scan1(LONG_MAX) == 0);

    // mp_and
    integer_class res;
    mp_and(res, 9, 12);
    REQUIRE(res == 8);

    // fits
    integer_class long_max(LONG_MAX);
    integer_class past_long_max(long_max + 1);
    integer_class ulong_max(ULONG_MAX);
    integer_class past_ulong_max(ulong_max + 1);
    REQUIRE(mp_fits_slong_p(long_max));
    REQUIRE(!mp_fits_slong_p(past_long_max));
    REQUIRE(mp_fits_ulong_p(long_max));
    REQUIRE(!mp_fits_ulong_p(past_ulong_max));

    // mp_fac_ui
    mp_fac_ui(res, 5);
    REQUIRE(res == 120);

    // mp_bin_ui
    mp_bin_ui(res, 5, 2);
    REQUIRE(res == 10);
    mp_bin_ui(res, integer_class(30), 8);
    REQUIRE(res == 5852925);

    // add_mul
    integer_class i = 3;
    mp_addmul(i, 4, 5);
    REQUIRE(i == 23);

    // rational_class
    auto r = rational_class(2, 3);
    REQUIRE(get_num(r) == 2);
    REQUIRE(get_den(r) == 3);
    r = rational_class(2, 3);
    REQUIRE(mp_sign(r) > 0);
    r = rational_class(-2, 3);
    REQUIRE(mp_sign(r) < 0);
    r = rational_class(0, 1);
    REQUIRE(mp_sign(r) == 0);

    r = rational_class(-1, 2);
    REQUIRE(mp_abs(r) == .5);

    r = rational_class(2, 3);
    rational_class p;
    mp_pow_ui(p, r, 4);
    REQUIRE(get_num(p) == 16);
    REQUIRE(get_den(p) == 81);
}
