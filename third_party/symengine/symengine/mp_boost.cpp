#include "mp_class.h"
#include <boost/multiprecision/miller_rabin.hpp>
#include <boost/mpl/int.hpp>
#include <utility>
#include <cmath>
#include <symengine/symengine_assert.h>
#include <symengine/symengine_exception.h>
#include <symengine/prime_sieve.h>

using boost::multiprecision::denominator;
using boost::multiprecision::miller_rabin_test;
using boost::multiprecision::numerator;
using boost::multiprecision::detail::find_lsb;

#if SYMENGINE_INTEGER_CLASS == SYMENGINE_BOOSTMP

namespace SymEngine
{

integer_class pow(const integer_class &a, unsigned long b)
{
    return boost::multiprecision::pow(a, numeric_cast<unsigned>(b));
}

void mp_fdiv_qr(integer_class &q, integer_class &r, const integer_class &a,
                const integer_class &b)
{
    /*boost::multiprecision doesn't have a built-in fdiv_qr (floored division).
      Its divide_qr uses truncated division, as does its
      modulus operator. Thus, using boost::multiprecision::divide_qr we get:
      divide_qr(-5, 3, quo, rem) //quo == -1, rem == -2
      divide_qr(5, -3, quo, rem) //quo == -1, rem == 2
      but we want:
      mp_fdiv_r(quo, rem, -5, 3) //quo == -2, rem == 1
      mp_fdiv_r(quo, rem, 5, -3) //rem == -2, rem == -1
      The problem arises only when the quotient is negative.  To convert
      a truncated result into a floored result in this case, simply subtract
      one from the truncated quotient and add the divisor to the truncated
      remainder.
      */

    // must copy a and b before calling divide_qr because a or b may refer to
    // the same
    // object as q or r, causing incorrect results
    integer_class a_cpy = a, b_cpy = b;
    bool neg_quotient = ((a < 0 && b > 0) || (a > 0 && b < 0)) ? true : false;
    boost::multiprecision::divide_qr(a_cpy, b_cpy, q, r);
    // floor the quotient if necessary
    if (neg_quotient && r != 0) {
        q -= 1;
    }
    // remainder should have same sign as divisor
    if ((b_cpy > 0 && r < 0) || (b_cpy < 0 && r > 0)) {
        r += b_cpy;
        return;
    }
}

void mp_cdiv_qr(integer_class &q, integer_class &r, const integer_class &a,
                const integer_class &b)
{
    integer_class a_cpy = a, b_cpy = b;
    bool pos_quotient = ((a < 0 && b < 0) || (a > 0 && b > 0)) ? true : false;
    boost::multiprecision::divide_qr(a_cpy, b_cpy, q, r);
    // ceil the quotient if necessary
    if (pos_quotient && r != 0) {
        q += 1;
    }
    // remainder should have opposite sign as divisor
    if ((b_cpy > 0 && r > 0) || (b_cpy < 0 && r < 0)) {
        r -= b_cpy;
        return;
    }
}

void mp_gcdext(integer_class &gcd, integer_class &s, integer_class &t,
               const integer_class &a, const integer_class &b)
{

    integer_class this_s(1);
    integer_class this_t(0);
    integer_class next_s(0);
    integer_class next_t(1);
    integer_class this_r(a);
    integer_class next_r(b);
    integer_class q;
    while (next_r != 0) {
        // should use truncated division, so use
        // boost::multiprecision::divide_qr
        // beware of overwriting this_r during internal operations of divide_qr
        // copy it first
        integer_class this_r_cpy = this_r;
        boost::multiprecision::divide_qr(this_r_cpy, next_r, q, this_r);
        this_s -= q * next_s;
        this_t -= q * next_t;
        std::swap(this_s, next_s);
        std::swap(this_t, next_t);
        std::swap(this_r, next_r);
    }
    // normalize the gcd, s and t
    if (this_r < 0) {
        this_r *= -1;
        this_s *= -1;
        this_t *= -1;
    }
    gcd = std::move(this_r);
    s = std::move(this_s);
    t = std::move(this_t);
}

bool mp_invert(integer_class &res, const integer_class &a,
               const integer_class &m)
{
    integer_class gcd, s, t;
    mp_gcdext(gcd, s, t, a, m);
    if (gcd != 1) {
        res = 0;
        return false;
    } else {
        mp_fdiv_r(s, s, m); // reduce s modulo m.  undefined behavior when m ==
                            // 0, so don't need to check
        if (s < 0) {
            s += mp_abs(m);
        } // give the canonical representative of s
        res = s;
        return true;
    }
}

// floored modulus
integer_class fmod(const integer_class &a, const integer_class &mod)
{
    integer_class res = a % mod;
    if (res < 0) {
        res += mod;
    }
    return res;
}

void mp_pow_ui(rational_class &res, const rational_class &i, unsigned long n)
{
    integer_class num = numerator(i);   // copy
    integer_class den = denominator(i); // copy
    num = pow(num, n);
    den = pow(den, n);
    res = rational_class(std::move(num), std::move(den));
}

void mp_powm(integer_class &res, const integer_class &base,
             const integer_class &exp, const integer_class &m)
{
    // if exp is negative, interpret as follows
    // base**(exp) mod m 	== (base**(-1))**abs(exp) mod m
    // 						== (base**(-1) mod m) ** abs(exp) mod m
    // where base**(-1) mod m is the modular inverse
    if (exp < 0) {
        integer_class base_inverse;
        if (!mp_invert(base_inverse, base, m)) {
            throw SymEngine::SymEngineException("negative exponent undefined "
                                                "in powm if base is not "
                                                "invertible mod m");
        }
        res = boost::multiprecision::powm(base_inverse, mp_abs(exp), m);
        return;
    } else {
        res = boost::multiprecision::powm(base, exp, m);
        // boost's powm calculates base**exp % m, but uses truncated
        // modulus, e.g. powm(-2,3,5) == -3.  We want powm(-2,3,5) == 2
        if (res < 0) {
            res += m;
        }
    }
}

integer_class step(const unsigned long &n, const integer_class &i,
                   integer_class &x)
{
    SYMENGINE_ASSERT(n > 1);
    unsigned long m = n - 1;
    integer_class &&x_m = pow(x, m);
    return integer_class((integer_class(m * x) + integer_class(i / x_m)) / n);
}

bool positive_root(integer_class &res, const integer_class &i,
                   const unsigned long n)
{
    integer_class x
        = 1; // TODO: make a better starting guess based on (number of bits)/n
    integer_class y = step(n, i, x);
    do {
        x = y;
        y = step(n, i, x);
    } while (y < x);
    res = x;
    if (pow(x, n) == i) {
        return true;
    }
    return false;
}

// return true if i is a perfect nth power, i.e. res**i == n
bool mp_root(integer_class &res, const integer_class &i, const unsigned long n)
{
    if (n == 0) {
        throw std::runtime_error("0th root is undefined");
    }
    if (n == 1) {
        res = i;
        return true;
    }
    if (i == 0) {
        res = 0;
        return true;
    }
    if (i > 0) {
        return positive_root(res, i, n);
    }
    if (i < 0 && (n % 2 == 0)) {
        throw std::runtime_error("even root of a negative is non-real");
    }
    bool b = positive_root(res, -i, n);
    res *= -1;
    return b;
}

integer_class mp_sqrt(const integer_class &i)
{
    // as of 11/1/2016, boost::multiprecision::sqrt() is buggy:
    // https://svn.boost.org/trac/boost/ticket/12559
    // implement with mp_root for now
    integer_class res;
    mp_root(res, i, 2);
    return res;
}

void mp_rootrem(integer_class &a, integer_class &b, const integer_class &i,
                unsigned long n)
{
    mp_root(a, i, n);
    integer_class p = pow(a, n);
    ;
    b = i - p;
}

void mp_sqrtrem(integer_class &a, integer_class &b, const integer_class &i)
{
    a = mp_sqrt(i);
    b = i - boost::multiprecision::pow(a, 2);
}

// return nonzero if i is probably prime.
int mp_probab_prime_p(const integer_class &i, unsigned retries)
{
    if (i % 2 == 0)
        return (i == 2);
    return miller_rabin_test(i, retries);
}

void mp_nextprime(integer_class &res, const integer_class &i)
{
    // simple implementation:  just check all odds bigger than i for primality
    if (i < 2) {
        res = 2;
        return;
    }
    integer_class candidate;
    candidate = (i % 2 == 0) ? i + 1 : i + 2;
    // Knuth recommends 25 trials for a pretty strong likelihood that candidate
    // is prime
    while (!mp_probab_prime_p(candidate, 25)) {
        candidate += 2;
    }
    res = std::move(candidate);
}

unsigned long mp_scan1(const integer_class &i)
{
    if (i == 0) {
        return ULONG_MAX;
    }
    return find_lsb(i, {});
}

// define simple 2x2 matrix with exponentiation by repeated squaring
// to use in logarithmic-time fibonacci calculation

struct two_by_two_matrix {
    integer_class data[2][2]; // data[1][0] is row 1, column 0 entry of matrix

    two_by_two_matrix(integer_class a, integer_class b, integer_class c,
                      integer_class d)
        : data{{a, b}, {c, d}}
    {
    }
    two_by_two_matrix() : data{{0, 0}, {0, 0}} {}
    two_by_two_matrix(const two_by_two_matrix &other) = default;
    two_by_two_matrix &operator=(const two_by_two_matrix &other)
    {
        this->data[0][0] = other.data[0][0];
        this->data[0][1] = other.data[0][1];
        this->data[1][0] = other.data[1][0];
        this->data[1][1] = other.data[1][1];
        return *this;
    }
    two_by_two_matrix operator*(const two_by_two_matrix &other)
    {
        two_by_two_matrix res;
        res.data[0][0] = this->data[0][0] * other.data[0][0]
                         + this->data[0][1] * other.data[1][0];
        res.data[0][1] = this->data[0][0] * other.data[0][1]
                         + this->data[0][1] * other.data[1][1];
        res.data[1][0] = this->data[1][0] * other.data[0][0]
                         + this->data[1][1] * other.data[1][0];
        res.data[1][1] = this->data[1][0] * other.data[0][1]
                         + this->data[1][1] * other.data[1][1];
        return res;
    }
    // recursive repeated squaring
    two_by_two_matrix pow(unsigned long n)
    {
        if (n == 0) {
            return two_by_two_matrix::identity();
        }
        if (n == 1) {
            return *this;
        }
        if (n == 2) {
            return (*this) * (*this);
        }
        if (n % 2 == 0) {
            return (this->pow(n / 2)).pow(2);
        }
        return ((this->pow((n - 1) / 2)).pow(2)) * (*this);
    }

    static two_by_two_matrix identity()
    {
        return two_by_two_matrix(1, 0, 0, 1);
    }
};

inline two_by_two_matrix fib_matrix(unsigned long n)
{
    two_by_two_matrix x(1, 1, 1, 0);
    return x.pow(n);
}

void mp_fib_ui(integer_class &res, unsigned long n)
{
    // reference: https://www.nayuki.io/page/fast-fibonacci-algorithms
    res = fib_matrix(n).data[0][1];
}

// sets a = Fibonacci(n) and b = Fibonacci(n-1)
void mp_fib2_ui(integer_class &a, integer_class &b, unsigned long n)
{
    // reference: https://www.nayuki.io/page/fast-fibonacci-algorithms
    two_by_two_matrix result_matrix = fib_matrix(n);
    a = result_matrix.data[0][1];
    b = result_matrix.data[1][1];
}

inline two_by_two_matrix luc_matrix(unsigned long n)
{
    two_by_two_matrix multiplier(1, 1, 1, 0);
    two_by_two_matrix start(1, 0, 2, 0);
    return multiplier.pow(n) * start;
}

void mp_lucnum_ui(integer_class &res, unsigned long n)
{
    // implementation based on the following fact:
    // [[1,1],[1,0]]^(n-1)*[2,0,1,0] = [[L(n+1),0][L(n),0]]
    // where L(n) is the nth Lucas number
    res = luc_matrix(n).data[1][0];
}

void mp_lucnum2_ui(integer_class &a, integer_class &b, unsigned long n)
{
    if (n == 0) {
        throw std::runtime_error("index of lucas number cannot be negative");
    }
    two_by_two_matrix result_matrix = luc_matrix(n - 1);
    a = result_matrix.data[0][0];
    b = result_matrix.data[1][0];
}

void mp_fac_ui(integer_class &res, unsigned long n)
{
    // couldn't make boost's template version of factorial work,
    // so implement slow, naive version for now
    res = 1;
    for (unsigned long i = 2; i <= n; ++i) {
        res *= i;
    }
}

void mp_bin_ui(integer_class &res, const integer_class &n, unsigned long r)
{
    // slow, naive implementation
    integer_class x = n - r;
    res = 1;
    for (unsigned long i = 1; i <= r; ++i) {
        res *= x + i;
        res /= i;
    }
}

// this is extremely slow!
bool mp_perfect_power_p(const integer_class &i)
{
    if (i == 0 || i == 1 || i == -1) {
        return true;
    }
    // if i == a**k, with k == pq for some integers p,q
    // then i == a**(pq) == (a**p)**q == (a**q)**p
    // hence i is a pth power and a qth power
    // Hence if i == p**k, then i is a pth power for any p
    // in the prime factorization of k, and it suffices
    // to check whether i is a prime power.

    // the largest possible prime p would arise from the
    // case where i is a prime power of two
    // so check all prime roots up to log(i) base 2.

    unsigned long max = std::ilogb(i.convert_to<double>());

    // treat case p=2 separately b/c mp_root throws exception
    // with an even root of a negative
    if (mp_perfect_square_p(i)) {
        return true;
    }
    integer_class p(2);
    integer_class root(0);
    while (true) {
        mp_nextprime(p, p);
        if (p > max) {
            return false;
        }
        if (mp_root(root, i, p.convert_to<unsigned long>())) {
            return true;
        }
    }
}

bool mp_perfect_square_p(const integer_class &i)
{
    if (i < 0) {
        return false;
    }
    integer_class root;
    return mp_root(root, i, 2);
}

integer_class mp_primorial(unsigned long n)
{
    integer_class res = 1;
    Sieve::iterator pi(static_cast<unsigned>(n));
    unsigned int p;
    while ((p = pi.next_prime()) <= n) {
        res *= p;
    }
    return res;
}

// according to the gmp documentation, the behavior of the
// corresponding function mpz_legendre is
// undefined if n is not a positive odd prime.
// hence we treat n as though it were a positive odd prime
// but it is up to the caller to very this.
int mp_legendre(const integer_class &a, const integer_class &n)
{
    integer_class res;
    mp_powm(res, a, integer_class((n - 1) / 2), n);
    return res <= 1 ? res.convert_to<int>() : -1;
}

// private function that computes jacobi symbols
// without checking that arguments satisfy a >= 0
// and n is odd.
int unchecked_jacobi(const integer_class &a, const integer_class &n)
{
    // https://en.wikipedia.org/wiki/Jacobi_symbol#Calculating_the_Jacobi_symbol
    if (a == 1) {
        return 1;
    }
    integer_class num = a;
    integer_class den = n;
    // (1) Reduce the "numerator" modulo the "denominator"
    num = fmod(num, den);

    // (2) Extract any factors of 2 from the "numerator"
    unsigned long factors_of_two = 0;
    while (num % 2 == 0 && num != 0) {
        num /= 2; // use a shift instead of division here? faster?
        ++factors_of_two;
    }
    int product_of_twos = 1; // (2 | den)**factors_of_two

    // (2 | den) is -1 iff den % 8 = 3 or den % 8 = 5.
    // If factors_of_two is odd and (2 | den) is -1,
    // then (2 | den)**factors_of_two is -1.
    // Otherwise, (2 | den)**factors_of_two is 1.
    int den_mod_8 = fmod(den, 8).convert_to<int>();
    if ((factors_of_two % 2 == 1) && (den_mod_8 == 3 || den_mod_8 == 5)) {
        product_of_twos = -1;
    }

    // (3) If the remaining "numerator" (after extraction of twos) is 1,
    // then the remaining jacobi symbol is 1.
    // If the "numerator" and "denominator" are not coprime, result is 0.
    if (num == 1) {
        return product_of_twos;
    }
    if (boost::multiprecision::gcd(num, den) != 1) {
        return 0;
    }

    // (4) Otherwise, the "numerator" and "denominator" are now odd
    // positive coprime integers, so we can flip the symbol
    // with quadratic reciprocity, then return to step (1)

    // (num | den) == (den | num), unless num % 4 and den %4 are both
    // 3, in which case (num | den) == (-1) * (den | num)
    int quadratic_reciprocity_factor = 1;
    if ((fmod(num, 4) == 3) && (fmod(den, 4) == 3)) {
        quadratic_reciprocity_factor = -1;
    }
    return product_of_twos * quadratic_reciprocity_factor
           * unchecked_jacobi(den, num);
}

// public interface for computing jacobi symbols.  performs checking.
int mp_jacobi(const integer_class &a, const integer_class &n)
{
    if (n < 0) {
        throw std::runtime_error("jacobi denominator must be positive");
    }
    if (n % 2 == 0) {
        throw std::runtime_error("jacobi denominator must be odd");
    }
    return unchecked_jacobi(a, n);
}

int mp_kronecker(const integer_class &a, const integer_class &n)
{
    /*
    https://en.wikipedia.org/wiki/Kronecker_symbol
    We compute the Kronecker symbol in terms of the Jacobi symbol.
    For an integer n!=0, let n = u * PRODUCT (p_i**e_i), where u = -1 or 1 and
    the p_i's and e_i's are the primes and corresponding powers
    in the prime factorization of |n|.
    Then for an integer a, the Kronecker symbol is given by
    (a | n) = (a | u) * PRODUCT [(a | p_i)**e_i]
    where
    (a | u) is -1 if u is -1 and a < 0. Otherwise it is 1.
    (a | 2) is 0 if a is even, 1 if (a mod 8) == 1 or 7, and -1 if (a mod 8) ==
    3 or 5.
    (a | p) is the Legendre symbol if p is an odd prime.

    Let j be the power of the prime 2 in n's prime factorization, and let
    m = |n|/(2**j) -- that is, m is |n| with all factors of 2 extracted.

    Notice that, if p_1 is 2, then
    PRODUCT [(a | p_i)**e_i]
    == (a | 2)**j * PRODUCT [(a | p_i)**e_i]	here the p_i's are the remaining
    (odd) prime factors
    == (a | 2)**j * (a | m), 					here (a | m) is the Jacobi
    symbol, because m>0 is odd

    Thus,
    (a | n) == (a | u) * (a | 2)**j * (a | m)	if n is even
    (a | n) == (a | u) * (a | m) 				if n is odd
    */

    if (n == 0) {
        throw std::runtime_error("second arg of Kronecker cannot be zero");
    }

    // Compute (a | u)
    int kr_a_u = 1;
    if (n.sign() == -1 && a < 0) {
        kr_a_u = -1;
    }

    // Compute m, j
    integer_class m = boost::multiprecision::abs(n);
    unsigned long j = 0;
    while (m % 2 == 0 && m != 0) { // while m is even
        m /= 2;                    // implement with shift?  faster?
        ++j;
    }

    // if n is even, compute (a | 2)**j
    int kr_a_2 = 0;
    int kr_a_2_to_j = 0;
    int a_mod_8 = fmod(a, 8).convert_to<int>();

    if (a % 2 != 0) {
        kr_a_2 = (a_mod_8 == 1 || a_mod_8 == 7) ? 1 : -1;
        kr_a_2_to_j = (kr_a_2 == -1 && (j % 2 != 0))
                          ? -1
                          : 1; //(-1)**odd == -1; (-1)**even=(1)**integer=1
    }

    if (n % 2 == 0) {
        return kr_a_u * kr_a_2_to_j * unchecked_jacobi(a, m);
    } else {
        return kr_a_u * unchecked_jacobi(a, m);
    }
}

} // namespace SymEngine

#endif // SYMENGINE_INTEGER_CLASS == SYMENGINE_BOOSTMP
