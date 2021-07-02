/**
 *  \file ntheory.h
 *  Basic number theory functions
 *
 **/

#ifndef SYMENGINE_NTHEORY_H
#define SYMENGINE_NTHEORY_H

#include <symengine/integer.h>

namespace SymEngine
{

// Prime Functions
//! Probabilistic Prime
int probab_prime_p(const Integer &a, unsigned reps = 25);
//! \return next prime after `a`
RCP<const Integer> nextprime(const Integer &a);

// Basic Number-theoretic functions
//! Greatest Common Divisor
RCP<const Integer> gcd(const Integer &a, const Integer &b);
//! Least Common Multiple
RCP<const Integer> lcm(const Integer &a, const Integer &b);
//! Extended GCD
void gcd_ext(const Ptr<RCP<const Integer>> &g, const Ptr<RCP<const Integer>> &s,
             const Ptr<RCP<const Integer>> &t, const Integer &a,
             const Integer &b);
//! modulo round toward zero
RCP<const Integer> mod(const Integer &n, const Integer &d);
//! \return quotient round toward zero when `n` is divided by `d`
RCP<const Integer> quotient(const Integer &n, const Integer &d);
//! \return modulo and quotient round toward zero
void quotient_mod(const Ptr<RCP<const Integer>> &q,
                  const Ptr<RCP<const Integer>> &r, const Integer &a,
                  const Integer &b);
//! modulo round toward -inf
RCP<const Integer> mod_f(const Integer &n, const Integer &d);
//! \return quotient round toward -inf when `n` is divided by `d`
RCP<const Integer> quotient_f(const Integer &n, const Integer &d);
//! \return modulo and quotient round toward -inf
void quotient_mod_f(const Ptr<RCP<const Integer>> &q,
                    const Ptr<RCP<const Integer>> &r, const Integer &a,
                    const Integer &b);
//! inverse modulo
int mod_inverse(const Ptr<RCP<const Integer>> &b, const Integer &a,
                const Integer &m);

//! Chinese remainder function. Return true when a solution exists.
bool crt(const Ptr<RCP<const Integer>> &R,
         const std::vector<RCP<const Integer>> &rem,
         const std::vector<RCP<const Integer>> &mod);

//! Fibonacci number
RCP<const Integer> fibonacci(unsigned long n);

//! Fibonacci n and n-1
void fibonacci2(const Ptr<RCP<const Integer>> &g,
                const Ptr<RCP<const Integer>> &s, unsigned long n);

//! Lucas number
RCP<const Integer> lucas(unsigned long n);

//! Lucas number n and n-1
void lucas2(const Ptr<RCP<const Integer>> &g, const Ptr<RCP<const Integer>> &s,
            unsigned long n);

//! Binomial Coefficient
RCP<const Integer> binomial(const Integer &n, unsigned long k);

//! Factorial
RCP<const Integer> factorial(unsigned long n);

//! \return true if `b` divides `a`
bool divides(const Integer &a, const Integer &b);

//! Factorization
//! \param B1 is only used when `n` is factored using gmp-ecm
int factor(const Ptr<RCP<const Integer>> &f, const Integer &n, double B1 = 1.0);

//! Factor using trial division.
//! \return 1 if a non-trivial factor is found, otherwise 0.
int factor_trial_division(const Ptr<RCP<const Integer>> &f, const Integer &n);

//! Factor using lehman's methods
int factor_lehman_method(const Ptr<RCP<const Integer>> &f, const Integer &n);

//! Factor using Pollard's p-1 method
int factor_pollard_pm1_method(const Ptr<RCP<const Integer>> &f,
                              const Integer &n, unsigned B = 10,
                              unsigned retries = 5);

//! Factor using Pollard's rho methods
int factor_pollard_rho_method(const Ptr<RCP<const Integer>> &f,
                              const Integer &n, unsigned retries = 5);

//! Find prime factors of `n`
void prime_factors(std::vector<RCP<const Integer>> &primes, const Integer &n);
//! Find multiplicities of prime factors of `n`
void prime_factor_multiplicities(map_integer_uint &primes, const Integer &n);
// Sieve class stores all the primes upto a limit. When a prime or a list of
// prime
// is requested, if the prime is not there in the sieve, it is extended to hold
// that
// prime. The implementation is a very basic Eratosthenes sieve, but the code
// should
// be quite optimized. For limit=1e8, it is about 20x slower than the
// `primesieve` library (1206ms vs 55.63ms).
class Sieve
{

private:
    static std::vector<unsigned> _primes;
    static void _extend(unsigned limit);
    static unsigned _sieve_size;
    static bool _clear;

public:
    // Returns all primes up to the `limit` (including). The vector `primes`
    // should
    // be empty on input and it will be filled with the primes.
    //! \param primes: holds all primes up to the `limit` (including).
    static void generate_primes(std::vector<unsigned> &primes, unsigned limit);
    // Clear the array of primes stored
    static void clear();
    // Set the sieve size in kilobytes. Set it to L1d cache size for best
    // performance.
    // Default value is 32.
    static void set_sieve_size(unsigned size);
    // Set whether the sieve is cleared after the sieve is extended in internal
    // functions
    static void set_clear(bool clear);

    class iterator
    {

    private:
        unsigned _index;
        unsigned _limit;

    public:
        // Iterator that generates primes upto limit
        iterator(unsigned limit);
        // Iterator that generates primes with no limit.
        iterator();
        // Destructor
        ~iterator();
        // Next prime
        unsigned next_prime();
    };
};

//! Computes the Bernoulli number Bn as an exact fraction, for an isolated
//! integer n
RCP<const Number> bernoulli(unsigned long n);
//! Computes the sum of the inverses of the first perfect mth powers
RCP<const Number> harmonic(unsigned long n, long m = 1);
//! Computes a primitive root. Returns false if no primitive root exists.
// Primitive root calculated is the smallest when n is prime.
bool primitive_root(const Ptr<RCP<const Integer>> &g, const Integer &n);
//! Computes all primitive roots less than n. Returns false if no primitive root
//! exists.
void primitive_root_list(std::vector<RCP<const Integer>> &roots,
                         const Integer &n);
//! Euler's totient function
RCP<const Integer> totient(const RCP<const Integer> &n);
//! Carmichael function
RCP<const Integer> carmichael(const RCP<const Integer> &n);
//! Multiplicative order. Return false if order does not exist
bool multiplicative_order(const Ptr<RCP<const Integer>> &o,
                          const RCP<const Integer> &a,
                          const RCP<const Integer> &n);
//! Legendre Function
int legendre(const Integer &a, const Integer &n);
//! Jacobi Function
int jacobi(const Integer &a, const Integer &n);
//! Kronecker Function
int kronecker(const Integer &a, const Integer &n);
//! All Solutions to x**n == a mod m. Return false if none exists.
void nthroot_mod_list(std::vector<RCP<const Integer>> &roots,
                      const RCP<const Integer> &a, const RCP<const Integer> &n,
                      const RCP<const Integer> &m);
//! A solution to x**n == a mod m. Return false if none exists.
bool nthroot_mod(const Ptr<RCP<const Integer>> &root,
                 const RCP<const Integer> &a, const RCP<const Integer> &n,
                 const RCP<const Integer> &m);
//! A solution to x**s == a**r mod m where b = r / s. Return false if none
//! exists.
bool powermod(const Ptr<RCP<const Integer>> &powm, const RCP<const Integer> &a,
              const RCP<const Number> &b, const RCP<const Integer> &m);
//! All solutions to x**s == a**r mod m where b = r / s. Return false if none
//! exists.
void powermod_list(std::vector<RCP<const Integer>> &pows,
                   const RCP<const Integer> &a, const RCP<const Number> &b,
                   const RCP<const Integer> &m);

//! Finds all Quadratic Residues of a Positive Integer
vec_integer_class quadratic_residues(const Integer &a);

//! Returns true if 'a' is a quadratic residue of 'p'
bool is_quad_residue(const Integer &a, const Integer &p);
//! Returns true if 'a' is a nth power residue of 'mod'
bool is_nth_residue(const Integer &a, const Integer &n, const Integer &mod);
//! Mobius Function
// mu(n) = 1 if n is a square-free positive integer with an even number of prime
// factors
// mu(n) = −1 if n is a square-free positive integer with an odd number of prime
// factors
// mu(n) = 0 if n has a squared prime factor
int mobius(const Integer &a);
// Mertens Function
// mertens(n) -> Sum of mobius(i) for i from 1 to n
long mertens(const unsigned long a);
}
#endif
