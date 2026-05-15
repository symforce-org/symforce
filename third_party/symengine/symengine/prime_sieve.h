#ifndef SYMENGINE_PRIME_SIEVE_H
#define SYMENGINE_PRIME_SIEVE_H

#include <vector>
#include <symengine/symengine_config.h>

// Sieve class stores all the primes upto a limit. When a prime or a list of
// prime
// is requested, if the prime is not there in the sieve, it is extended to hold
// that
// prime. The implementation is a very basic Eratosthenes sieve, but the code
// should
// be quite optimized. For limit=1e8, it is about 20x slower than the
// `primesieve` library (1206ms vs 55.63ms).

namespace SymEngine
{

class Sieve
{

private:
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

}; // namespace SymEngine

#endif
