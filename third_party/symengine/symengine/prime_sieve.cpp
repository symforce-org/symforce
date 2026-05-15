#include <symengine/prime_sieve.h>
#include <ciso646>
#include <cmath>
#include <valarray>
#include <algorithm>
#include <vector>
#include <iterator>
#ifdef HAVE_SYMENGINE_PRIMESIEVE
#include <primesieve.hpp>
#endif

namespace SymEngine
{

static std::vector<unsigned> &sieve_primes()
{
    static std::vector<unsigned> primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    return primes;
}

bool Sieve::_clear = true;
unsigned Sieve::_sieve_size = 32 * 1024 * 8; // 32K in bits

void Sieve::set_clear(bool clear)
{
    _clear = clear;
}

void Sieve::clear()
{
    std::vector<unsigned> &_primes = sieve_primes();
    _primes.erase(_primes.begin() + 10, _primes.end());
}

void Sieve::set_sieve_size(unsigned size)
{
#ifdef HAVE_SYMENGINE_PRIMESIEVE
    primesieve::set_sieve_size(size);
#else
    _sieve_size = size * 1024 * 8; // size in bits
#endif
}

void Sieve::_extend(unsigned limit)
{
    std::vector<unsigned> &_primes = sieve_primes();
#ifdef HAVE_SYMENGINE_PRIMESIEVE
    if (_primes.back() < limit)
        primesieve::generate_primes(_primes.back() + 1, limit, &_primes);
#else
    const unsigned sqrt_limit
        = static_cast<unsigned>(std::floor(std::sqrt(limit)));
    unsigned start = _primes.back() + 1;
    if (limit <= start)
        return;
    if (sqrt_limit >= start) {
        _extend(sqrt_limit);
        start = _primes.back() + 1;
    }

    unsigned segment = _sieve_size;
    std::valarray<bool> is_prime(segment);
    for (; start <= limit; start += 2 * segment) {
        unsigned finish = std::min(start + segment * 2 + 1, limit);
        is_prime[std::slice(0, segment, 1)] = true;
        // considering only odd integers. An odd number n corresponds to
        // n-start/2 in the array.
        for (unsigned index = 1; index < _primes.size()
                                 and _primes[index] * _primes[index] <= finish;
             ++index) {
            unsigned n = _primes[index];
            unsigned multiple = (start / n + 1) * n;
            if (multiple % 2 == 0)
                multiple += n;
            if (multiple > finish)
                continue;
            std::slice sl = std::slice((multiple - start) / 2,
                                       1 + (finish - multiple) / (2 * n), n);
            // starting from n*n, all the odd multiples of n are marked not
            // prime.
            is_prime[sl] = false;
        }
        for (unsigned n = start + 1; n <= finish; n += 2) {
            if (is_prime[(n - start) / 2])
                _primes.push_back(n);
        }
    }
#endif
}

void Sieve::generate_primes(std::vector<unsigned> &primes, unsigned limit)
{
    _extend(limit);
    std::vector<unsigned> &_primes = sieve_primes();
    auto it = std::upper_bound(_primes.begin(), _primes.end(), limit);
    // find the first position greater than limit and reserve space for the
    // primes
    primes.reserve(it - _primes.begin());
    std::copy(_primes.begin(), it, std::back_inserter(primes));
    if (_clear)
        clear();
}

Sieve::iterator::iterator(unsigned max)
{
    _limit = max;
    _index = 0;
}

Sieve::iterator::iterator()
{
    _limit = 0;
    _index = 0;
}

Sieve::iterator::~iterator()
{
    if (_clear)
        Sieve::clear();
}

unsigned Sieve::iterator::next_prime()
{
    std::vector<unsigned> &_primes = sieve_primes();
    if (_index >= _primes.size()) {
        unsigned extend_to = _primes[_index - 1] * 2;
        if (_limit > 0 and _limit < extend_to) {
            extend_to = _limit;
        }
        _extend(extend_to);
        if (_index >= _primes.size()) { // the next prime is greater than _limit
            return _limit + 1;
        }
    }
    return _primes[_index++];
}

}; // namespace SymEngine
