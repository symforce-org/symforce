#include <chrono>
#include <iostream>

#include <symengine/ntheory.h>
using std::cout;
using std::endl;

using SymEngine::integer;
using SymEngine::mertens;
using SymEngine::mobius;
using SymEngine::prime_factor_multiplicities;

void _bench_mertens(const unsigned long a)
{
    std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;

    cout << "mertens(" << a << "):";
    t1 = std::chrono::high_resolution_clock::now();
    mertens(a);
    t2 = std::chrono::high_resolution_clock::now();
    cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                .count()
         << "ms" << endl;
}

void bench_mertens()
{
    _bench_mertens(1);
    _bench_mertens(2);
    _bench_mertens(3);
    _bench_mertens(4);
    _bench_mertens(8);
    _bench_mertens(16);
    _bench_mertens(32);
    _bench_mertens(64);
    _bench_mertens(113);
    cout << endl;
}

void _bench_mobius(const unsigned long a)
{
    cout << "mobius(" << a << "): ";
    auto t1 = std::chrono::high_resolution_clock::now();
    mobius(*integer(a));
    auto t2 = std::chrono::high_resolution_clock::now();
    cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                .count()
         << "ms" << endl;
}
void bench_mobius()
{
    _bench_mobius(2);
    _bench_mobius(3);
    _bench_mobius(4);
    _bench_mobius(8);
    _bench_mobius(16);
}

void _bench_prime_factor_multiplicities(const unsigned long &a)
{
    SymEngine::map_integer_uint primes_mul;
    auto t1 = std::chrono::high_resolution_clock::now();
    cout << "prime_factor_multiplicities(primes_mul," << a << "): ";
    prime_factor_multiplicities(primes_mul, *integer(a));
    auto t2 = std::chrono::high_resolution_clock::now();
    cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                .count()
         << "ms" << endl
         << endl;
}
void bench_prime_factor_multiplicities()
{
    _bench_prime_factor_multiplicities(2);
    _bench_prime_factor_multiplicities(3);
    _bench_prime_factor_multiplicities(4);
    _bench_prime_factor_multiplicities(8);
    _bench_prime_factor_multiplicities(16);
}

void _bench_mp_sqrt(const unsigned long &a)
{
    SymEngine::map_integer_uint primes_mul;
    auto t1 = std::chrono::high_resolution_clock::now();
    cout << "mp_sqrt(" << a << "): ";
    prime_factor_multiplicities(primes_mul, *integer(a));
    auto t2 = std::chrono::high_resolution_clock::now();
    cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                .count()
         << "ms" << endl
         << endl;
}
void bench_mp_sqrt()
{
    _bench_mp_sqrt(2);
    _bench_mp_sqrt(3);
    _bench_mp_sqrt(4);
}

int main()
{
    bench_mertens();
    bench_mobius();
    bench_prime_factor_multiplicities();
    bench_mp_sqrt();
}