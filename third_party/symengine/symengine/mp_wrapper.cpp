#include <symengine/mp_wrapper.h>
#include <stdlib.h>
#include <cstring>

namespace SymEngine
{
#if SYMENGINE_INTEGER_CLASS == SYMENGINE_FLINT
std::ostream &operator<<(std::ostream &os, const fmpz_wrapper &f)
{
    void (*freefunc)(void *, size_t);
    mp_get_memory_functions(NULL, NULL, &freefunc);
    char *c = fmpz_get_str(NULL, 10, f.get_fmpz_t());
    os << std::string(c);
    freefunc(c, strlen(c) + 1);
    return os;
}

std::ostream &operator<<(std::ostream &os, const fmpq_wrapper &f)
{
    void (*freefunc)(void *, size_t);
    mp_get_memory_functions(NULL, NULL, &freefunc);
    char *c = fmpq_get_str(NULL, 10, f.get_fmpq_t());
    os << std::string(c);
    freefunc(c, strlen(c) + 1);
    return os;
}
#elif SYMENGINE_INTEGER_CLASS == SYMENGINE_GMP

std::ostream &operator<<(std::ostream &os, const mpz_wrapper &f)
{
    void (*freefunc)(void *, size_t);
    mp_get_memory_functions(NULL, NULL, &freefunc);
    char *c = mpz_get_str(NULL, 10, f.get_mpz_t());
    os << std::string(c);
    freefunc(c, strlen(c) + 1);
    return os;
}

std::ostream &operator<<(std::ostream &os, const mpq_wrapper &f)
{
    void (*freefunc)(void *, size_t);
    mp_get_memory_functions(NULL, NULL, &freefunc);
    char *c = mpq_get_str(NULL, 10, f.get_mpq_t());
    os << std::string(c);
    freefunc(c, strlen(c) + 1);
    return os;
}
#endif

} // namespace SymEngine
