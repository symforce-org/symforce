#include <symengine/monomials.h>

namespace SymEngine
{

// This is the fastest implementation:
void monomial_mul(const vec_int &A, const vec_int &B, vec_int &C)
{
    size_t n = A.size();
    for (size_t i = 0; i < n; ++i) {
        C[i] = A[i] + B[i];
    }
}

/*
// Other implementation of monomial_mul() are below. Those are slightly slower,
// so they are commented out.

// This is slightly slower than monomial_mul
void monomial_mul2(const vec_int &A, const vec_int &B, vec_int &C)
{
    std::transform(A.begin(), A.end(), B.begin(), C.begin(), std::plus<int>());
}

// The same as monomial_mul2
void monomial_mul3(const vec_int &A, const vec_int &B, vec_int &C)
{
    std::transform(A.begin(), A.end(), B.begin(), C.begin(),
        [] (int a, int b) { return a + b; });
}
*/

} // namespace SymEngine
