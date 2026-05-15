#ifndef SYMENGINE_NTHEORY_FUNCS_H
#define SYMENGINE_NTHEORY_FUNCS_H

#include <symengine/basic.h>
#include <symengine/symengine_casts.h>
#include <symengine/constants.h>
#include <symengine/functions.h>
#include <symengine/add.h>
#include <symengine/pow.h>

namespace SymEngine
{

class PrimePi : public OneArgFunction
{
    /*! The prime counting function pi(x)
     * A function that takes a real value x and returns the number of
     * primes less than or equal to x.
     * https://en.wikipedia.org/wiki/Prime-counting_function
     **/

public:
    IMPLEMENT_TYPEID(SYMENGINE_PRIMEPI)
    PrimePi(const RCP<const Basic> &arg);
    bool is_canonical(const RCP<const Basic> &arg) const;
    RCP<const Basic> create(const RCP<const Basic> &arg) const override;
};

RCP<const Basic> primepi(const RCP<const Basic> &arg);

class Primorial : public OneArgFunction
{
    /*! The primorial of n (n#)
     * The product all primes up to n
     * https://en.wikipedia.org/wiki/Primorial
     **/

public:
    IMPLEMENT_TYPEID(SYMENGINE_PRIMORIAL)
    Primorial(const RCP<const Basic> &arg);
    bool is_canonical(const RCP<const Basic> &arg) const;
    RCP<const Basic> create(const RCP<const Basic> &arg) const override;
};

RCP<const Basic> primorial(const RCP<const Basic> &arg);

/**
 * @brief n:th s-gonal number
 * @param s Number of sides of the polygon. Must be greater than 2.
 * @param n Must be greater than 0
 * @returns The n:th s-gonal number
 *
 * Symbolic calculation of the n:th s-gonal number.
 * Sources: https://en.wikipedia.org/wiki/Polygonal_number
 * https://reference.wolfram.com/language/ref/PolygonalNumber.html
 */
RCP<const Basic> polygonal_number(const RCP<const Basic> &s,
                                  const RCP<const Basic> &n);

/**
 * @brief Principal s-gonal root of x
 * @param s Number of sides of the polygon. Must be greater than 2.
 * @param x An integer greater than 0
 * @returns The root
 *
 * Symbolic calculation of the principal (i.e. positive) s-gonal root
 * of x.
 * References https://en.wikipedia.org/wiki/Polygonal_number
 * http://oeis.org/wiki/Polygonal_numbers#Polygonal_roots
 */
RCP<const Basic> principal_polygonal_root(const RCP<const Basic> &s,
                                          const RCP<const Basic> &x);

} // namespace SymEngine

#endif
