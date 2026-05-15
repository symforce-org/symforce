#include "catch.hpp"
#include <symengine/complex_mpc.h>
#include <symengine/functions.h>
#include <symengine/add.h>
#include <symengine/eval_double.h>
#include <symengine/eval_mpc.h>
#include <symengine/eval_mpfr.h>
#include <symengine/symengine_exception.h>
#include <symengine/pow.h>

using SymEngine::add;
using SymEngine::Basic;
using SymEngine::Complex;
using SymEngine::complex_double;
using SymEngine::down_cast;
using SymEngine::eq;
using SymEngine::eval_double;
using SymEngine::hash_t;
using SymEngine::Integer;
using SymEngine::integer;
using SymEngine::integer_class;
using SymEngine::is_a;
using SymEngine::make_rcp;
using SymEngine::minus_one;
using SymEngine::Number;
using SymEngine::NumberWrapper;
using SymEngine::one;
using SymEngine::print_stack_on_segfault;
using SymEngine::Rational;
using SymEngine::rational_class;
using SymEngine::RCP;
using SymEngine::real_double;
using SymEngine::sqrt;
using SymEngine::SymEngineException;
#ifdef HAVE_SYMENGINE_MPFR
using SymEngine::eval_mpfr;
using SymEngine::mpfr_class;
using SymEngine::real_mpfr;
using SymEngine::RealMPFR;
#endif
#ifdef HAVE_SYMENGINE_MPC
using SymEngine::complex_mpc;
using SymEngine::ComplexMPC;
using SymEngine::eval_mpc;
using SymEngine::mpc_class;
#endif

TEST_CASE("RealMPFR: arithmetic", "[number]")
{
#ifdef HAVE_SYMENGINE_MPFR
    mpfr_class a(100), b(100), d(100), c(100);
    mpfr_set_ui(a.get_mpfr_t(), 10, MPFR_RNDN);
    mpfr_set_ui(b.get_mpfr_t(), 20, MPFR_RNDN);
    mpfr_set_ui(c.get_mpfr_t(), 100, MPFR_RNDN);
    mpfr_set_ui(d.get_mpfr_t(), 1024, MPFR_RNDN);
    RCP<const Number> r1 = real_mpfr(std::move(a));
    RCP<const Number> r2 = real_mpfr(std::move(b));
    RCP<const Number> r3 = real_mpfr(std::move(c));
    RCP<const Number> r4 = real_mpfr(std::move(d));
    RCP<const Number> r5 = subnum(integer(0), r1);
    RCP<const Number> i1 = integer(1);
    RCP<const Number> i2 = integer(2);
    RCP<const Number> half = integer(1)->div(*integer(2));
    RCP<const Number> c1 = Complex::from_two_nums(*i1, *i1);
    RCP<const Number> rat1 = Rational::from_two_ints(*integer(10), *integer(3));
    RCP<const Number> rd1 = real_double(10.0);
    RCP<const Number> cd1 = complex_double(std::complex<double>(1, 2));

    REQUIRE(not r1->is_one());
    REQUIRE(not r1->is_minus_one());
    REQUIRE(not r1->is_complex());

    REQUIRE(eq(*r2, *addnum(r1, r1)));
    REQUIRE(is_a<RealMPFR>(*addnum(r1, rat1)));
    REQUIRE(eq(*r2, *addnum(r1, rd1)));
    REQUIRE(eq(*r1, *subnum(r1, integer(0))));
    REQUIRE(is_a<RealMPFR>(*subnum(r2, rat1)));
    REQUIRE(eq(*r1, *subnum(r2, rd1)));
    REQUIRE(is_a<RealMPFR>(*subnum(rat1, r2)));
    REQUIRE(eq(*neg(r1), *subnum(rd1, r2)));
    REQUIRE(eq(*r2, *mulnum(r1, i2)));
    REQUIRE(eq(*r2, *mulnum(i2, r1)));
    REQUIRE(eq(*r3, *mulnum(r1, rd1)));
    REQUIRE(is_a<RealMPFR>(*mulnum(rat1, r1)));
    REQUIRE(eq(*r3, *mulnum(r1, r1)));
    REQUIRE(eq(*r1, *divnum(r2, i2)));
    REQUIRE(eq(*r1, *divnum(r3, rd1)));
    REQUIRE(is_a<RealMPFR>(*divnum(r3, rat1)));
    REQUIRE(eq(*r1, *divnum(r3, r1)));
    REQUIRE(is_a<RealMPFR>(*divnum(mulnum(rat1, rat1), r1)));
    REQUIRE(eq(*r1, *divnum(mulnum(rd1, rd1), r1)));
    REQUIRE(eq(*divnum(i1, r1), *divnum(i2, r2)));
    REQUIRE(eq(*r1, *subnum(r2, r1)));
    REQUIRE(eq(*r1, *subnum(integer(20), r1)));
    REQUIRE(eq(*r1, *mulnum(r2, half)));
    REQUIRE(eq(*r3, *pownum(r1, i2)));
    REQUIRE(eq(*r4, *pownum(i2, r1)));
    REQUIRE(eq(*r1, *pownum(r3, half)));
    REQUIRE(eq(*r3, *pownum(r1, real_double(2.0))));
    REQUIRE(eq(*r3, *pownum(r1, divnum(r2, r1))));
    REQUIRE(eq(*r4, *pownum(real_double(2.0), r1)));
    REQUIRE(is_a<RealMPFR>(*pownum(rat1, r1)));
    REQUIRE(eq(*divnum(i1, r4), *half->pow(*r1)));

    mpfr_class e(100);
    mpfr_set_ui(e.get_mpfr_t(), 10, MPFR_RNDN);
    r2 = real_mpfr(std::move(e));
    REQUIRE(r1->__hash__() == r2->__hash__());
    REQUIRE(r1->compare(*r2) == 0);
    REQUIRE(r1->compare(*r3) == -1);
    REQUIRE(r3->compare(*r2) == 1);

    // to increase coverage
    mpfr_class ee(10);
    mpfr_set_ui(ee.get_mpfr_t(), 101, MPFR_RNDN);
    r2 = real_mpfr(std::move(ee));
    REQUIRE(r1->compare(*r2) == 1); // TO-DO is this is a bug or what ?

#ifdef HAVE_SYMENGINE_MPC
    REQUIRE(is_a<ComplexMPC>(*addnum(r1, c1)));
    REQUIRE(is_a<ComplexMPC>(*addnum(r1, cd1)));
    REQUIRE(is_a<ComplexMPC>(*subnum(r1, c1)));
    REQUIRE(is_a<ComplexMPC>(*subnum(r1, cd1)));
    REQUIRE(is_a<ComplexMPC>(*subnum(c1, r1)));
    REQUIRE(is_a<ComplexMPC>(*subnum(cd1, r1)));
    REQUIRE(is_a<ComplexMPC>(*mulnum(c1, r2)));
    REQUIRE(is_a<ComplexMPC>(*mulnum(cd1, r2)));
    REQUIRE(is_a<ComplexMPC>(*divnum(c1, r2)));
    REQUIRE(is_a<ComplexMPC>(*divnum(cd1, r2)));
    REQUIRE(is_a<ComplexMPC>(*divnum(r1, c1)));
    REQUIRE(is_a<ComplexMPC>(*divnum(r1, cd1)));
    REQUIRE(is_a<ComplexMPC>(*pownum(r5, half)));
    REQUIRE(is_a<ComplexMPC>(*pownum(integer(-2), r5)));
    REQUIRE(is_a<ComplexMPC>(*pownum(r1, c1)));
    REQUIRE(is_a<ComplexMPC>(*pownum(r5, cd1)));
    REQUIRE(is_a<ComplexMPC>(*pownum(r5, real_double(-0.2))));
    REQUIRE(is_a<ComplexMPC>(*pownum(r5, mulnum(minus_one, r1))));
    REQUIRE(is_a<ComplexMPC>(*pownum(c1, r1)));
    REQUIRE(is_a<ComplexMPC>(*pownum(cd1, r1)));
    REQUIRE(is_a<ComplexMPC>(*pownum(mulnum(minus_one, rat1), r1)));
    REQUIRE(is_a<ComplexMPC>(*pownum(real_double(-0.2), r5)));
    REQUIRE(is_a<ComplexMPC>(*pownum(mulnum(minus_one, r1), r5)));

#else
    CHECK_THROWS_AS(addnum(r1, c1), SymEngineException);
    CHECK_THROWS_AS(pownum(r5, half), SymEngineException);
    CHECK_THROWS_AS(pownum(integer(-2), r1), SymEngineException);
#endif // HAVE_SYMENGINE_MPC
#endif // HAVE_SYMENGINE_MPFR
}

TEST_CASE("ComplexMPC: arithmetic", "[number]")
{
#ifdef HAVE_SYMENGINE_MPC
    mpc_class a(100), b(100), d(100), c(100), e(100);
    mpc_set_ui_ui(a.get_mpc_t(), 10, 7, MPFR_RNDN);
    mpc_set_ui_ui(b.get_mpc_t(), 20, 14, MPFR_RNDN);
    mpc_pow_ui(c.get_mpc_t(), a.get_mpc_t(), 2, MPFR_RNDN);
    mpc_set_ui(d.get_mpc_t(), 2, MPFR_RNDN);
    mpc_pow(d.get_mpc_t(), d.get_mpc_t(), a.get_mpc_t(), MPFR_RNDN);
    mpc_set_ui_ui(e.get_mpc_t(), 10, 14, MPFR_RNDN);

    RCP<const Number> r1 = complex_mpc(std::move(a));
    RCP<const Number> r2 = complex_mpc(std::move(b));
    RCP<const Number> r3 = complex_mpc(std::move(c));
    RCP<const Number> r4 = complex_mpc(std::move(d));
    RCP<const Number> r5 = complex_mpc(std::move(e));
    RCP<const Number> i1 = integer(1);
    RCP<const Number> i2 = integer(2);
    RCP<const Number> i10 = integer(10);
    RCP<const Number> half = integer(1)->div(*integer(2));
    RCP<const Number> c1 = Complex::from_two_nums(*integer(10), *integer(7));
    RCP<const Number> rat1 = Rational::from_two_ints(*integer(10), *integer(3));
    RCP<const Number> rd1 = real_double(10.0);
    RCP<const Number> cd1 = complex_double(std::complex<double>(10, 7));

    REQUIRE(eq(*r2, *addnum(c1, r1)));
    REQUIRE(eq(*r2, *addnum(r1, r1)));
    REQUIRE(eq(*r2, *addnum(r5, i10)));
    REQUIRE(is_a<ComplexMPC>(*addnum(r5, rat1)));
    REQUIRE(eq(*r2, *addnum(r5, rd1)));
    REQUIRE(eq(*r2, *addnum(r1, cd1)));
    REQUIRE(eq(*r1, *subnum(r2, r1)));
    REQUIRE(eq(*r1, *subnum(mulnum(c1, i2), r1)));
    REQUIRE(is_a<ComplexMPC>(*subnum(r2, rat1)));
    REQUIRE(eq(*r5, *subnum(r2, rd1)));
    REQUIRE(eq(*r5, *subnum(r2, i10)));
    REQUIRE(eq(*r1, *subnum(r2, c1)));
    REQUIRE(eq(*r1, *subnum(r2, cd1)));
    REQUIRE(is_a<ComplexMPC>(*subnum(rat1, r2)));
    REQUIRE(eq(*neg(r5), *subnum(rd1, r2)));
    REQUIRE(eq(*neg(r5), *subnum(i10, r2)));
    REQUIRE(eq(*neg(r1), *subnum(c1, r2)));
    REQUIRE(eq(*neg(r1), *subnum(cd1, r2)));
    REQUIRE(eq(*r2, *mulnum(r1, i2)));
    REQUIRE(eq(*r2, *mulnum(i2, r1)));
    REQUIRE(eq(*mulnum(cd1, r1), *mulnum(c1, r1)));
    REQUIRE(eq(*r3, *mulnum(r1, r1)));
    REQUIRE(eq(*mulnum(r1, i10), *mulnum(r1, rd1)));
    REQUIRE(eq(*r1, *mulnum(r2, half)));
    REQUIRE(eq(*r1, *divnum(r2, i2)));
    REQUIRE(eq(*r2, *divnum(r1, half)));
    REQUIRE(eq(*divnum(r2, c1), *divnum(r2, cd1)));
    REQUIRE(eq(*divnum(i1, r1), *divnum(i2, r2)));
    REQUIRE(eq(*r1, *divnum(r3, r1)));
    REQUIRE(eq(*r1, *divnum(r2, real_double(2.0))));
    REQUIRE(eq(*divnum(one, r1), *divnum(i2, r2)));
    REQUIRE(eq(*divnum(one, r2), *divnum(half, r1)));
    REQUIRE(eq(*divnum(c1, r2), *divnum(cd1, r2)));
    REQUIRE(eq(*divnum(i1, r1), *divnum(i2, r2)));
    REQUIRE(eq(*divnum(one, r1), *divnum(real_double(2.0), r2)));
    REQUIRE(eq(*r3, *pownum(r1, i2)));
    REQUIRE(eq(*r3, *pownum(r1, real_double(2.0))));
    REQUIRE(is_a<ComplexMPC>(*pownum(r1, cd1)));
    REQUIRE(eq(*pownum(r1, r1), *pownum(r1, cd1)));
    REQUIRE(eq(*pownum(r1, c1), *pownum(r1, cd1)));
    REQUIRE(eq(*r3, *pownum(r1, real_double(2.0))));
    REQUIRE(eq(*r1, *pownum(r3, half)));
    REQUIRE(eq(*pownum(c1, r1), *pownum(cd1, r1)));
    REQUIRE(eq(*r1, *pownum(r3, real_double(0.5))));
    REQUIRE(eq(*r4, *pownum(i2, r1)));
    REQUIRE(eq(*r4, *pownum(real_double(2.0), r1)));
    REQUIRE(eq(*divnum(i1, r4), *half->pow(*r1)));

    mpc_class ee(100);
    mpc_set_ui_ui(ee.get_mpc_t(), 20, 14, MPFR_RNDN);
    RCP<const Number> r6 = complex_mpc(std::move(ee));
    REQUIRE(r6->__hash__() == r2->__hash__());
    REQUIRE(r6->compare(*r2) == 0);
    REQUIRE(r6->compare(*r3) == -1);
    REQUIRE(r3->compare(*r6) == 1);
    REQUIRE(r2->compare(*r5) == 1);

    mpc_class temp(101);
    mpc_set_ui_ui(temp.get_mpc_t(), 20, 14, MPFR_RNDN);
    REQUIRE(complex_mpc(std::move(temp))->compare(*r6) == 1);

#ifdef HAVE_SYMENGINE_MPFR
    mpfr_class f(100);
    mpfr_set_ui(f.get_mpfr_t(), 10, MPFR_RNDN);
    r6 = real_mpfr(std::move(f));

    REQUIRE(eq(*r2, *addnum(r5, r6)));
    REQUIRE(eq(*r5, *subnum(r2, r6)));
    REQUIRE(eq(*neg(r5), *subnum(r6, r2)));
    REQUIRE(eq(*mulnum(r2, i10), *mulnum(r6, r2)));
    REQUIRE(is_a<ComplexMPC>(*divnum(r3, r6)));
    REQUIRE(is_a<ComplexMPC>(*divnum(r6, r3)));
    REQUIRE(eq(*r3, *pownum(r1, divnum(r6, integer(5)))));
    REQUIRE(eq(*r1, *pownum(r3, divnum(r6, integer(20)))));
    REQUIRE(eq(*r4, *pownum(divnum(r6, integer(5)), r1)));
#endif // HAVE_SYMENGINE_MPFR
#endif // HAVE_SYMENGINE_MPC
}

TEST_CASE("Test is_exact", "[number]")
{
    RCP<const Number> n1;

    n1 = integer(0);
    REQUIRE(n1->is_exact());
    REQUIRE(n1->is_zero());
    REQUIRE(n1->is_exact_zero());
    REQUIRE(not n1->is_negative());
    REQUIRE(not n1->is_positive());
    REQUIRE(not n1->is_complex());

    n1 = integer(-1);
    REQUIRE(n1->is_exact());
    REQUIRE(not n1->is_zero());
    REQUIRE(not n1->is_exact_zero());
    REQUIRE(n1->is_negative());
    REQUIRE(not n1->is_positive());
    REQUIRE(not n1->is_complex());

    n1 = Rational::from_mpq(rational_class(2, 1));
    REQUIRE(n1->is_exact());
    REQUIRE(not n1->is_zero());
    REQUIRE(not n1->is_exact_zero());
    REQUIRE(not n1->is_negative());
    REQUIRE(n1->is_positive());

    n1 = Complex::from_mpq(rational_class(1, 1), rational_class(1, 2));
    REQUIRE(n1->is_exact());
    REQUIRE(not n1->is_zero());
    REQUIRE(not n1->is_exact_zero());
    REQUIRE(not n1->is_negative());
    REQUIRE(not n1->is_positive());

    n1 = real_double(0.0);
    REQUIRE(not n1->is_exact());
    REQUIRE(n1->is_zero());
    REQUIRE(not n1->is_exact_zero());
    REQUIRE(not n1->is_negative());
    REQUIRE(not n1->is_positive());

    n1 = real_double(1.0);
    REQUIRE(not n1->is_exact());
    REQUIRE(not n1->is_zero());
    REQUIRE(not n1->is_exact_zero());
    REQUIRE(not n1->is_negative());
    REQUIRE(n1->is_positive());
    REQUIRE(not n1->is_complex());
    REQUIRE(not n1->is_minus_one());

    n1 = complex_double(1.0);
    REQUIRE(not n1->is_exact());
    REQUIRE(not n1->is_zero());
    REQUIRE(not n1->is_exact_zero());
    REQUIRE(not n1->is_negative());
    REQUIRE(not n1->is_positive());

#ifdef HAVE_SYMENGINE_MPFR
    mpfr_class a(100);
    mpfr_set_d(a.get_mpfr_t(), 0.0, MPFR_RNDN);
    n1 = real_mpfr(std::move(a));
    REQUIRE(not n1->is_exact());
    REQUIRE(n1->is_zero());
    REQUIRE(not n1->is_exact_zero());
    REQUIRE(not n1->is_negative());
    REQUIRE(not n1->is_positive());

    a = mpfr_class(100);
    mpfr_set_d(a.get_mpfr_t(), 1.0, MPFR_RNDN);
    n1 = real_mpfr(std::move(a));
    REQUIRE(not n1->is_exact());
    REQUIRE(not n1->is_zero());
    REQUIRE(not n1->is_exact_zero());
    REQUIRE(not n1->is_negative());
    REQUIRE(n1->is_positive());

#ifdef HAVE_SYMENGINE_MPC
    mpc_class b(100);
    mpc_set_ui_ui(b.get_mpc_t(), 10, 7, MPFR_RNDN);
    n1 = complex_mpc(std::move(b));
    REQUIRE(not n1->is_exact());
    REQUIRE(not n1->is_zero());
    REQUIRE(not n1->is_exact_zero());
    REQUIRE(not n1->is_negative());
    REQUIRE(not n1->is_positive());
#endif // HAVE_SYMENGINE_MPC
#endif // HAVE_SYMENGINE_MPFR
}

TEST_CASE("Test NumberWrapper", "[number]")
{
    class Long : public NumberWrapper
    {
    public:
        long i_;
        Long(long i) : i_(i) {}

        virtual std::string __str__() const
        {
            std::stringstream ss;
            ss << i_;
            return ss.str();
        };
        virtual RCP<const Number> eval(long bits) const
        {
            return integer(integer_class(i_));
        };
        long number_to_long(const Number &x) const
        {
            long l;
            std::istringstream ss(x.__str__());
            ss >> l;
            return l;
        }
        virtual RCP<const Number> add(const Number &x) const
        {
            return make_rcp<Long>(i_ + number_to_long(x));
        }
        virtual RCP<const Number> sub(const Number &x) const
        {
            return make_rcp<Long>(i_ - number_to_long(x));
        }
        virtual RCP<const Number> rsub(const Number &x) const
        {
            return make_rcp<Long>(-i_ + number_to_long(x));
        }
        virtual RCP<const Number> mul(const Number &x) const
        {
            return make_rcp<Long>(i_ * number_to_long(x));
        }
        virtual RCP<const Number> div(const Number &x) const
        {
            return make_rcp<Long>(i_ / number_to_long(x));
        }
        virtual RCP<const Number> rdiv(const Number &x) const
        {
            return make_rcp<Long>(number_to_long(x) / i_);
        }
        virtual RCP<const Number> pow(const Number &x) const
        {
            return make_rcp<Long>(std::pow(i_, number_to_long(x)));
        }
        virtual RCP<const Number> rpow(const Number &x) const
        {
            return make_rcp<Long>(std::pow(number_to_long(x), i_));
        }
        virtual bool is_zero() const
        {
            return i_ == 0;
        }
        //! \return true if `1`
        virtual bool is_one() const
        {
            return i_ == 1;
        }
        //! \return true if `-1`
        virtual bool is_minus_one() const
        {
            return i_ == -1;
        }
        //! \return true if negative
        virtual bool is_negative() const
        {
            return i_ < 0;
        }
        //! \return true if positive
        virtual bool is_positive() const
        {
            return i_ > 0;
        }
        //! \returns `false`
        // False is returned because a long cannot have an imaginary part
        virtual bool is_complex() const
        {
            return false;
        }
        virtual hash_t __hash__() const
        {
            return i_;
        };
        //! true if `this` is equal to `o`.
        virtual bool __eq__(const Basic &o) const
        {
            return i_ == down_cast<const Long &>(o).i_;
        };
        virtual int compare(const Basic &o) const
        {
            long j = down_cast<const Long &>(o).i_;
            if (i_ == j)
                return 0;
            return i_ > j ? 1 : -1;
        };
    };

    RCP<const Number> n = make_rcp<Long>(10);
    RCP<const Number> m = integer(5)->add(*n);

    REQUIRE(eq(*integer(5)->add(*n), *make_rcp<Long>(15)));
    REQUIRE(eq(*integer(5)->mul(*n), *make_rcp<Long>(50)));

    RCP<const Basic> r = add(sqrt(integer(2)), n);
    double d1 = eval_double(*r);
    REQUIRE(std::abs(d1 - 11.4142135623730951) < 1e-12);
    REQUIRE(n->__str__() == "10");

#ifdef HAVE_SYMENGINE_MPFR
    mpfr_class aa(100);
    eval_mpfr(aa.get_mpfr_t(), *r, MPFR_RNDN);
    REQUIRE(mpfr_cmp_d(aa.get_mpfr_t(), 11.41421356237309) == 1);
    REQUIRE(mpfr_cmp_d(aa.get_mpfr_t(), 11.41421356237310) == -1);
#ifdef HAVE_SYMENGINE_MPC
    mpc_class a(100);
    eval_mpc(a.get_mpc_t(), *r, MPFR_RNDN);
    mpc_abs(aa.get_mpfr_t(), a.get_mpc_t(), MPFR_RNDN);
    REQUIRE(mpfr_cmp_d(aa.get_mpfr_t(), 11.41421356237309) == 1);
    REQUIRE(mpfr_cmp_d(aa.get_mpfr_t(), 11.41421356237310) == -1);
#endif // HAVE_SYMENGINE_MPC
#endif // HAVE_SYMENGINE_MPFR
}
