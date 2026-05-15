#include <symengine/cwrapper.h>

#if defined(HAVE_C_FUNCTION_NOT_FUNC)
#define __func__ __FUNCTION__
#endif

#include <string.h>
#include <math.h>

#ifdef HAVE_SYMENGINE_MPFR
#include <mpfr.h>
#endif // HAVE_SYMENGINE_MPFR

void test_version()
{
    SYMENGINE_C_ASSERT(strcmp(SYMENGINE_VERSION, symengine_version()) == 0);
}

void test_cwrapper()
{
    char *s;
    basic x, y, z;
    basic f;
    basic_new_stack(x);
    basic_new_stack(y);
    basic_new_stack(z);
    symbol_set(x, "x");
    symbol_set(y, "y");
    symbol_set(z, "z");

    SYMENGINE_C_ASSERT(is_a_Number(x) == 0);
    SYMENGINE_C_ASSERT(is_a_Number(y) == 0);
    SYMENGINE_C_ASSERT(is_a_Number(z) == 0);

    s = basic_str(x);
    SYMENGINE_C_ASSERT(strcmp(s, "x") == 0);
    basic_str_free(s);

    basic_new_stack(f);
    CVecBasic *vec = vecbasic_new();
    vecbasic_push_back(vec, x);
    vecbasic_push_back(vec, y);
    vecbasic_push_back(vec, z);
    function_symbol_set(f, "f", vec);
    s = basic_str(f);
    SYMENGINE_C_ASSERT(strcmp(s, "f(x, y, z)") == 0);
    vecbasic_free(vec);
    basic_str_free(s);

    basic e;
    basic_new_stack(e);
    integer_set_ui(e, 123);
    s = basic_str(e);
    SYMENGINE_C_ASSERT(strcmp(s, "123") == 0);
    basic_str_free(s);

    integer_set_ui(e, 456);
    basic_add(e, e, x);
    basic_mul(e, e, y);
    basic_div(e, e, z);
    s = basic_str(e);
    SYMENGINE_C_ASSERT(strcmp(s, "y*(456 + x)/z") == 0);
    basic_str_free(s);

    basic numer, denom;
    basic_new_stack(numer);
    basic_new_stack(denom);
    basic_as_numer_denom(numer, denom, e);
    basic_mul(e, e, z);
    SYMENGINE_C_ASSERT(basic_eq(numer, e) == 1);
    SYMENGINE_C_ASSERT(basic_eq(denom, z) == 1);
    basic_div(e, e, z);

    basic_diff(e, e, z);
    s = basic_str(e);
    SYMENGINE_C_ASSERT(strcmp(s, "-y*(456 + x)/z**2") == 0);
    basic_str_free(s);
    s = basic_str_julia(e);
    SYMENGINE_C_ASSERT(strcmp(s, "-y*(456 + x)/z^2") == 0);
    basic_str_free(s);

    rational_set_ui(e, 100, 47);
    s = basic_str(e);

    SYMENGINE_C_ASSERT(strcmp(s, "100/47") == 0);
    SYMENGINE_C_ASSERT(!is_a_Symbol(e));
    SYMENGINE_C_ASSERT(is_a_Rational(e));
    SYMENGINE_C_ASSERT(!is_a_Integer(e));
    basic_str_free(s);

    integer_set_ui(e, 123);
    basic_sqrt(e, e);
    basic_exp(e, e);

    s = basic_str(e);
    SYMENGINE_C_ASSERT(strcmp(s, "exp(sqrt(123))") == 0);
    basic_str_free(s);

    s = basic_str_julia(e);
    SYMENGINE_C_ASSERT(strcmp(s, "exp(sqrt(123))") == 0);
    basic_str_free(s);

    unsigned long size = 0;
    basic deserialized;

    char *serialized = basic_dumps(e, &size);
    basic_new_stack(deserialized);
    basic_loads(deserialized, serialized, size);
    SYMENGINE_C_ASSERT(basic_eq(deserialized, e) == 1);
    basic_str_free(serialized);
    basic_free_stack(deserialized);

    rational_set_si(e, 100, 47);
    s = basic_str(e);

    SYMENGINE_C_ASSERT(strcmp(s, "100/47") == 0);
    SYMENGINE_C_ASSERT(!is_a_Symbol(e));
    SYMENGINE_C_ASSERT(is_a_Rational(e));
    SYMENGINE_C_ASSERT(!is_a_Integer(e));

#if SYMENGINE_INTEGER_CLASS != SYMENGINE_BOOSTMP
    mpq_t testr;
    mpq_init(testr);
#endif
    basic a, b;
    basic_new_stack(a);
    basic_new_stack(b);

    integer_set_ui(a, 2);
    integer_set_ui(b, 4);
    rational_set(e, a, b);
#if SYMENGINE_INTEGER_CLASS != SYMENGINE_BOOSTMP
    rational_get_mpq(testr, e);
    SYMENGINE_C_ASSERT(mpq_cmp_ui(testr, 1, 2) == 0);
#endif

    integer_set_si(e, 0);
    SYMENGINE_C_ASSERT(integer_get_si(e) == 0);
    SYMENGINE_C_ASSERT(number_is_zero(e) == 1);
    SYMENGINE_C_ASSERT(number_is_negative(e) == 0);
    SYMENGINE_C_ASSERT(number_is_positive(e) == 0);
    SYMENGINE_C_ASSERT(number_is_complex(e) == 0);

    integer_set_ui(e, 123);
    SYMENGINE_C_ASSERT(integer_get_ui(e) == 123);
    SYMENGINE_C_ASSERT(number_is_zero(e) == 0);
    SYMENGINE_C_ASSERT(number_is_negative(e) == 0);
    SYMENGINE_C_ASSERT(number_is_positive(e) == 1);
    SYMENGINE_C_ASSERT(number_is_complex(e) == 0);

    integer_set_si(e, -123);
    SYMENGINE_C_ASSERT(integer_get_si(e) == -123);
    SYMENGINE_C_ASSERT(is_a_Number(e) == 1);
    SYMENGINE_C_ASSERT(number_is_zero(e) == 0);
    SYMENGINE_C_ASSERT(number_is_negative(e) == 1);
    SYMENGINE_C_ASSERT(number_is_positive(e) == 0);
    SYMENGINE_C_ASSERT(number_is_complex(e) == 0);

#if SYMENGINE_INTEGER_CLASS != SYMENGINE_BOOSTMP
    mpz_t test;
    mpz_init(test);

    integer_get_mpz(test, e);
    SYMENGINE_C_ASSERT(mpz_get_ui(test) == 123);
#endif

    char *str = "123 + 321";
    basic p;
    basic_new_stack(p);
    basic_parse(p, str);
    SYMENGINE_C_ASSERT(is_a_Integer(p));
    SYMENGINE_C_ASSERT(integer_get_si(p) == 444);

    basic_parse2(p, str, 1);
    SYMENGINE_C_ASSERT(is_a_Integer(p));
    SYMENGINE_C_ASSERT(integer_get_si(p) == 444);

#if SYMENGINE_INTEGER_CLASS != SYMENGINE_BOOSTMP
    mpq_clear(testr);
    mpz_clear(test);
#endif
    basic_free_stack(f);
    basic_free_stack(e);
    basic_free_stack(x);
    basic_free_stack(y);
    basic_free_stack(z);
    basic_free_stack(p);
    basic_free_stack(a);
    basic_free_stack(b);
    basic_free_stack(numer);
    basic_free_stack(denom);
    basic_str_free(s);
}

void test_basic()
{
    basic x;
    basic_new_stack(x);
    symbol_set(x, "x");

    basic_struct *y = basic_new_heap();
    symbol_set(y, "x");

    SYMENGINE_C_ASSERT(basic_eq(x, y))

    basic_free_heap(y);
    basic_free_stack(x);
}

void test_complex()
{
    basic e;
    basic f;
    char *s;
    basic_new_stack(e);
    basic_new_stack(f);
    rational_set_ui(e, 100, 47);
    rational_set_ui(f, 76, 59);
    complex_set(e, e, f);
    s = basic_str(e);

    SYMENGINE_C_ASSERT(strcmp(s, "100/47 + 76/59*I") == 0);
    SYMENGINE_C_ASSERT(!is_a_Symbol(e));
    SYMENGINE_C_ASSERT(!is_a_Rational(e));
    SYMENGINE_C_ASSERT(!is_a_Integer(e));
    SYMENGINE_C_ASSERT(is_a_Complex(e));
    SYMENGINE_C_ASSERT(number_is_zero(e) == 0);
    SYMENGINE_C_ASSERT(number_is_negative(e) == 0);
    SYMENGINE_C_ASSERT(number_is_positive(e) == 0);
    SYMENGINE_C_ASSERT(number_is_complex(e) == 1);

    basic_str_free(s);

    complex_base_real_part(f, e);
    s = basic_str(f);

    SYMENGINE_C_ASSERT(strcmp(s, "100/47") == 0);
    SYMENGINE_C_ASSERT(!is_a_Symbol(f));
    SYMENGINE_C_ASSERT(is_a_Rational(f));
    SYMENGINE_C_ASSERT(!is_a_Integer(f));
    SYMENGINE_C_ASSERT(!is_a_Complex(f));

    basic_str_free(s);

    complex_base_imaginary_part(f, e);
    s = basic_str(f);

    SYMENGINE_C_ASSERT(strcmp(s, "76/59") == 0);
    SYMENGINE_C_ASSERT(!is_a_Symbol(f));
    SYMENGINE_C_ASSERT(is_a_Rational(f));
    SYMENGINE_C_ASSERT(!is_a_Integer(f));
    SYMENGINE_C_ASSERT(!is_a_Complex(f));

    basic_str_free(s);

    basic_free_stack(e);
    basic_free_stack(f);
}

void test_complex_double()
{
    basic e;
    basic f;
    char *s;
    basic_new_stack(e);
    basic_new_stack(f);
    dcomplex k;

    basic_const_I(e);
    real_double_set_d(f, 76.59);
    basic_mul(f, f, e);
    real_double_set_d(e, 100.47);
    basic_add(e, e, f);
    s = basic_str(e);

    SYMENGINE_C_ASSERT(strcmp(s, "100.47 + 76.59*I") == 0);
    SYMENGINE_C_ASSERT(!is_a_Symbol(e));
    SYMENGINE_C_ASSERT(!is_a_Rational(e));
    SYMENGINE_C_ASSERT(!is_a_Integer(e));
    SYMENGINE_C_ASSERT(!is_a_Complex(e));
    SYMENGINE_C_ASSERT(is_a_ComplexDouble(e));
    SYMENGINE_C_ASSERT(number_is_zero(e) == 0);
    SYMENGINE_C_ASSERT(number_is_negative(e) == 0);
    SYMENGINE_C_ASSERT(number_is_positive(e) == 0);
    SYMENGINE_C_ASSERT(number_is_complex(e) == 1);

    basic_str_free(s);

    k = complex_double_get(e);
    SYMENGINE_C_ASSERT(k.real == 100.47);
    SYMENGINE_C_ASSERT(k.imag == 76.59);

    complex_base_real_part(f, e);
    s = basic_str(f);

    SYMENGINE_C_ASSERT(strcmp(s, "100.47") == 0);
    SYMENGINE_C_ASSERT(!is_a_Symbol(f));
    SYMENGINE_C_ASSERT(!is_a_Rational(f));
    SYMENGINE_C_ASSERT(!is_a_Integer(f));
    SYMENGINE_C_ASSERT(!is_a_Complex(f));
    SYMENGINE_C_ASSERT(is_a_RealDouble(f));

    basic_str_free(s);

    complex_base_imaginary_part(f, e);
    s = basic_str(f);

    SYMENGINE_C_ASSERT(strcmp(s, "76.59") == 0);
    SYMENGINE_C_ASSERT(!is_a_Symbol(f));
    SYMENGINE_C_ASSERT(!is_a_Rational(f));
    SYMENGINE_C_ASSERT(!is_a_Integer(f));
    SYMENGINE_C_ASSERT(!is_a_Complex(f));
    SYMENGINE_C_ASSERT(is_a_RealDouble(f));

    basic_str_free(s);

    basic_free_stack(e);
    basic_free_stack(f);
}

void test_real_double()
{
    basic d;
    basic_new_stack(d);
    real_double_set_d(d, 123.456);
    SYMENGINE_C_ASSERT(real_double_get_d(d) == 123.456);

    char *s2;
    s2 = basic_str(d);

    SYMENGINE_C_ASSERT(is_a_RealDouble(d));
    SYMENGINE_C_ASSERT(strcmp(s2, "123.456") == 0);
    SYMENGINE_C_ASSERT(number_is_zero(d) == 0);
    SYMENGINE_C_ASSERT(number_is_negative(d) == 0);
    SYMENGINE_C_ASSERT(number_is_positive(d) == 1);
    SYMENGINE_C_ASSERT(number_is_complex(d) == 0);

    basic_str_free(s2);

    basic_free_stack(d);
}

#ifdef HAVE_SYMENGINE_MPFR
void test_real_mpfr()
{
    basic d, e;
    basic_new_stack(e);
    basic_new_stack(d);

    real_mpfr_set_d(d, 123.456, 200);
    SYMENGINE_C_ASSERT(basic_get_type(d) == SYMENGINE_REAL_MPFR);
    SYMENGINE_C_ASSERT(real_mpfr_get_d(d) == 123.456);

    real_mpfr_set_str(e, "456.123", 200);
    SYMENGINE_C_ASSERT(basic_get_type(e) == SYMENGINE_REAL_MPFR);
    SYMENGINE_C_ASSERT(real_mpfr_get_d(e) == 456.123);
    SYMENGINE_C_ASSERT(real_mpfr_get_prec(e) == 200);

    mpfr_t mp;
    mpfr_init2(mp, 200);
    real_mpfr_get(mp, e);
    real_mpfr_set(d, mp);
    SYMENGINE_C_ASSERT(basic_get_type(d) == SYMENGINE_REAL_MPFR);
    SYMENGINE_C_ASSERT(real_mpfr_get_d(d) == 456.123);

    real_mpfr_set_d(d, 0, 200);
    SYMENGINE_C_ASSERT(number_is_zero(d) == 1);
    SYMENGINE_C_ASSERT(number_is_negative(d) == 0);
    SYMENGINE_C_ASSERT(number_is_positive(d) == 0);
    SYMENGINE_C_ASSERT(number_is_complex(d) == 0);

    real_mpfr_set_d(d, 0.000001, 200);
    SYMENGINE_C_ASSERT(number_is_zero(d) == 0);
    SYMENGINE_C_ASSERT(number_is_negative(d) == 0);
    SYMENGINE_C_ASSERT(number_is_positive(d) == 1);
    SYMENGINE_C_ASSERT(number_is_complex(d) == 0);

    real_mpfr_set_d(d, -0.000001, 200);
    SYMENGINE_C_ASSERT(number_is_zero(d) == 0);
    SYMENGINE_C_ASSERT(number_is_negative(d) == 1);
    SYMENGINE_C_ASSERT(number_is_positive(d) == 0);
    SYMENGINE_C_ASSERT(number_is_complex(d) == 0);

    mpfr_clear(mp);
    basic_free_stack(d);
    basic_free_stack(e);
}
#endif // HAVE_SYMENGINE_MPFR

#ifdef HAVE_SYMENGINE_MPC
void test_complex_mpc()
{
    basic d, d1, d2;
    basic_new_stack(d);
    basic_new_stack(d1);
    basic_new_stack(d2);

    basic_const_I(d2);

    real_mpfr_set_d(d, 0.000001, 200);
    real_mpfr_set_d(d1, 0.000001, 200);
    basic_mul(d2, d1, d2);
    basic_add(d2, d, d2);
    SYMENGINE_C_ASSERT(basic_get_type(d2) == SYMENGINE_COMPLEX_MPC);
    SYMENGINE_C_ASSERT(number_is_zero(d2) == 0);
    SYMENGINE_C_ASSERT(number_is_negative(d2) == 0);
    SYMENGINE_C_ASSERT(number_is_positive(d2) == 0);
    SYMENGINE_C_ASSERT(number_is_complex(d2) == 1);

    basic r1;
    basic_new_stack(r1);

    complex_base_real_part(r1, d2);
    SYMENGINE_C_ASSERT(basic_eq(r1, d));

    complex_base_imaginary_part(r1, d2);
    SYMENGINE_C_ASSERT(basic_eq(r1, d1));

    basic_free_stack(d);
    basic_free_stack(d1);
    basic_free_stack(d2);
    basic_free_stack(r1);
}
#endif // HAVE_SYMENGINE_MPC

void test_CVectorInt1()
{
    // Allocate on heap
    CVectorInt *vec = vectorint_new();
    vectorint_push_back(vec, 5);
    ;
    SYMENGINE_C_ASSERT(vectorint_get(vec, 0) == 5);
    vectorint_free(vec);
}

struct X {
    void *x;
};

void test_CVectorInt2()
{
    // Allocate on stack
    CVectorInt *vec;

    char data1[1]; // Not aligned properly
    SYMENGINE_C_ASSERT(vectorint_placement_new_check(data1, sizeof(data1))
                       == 1);

    struct X data2[1]; // Aligned properly but small
    SYMENGINE_C_ASSERT(vectorint_placement_new_check(data2, sizeof(data2))
                       == 1);

    struct X
        data3[50]; // Aligned properly and enough size to fit std::vector<int>
    SYMENGINE_C_ASSERT(vectorint_placement_new_check(data3, 1) == 1);
    SYMENGINE_C_ASSERT(vectorint_placement_new_check(data3, 2) == 1);
    SYMENGINE_C_ASSERT(vectorint_placement_new_check(data3, sizeof(data3))
                       == 0);
    vec = vectorint_placement_new(data3);
    vectorint_push_back(vec, 5);
    SYMENGINE_C_ASSERT(vectorint_get(vec, 0) == 5);
    vectorint_placement_free(vec);
}

void test_CVecBasic()
{
    CVecBasic *vec = vecbasic_new();
    SYMENGINE_C_ASSERT(vecbasic_size(vec) == 0);

    basic x;
    basic_new_stack(x);
    symbol_set(x, "x");
    vecbasic_push_back(vec, x);

    SYMENGINE_C_ASSERT(vecbasic_size(vec) == 1);

    basic y;
    basic_new_stack(y);
    vecbasic_get(vec, 0, y);

    SYMENGINE_C_ASSERT(basic_eq(x, y));

    vecbasic_push_back(vec, x);

    SYMENGINE_C_ASSERT(vecbasic_size(vec) == 2);

    vecbasic_erase(vec, 0);

    SYMENGINE_C_ASSERT(vecbasic_size(vec) == 1);

    basic z;
    basic_new_stack(z);
    symbol_set(z, "z");
    vecbasic_set(vec, 0, z);
    vecbasic_get(vec, 0, y);

    SYMENGINE_C_ASSERT(basic_eq(y, z));

    vecbasic_free(vec);
    basic_free_stack(x);
    basic_free_stack(y);
    basic_free_stack(z);
}

void test_CSetBasic()
{
    CSetBasic *set = setbasic_new();
    SYMENGINE_C_ASSERT(setbasic_size(set) == 0);

    basic x;
    basic_new_stack(x);
    symbol_set(x, "x");

    int has_insert;
    has_insert = setbasic_insert(set, x);
    SYMENGINE_C_ASSERT(has_insert == 1);
    SYMENGINE_C_ASSERT(setbasic_size(set) == 1);

    has_insert = setbasic_insert(set, x);
    SYMENGINE_C_ASSERT(has_insert == 0);

    basic y;
    basic_new_stack(y);
    symbol_set(y, "y");

    int is_found;
    is_found = setbasic_find(set, x);
    SYMENGINE_C_ASSERT(is_found == 1);

    is_found = setbasic_find(set, y);
    SYMENGINE_C_ASSERT(is_found == 0);

    setbasic_get(set, 0, y);
    SYMENGINE_C_ASSERT(basic_eq(x, y));

    int was_erased;
    symbol_set(y, "y");
    was_erased = setbasic_erase(set, y);
    SYMENGINE_C_ASSERT(was_erased == 0);
    SYMENGINE_C_ASSERT(setbasic_size(set) == 1);

    was_erased = setbasic_erase(set, x);
    SYMENGINE_C_ASSERT(was_erased == 1);
    SYMENGINE_C_ASSERT(setbasic_size(set) == 0);

    setbasic_free(set);
    basic_free_stack(x);
    basic_free_stack(y);
}

void test_CMapBasicBasic()
{
    CMapBasicBasic *map = mapbasicbasic_new();
    SYMENGINE_C_ASSERT(mapbasicbasic_size(map) == 0);

    basic x, y;
    basic_new_stack(x);
    basic_new_stack(y);
    symbol_set(x, "x");
    symbol_set(y, "y");

    mapbasicbasic_insert(map, x, y);
    SYMENGINE_C_ASSERT(mapbasicbasic_size(map) == 1);

    basic z;
    basic_new_stack(z);
    symbol_set(z, "z");

    int is_found;
    is_found = mapbasicbasic_get(map, x, z);
    SYMENGINE_C_ASSERT(is_found == 1);
    SYMENGINE_C_ASSERT(basic_eq(y, z));

    is_found = mapbasicbasic_get(map, y, z);
    SYMENGINE_C_ASSERT(is_found == 0);

    mapbasicbasic_free(map);
    basic_free_stack(x);
    basic_free_stack(y);
    basic_free_stack(z);
}

void test_get_args()
{
    basic x, y, z, e;
    basic_new_stack(x);
    basic_new_stack(y);
    basic_new_stack(z);
    basic_new_stack(e);
    symbol_set(x, "x");
    symbol_set(y, "y");
    symbol_set(z, "z");

    integer_set_ui(e, 123);
    basic_add(e, e, x);
    basic_mul(e, e, y);
    basic_div(e, e, z);

    CVecBasic *args = vecbasic_new();
    basic_get_args(e, args);
    SYMENGINE_C_ASSERT(vecbasic_size(args) == 3);
    vecbasic_free(args);

    basic_free_stack(e);
    basic_free_stack(x);
    basic_free_stack(y);
    basic_free_stack(z);
}

void test_free_symbols()
{
    basic x, y, z, e;
    basic_new_stack(x);
    basic_new_stack(y);
    basic_new_stack(z);
    basic_new_stack(e);
    symbol_set(x, "x");
    symbol_set(y, "y");
    symbol_set(z, "z");

    integer_set_ui(e, 123);
    basic_add(e, e, x);
    basic_pow(e, e, y);
    basic_div(e, e, z);

    CSetBasic *symbols = setbasic_new();
    basic_free_symbols(e, symbols);
    SYMENGINE_C_ASSERT(setbasic_size(symbols) == 3);
    setbasic_free(symbols);

    basic_free_stack(e);
    basic_free_stack(x);
    basic_free_stack(y);
    basic_free_stack(z);
}

void test_function_symbols()
{
    char *s;
    basic x, y, z, e;
    basic_new_stack(x);
    basic_new_stack(y);
    basic_new_stack(z);
    basic_new_stack(e);
    symbol_set(x, "x");
    symbol_set(y, "y");
    symbol_set(z, "z");

    basic f, g, h;
    basic_new_stack(f);
    basic_new_stack(g);
    basic_new_stack(h);

    CVecBasic *vec1 = vecbasic_new();
    vecbasic_push_back(vec1, x);
    function_symbol_set(g, "g", vec1);

    CVecBasic *vec2 = vecbasic_new();
    vecbasic_push_back(vec2, g);
    function_symbol_set(h, "h", vec2);

    CVecBasic *vec = vecbasic_new();
    basic_add(e, x, y);
    vecbasic_push_back(vec, e);
    vecbasic_push_back(vec, g);
    vecbasic_push_back(vec, h);

    function_symbol_set(f, "f", vec);

    basic_add(z, z, f);

    s = basic_str(z);
    SYMENGINE_C_ASSERT(strcmp(s, "z + f(x + y, g(x), h(g(x)))") == 0);

    CSetBasic *symbols = setbasic_new();
    basic_function_symbols(symbols, f);
    SYMENGINE_C_ASSERT(setbasic_size(symbols) == 3);
    setbasic_free(symbols);

    basic_free_stack(e);
    basic_free_stack(x);
    basic_free_stack(y);
    basic_free_stack(z);

    basic_free_stack(f);
    basic_free_stack(g);
    basic_free_stack(h);
    vecbasic_free(vec);
    vecbasic_free(vec1);
    vecbasic_free(vec2);
    basic_str_free(s);
}

void test_function_symbol_get_name()
{
    char *s1, *s2;
    basic x;
    basic_new_stack(x);
    symbol_set(x, "x");

    basic f, g;
    basic_new_stack(f);
    basic_new_stack(g);

    CVecBasic *vec1 = vecbasic_new();
    vecbasic_push_back(vec1, x);
    function_symbol_set(g, "g", vec1);

    CVecBasic *vec2 = vecbasic_new();
    vecbasic_push_back(vec2, g);
    function_symbol_set(f, "f", vec2);

    s1 = function_symbol_get_name(g);
    SYMENGINE_C_ASSERT(strcmp(s1, "g") == 0);

    s2 = function_symbol_get_name(f);
    SYMENGINE_C_ASSERT(strcmp(s2, "f") == 0);

    basic_free_stack(x);
    basic_free_stack(f);
    basic_free_stack(g);
    vecbasic_free(vec1);
    vecbasic_free(vec2);
    basic_str_free(s1);
    basic_str_free(s2);
}

void test_get_type()
{
    basic x, y;
    basic_new_stack(x);
    basic_new_stack(y);
    symbol_set(x, "x");
    integer_set_ui(y, 123);

    SYMENGINE_C_ASSERT(basic_get_type(x) == SYMENGINE_SYMBOL);
    SYMENGINE_C_ASSERT(basic_get_type(y) == SYMENGINE_INTEGER);

    SYMENGINE_C_ASSERT(basic_get_class_id("Integer") == SYMENGINE_INTEGER);
    SYMENGINE_C_ASSERT(basic_get_class_id("Add") == SYMENGINE_ADD);

    char *s;
    s = basic_get_class_from_id(SYMENGINE_INTEGER);
    SYMENGINE_C_ASSERT(strcmp(s, "Integer") == 0);
    basic_str_free(s);

    basic_free_stack(x);
    basic_free_stack(y);
}

void test_hash()
{
    basic x1, x2, y;
    basic_new_stack(x1);
    basic_new_stack(x2);
    basic_new_stack(y);
    symbol_set(x1, "x");
    symbol_set(x2, "x");
    symbol_set(y, "y");

    SYMENGINE_C_ASSERT(basic_hash(x1) == basic_hash(x2));
    if (basic_hash(x1) != basic_hash(y))
        SYMENGINE_C_ASSERT(basic_neq(x1, y));

    basic_free_stack(x1);
    basic_free_stack(x2);
    basic_free_stack(y);
}

void test_subs2()
{
    basic s, e, x, y, z;
    basic_new_stack(s);
    basic_new_stack(e);
    basic_new_stack(x);
    basic_new_stack(y);
    basic_new_stack(z);

    symbol_set(x, "x");
    symbol_set(y, "y");
    symbol_set(z, "z");
    basic_mul(e, x, y);
    basic_mul(e, e, z);
    // e should be x*y*z

    basic_subs2(s, e, y, x);
    basic_subs2(s, s, z, x);
    // s should be x**3

    integer_set_si(z, 3);
    basic_pow(e, x, z);
    // e should be x**3

    SYMENGINE_C_ASSERT(basic_eq(s, e));

    basic_free_stack(s);
    basic_free_stack(e);
    basic_free_stack(x);
    basic_free_stack(y);
    basic_free_stack(z);
}

void test_subs()
{
    basic s, e, x, y, z;
    basic_new_stack(s);
    basic_new_stack(e);
    basic_new_stack(x);
    basic_new_stack(y);
    basic_new_stack(z);

    symbol_set(x, "x");
    symbol_set(y, "y");
    symbol_set(z, "z");
    basic_mul(e, x, y);
    basic_mul(e, e, z);
    // e should be x*y*z

    CMapBasicBasic *map = mapbasicbasic_new();
    mapbasicbasic_insert(map, y, x);
    mapbasicbasic_insert(map, z, x);
    basic_subs(s, e, map);
    // s should be x**3

    integer_set_si(z, 3);
    basic_pow(e, x, z);
    // e should be x**3

    SYMENGINE_C_ASSERT(basic_eq(s, e));

    mapbasicbasic_free(map);
    basic_free_stack(s);
    basic_free_stack(e);
    basic_free_stack(x);
    basic_free_stack(y);
    basic_free_stack(z);
}

void test_coeff()
{
    basic x, y, z, e, n4;
    basic_new_stack(x);
    basic_new_stack(y);
    basic_new_stack(z);
    basic_new_stack(e);
    basic_new_stack(n4);
    symbol_set(x, "x");
    symbol_set(y, "y");
    symbol_set(z, "z");

    integer_set_si(n4, 4);
    basic_mul(e, n4, x);
    basic_add(e, e, y);
    basic_add(e, e, z);

    basic n1;
    basic c1, c2, c3;
    basic_new_stack(n1);
    basic_new_stack(c1);
    basic_new_stack(c2);
    basic_new_stack(c3);
    integer_set_si(n1, 1);
    basic_coeff(c1, e, x, n1);
    basic_coeff(c2, e, y, n1);
    basic_coeff(c3, e, z, n1);

    SYMENGINE_C_ASSERT(basic_eq(c1, n4));
    SYMENGINE_C_ASSERT(basic_eq(c2, n1));
    SYMENGINE_C_ASSERT(basic_eq(c3, n1));

    basic_free_stack(c3);
    basic_free_stack(c2);
    basic_free_stack(c1);
    basic_free_stack(n1);
    basic_free_stack(n4);
    basic_free_stack(e);
    basic_free_stack(x);
    basic_free_stack(y);
    basic_free_stack(z);
}

void test_linsolve()
{
    basic x, y;
    basic i2, i3, i4, i9;
    basic e1, e2;

    basic_new_stack(x);
    basic_new_stack(y);

    symbol_set(x, "x");
    symbol_set(y, "y");

    basic_new_stack(i2);
    basic_new_stack(i3);
    basic_new_stack(i4);
    basic_new_stack(i9);

    integer_set_si(i2, -2);
    integer_set_si(i3, 3);
    integer_set_si(i4, -4);
    integer_set_si(i9, 9);

    basic_new_stack(e1);
    basic_new_stack(e2);

    symbol_set(e1, "e1");
    symbol_set(e2, "e2");

    // -2x - 4 + y
    basic_mul(e1, i2, x);
    basic_add(e1, e1, i4);
    basic_add(e1, e1, y);

    // 3x + y - 9
    basic_mul(e2, i3, x);
    basic_add(e2, e2, y);
    basic_add(e2, e2, i9);

    CVecBasic *sym = vecbasic_new();
    vecbasic_push_back(sym, x);
    vecbasic_push_back(sym, y);

    CVecBasic *sys = vecbasic_new();
    vecbasic_push_back(sys, e1);
    vecbasic_push_back(sys, e2);

    CVecBasic *sol = vecbasic_new();
    vecbasic_linsolve(sol, sys, sym);
    SYMENGINE_C_ASSERT(vecbasic_size(sol) == 2);

    vecbasic_free(sym);
    vecbasic_free(sys);
    vecbasic_free(sol);

    basic_free_stack(e1);
    basic_free_stack(e2);

    basic_free_stack(x);
    basic_free_stack(y);

    basic_free_stack(i2);
    basic_free_stack(i3);
    basic_free_stack(i4);
    basic_free_stack(i9);
}

void test_solve_poly()
{
    basic x, a;
    basic m1, i2, i5;
    CWRAPPER_OUTPUT_TYPE error_code;

    basic_new_stack(x);
    basic_new_stack(a);

    symbol_set(x, "x");
    symbol_set(a, "a");

    basic_new_stack(m1);
    basic_new_stack(i2);
    basic_new_stack(i5);

    basic_const_minus_one(m1);
    integer_set_si(i2, 2);
    integer_set_si(i5, 5);

    // a = x^2 - 1
    basic_pow(a, x, i2);
    basic_add(a, a, m1);

    CSetBasic *r = setbasic_new();
    basic_solve_poly(r, a, x);
    SYMENGINE_C_ASSERT(setbasic_size(r) == 2);

    setbasic_free(r);

    // a = exp(x) - 1
    basic_exp(a, x);
    basic_add(a, a, m1);

    CSetBasic *r3 = setbasic_new();
    error_code = basic_solve_poly(r3, a, x);
    SYMENGINE_C_ASSERT(setbasic_size(r3) == 0);
    SYMENGINE_C_ASSERT(error_code == SYMENGINE_RUNTIME_ERROR);

    setbasic_free(r3);

    basic_free_stack(m1);
    basic_free_stack(a);
    basic_free_stack(x);
    basic_free_stack(i2);
    basic_free_stack(i5);
}

void test_constants()
{
    basic z, o, mo, i;
    basic_new_stack(z);
    basic_new_stack(o);
    basic_new_stack(mo);
    basic_new_stack(i);

    integer_set_si(z, 0);
    integer_set_si(o, 1);
    integer_set_si(mo, -1);
    complex_set(i, z, o);

    basic zero, one, minus_one, iota;
    basic_new_stack(zero);
    basic_new_stack(one);
    basic_new_stack(minus_one);
    basic_new_stack(iota);

    basic_const_zero(zero);
    basic_const_one(one);
    basic_const_minus_one(minus_one);
    basic_const_I(iota);

    SYMENGINE_C_ASSERT(basic_eq(z, zero));
    SYMENGINE_C_ASSERT(basic_eq(o, one));
    SYMENGINE_C_ASSERT(basic_eq(mo, minus_one));
    SYMENGINE_C_ASSERT(basic_eq(i, iota));

    basic_free_stack(z);
    basic_free_stack(zero);
    basic_free_stack(o);
    basic_free_stack(one);
    basic_free_stack(mo);
    basic_free_stack(minus_one);
    basic_free_stack(i);
    basic_free_stack(iota);

    basic custom, pi, e, euler_gamma, catalan, goldenratio;
    basic_new_stack(custom);
    basic_new_stack(pi);
    basic_new_stack(e);
    basic_new_stack(euler_gamma);
    basic_new_stack(catalan);
    basic_new_stack(goldenratio);

    basic_const_set(custom, "custom");
    basic_const_pi(pi);
    basic_const_E(e);
    basic_const_EulerGamma(euler_gamma);
    basic_const_Catalan(catalan);
    basic_const_GoldenRatio(goldenratio);

    char *s;
    s = basic_str(custom);
    SYMENGINE_C_ASSERT(strcmp(s, "custom") == 0);
    basic_str_free(s);
    s = basic_str(pi);
    SYMENGINE_C_ASSERT(strcmp(s, "pi") == 0);
    basic_str_free(s);
    s = basic_str(e);
    SYMENGINE_C_ASSERT(strcmp(s, "E") == 0);
    basic_str_free(s);
    s = basic_str_julia(e);
    SYMENGINE_C_ASSERT(strcmp(s, "exp(1)") == 0);
    basic_str_free(s);
    s = basic_str_julia(catalan);
    SYMENGINE_C_ASSERT(strcmp(s, "catalan") == 0);
    basic_str_free(s);
    s = basic_str(euler_gamma);
    SYMENGINE_C_ASSERT(strcmp(s, "EulerGamma") == 0);
    basic_str_free(s);
    s = basic_str(catalan);
    SYMENGINE_C_ASSERT(strcmp(s, "Catalan") == 0);
    basic_str_free(s);
    s = basic_str(goldenratio);
    SYMENGINE_C_ASSERT(strcmp(s, "GoldenRatio") == 0);
    basic_str_free(s);

    // Checking mpfr builds
    s = "mpfr";
#ifdef HAVE_SYMENGINE_MPFR
    SYMENGINE_C_ASSERT(symengine_have_component(s));
#else
    SYMENGINE_C_ASSERT(!symengine_have_component(s));
#endif
    // Checking mpc builds
    s = "mpc";
#ifdef HAVE_SYMENGINE_MPC
    SYMENGINE_C_ASSERT(symengine_have_component(s));
#else
    SYMENGINE_C_ASSERT(!symengine_have_component(s));
#endif
    // Checking arb builds
    s = "arb";
#ifdef HAVE_SYMENGINE_ARB
    SYMENGINE_C_ASSERT(symengine_have_component(s));
#else
    SYMENGINE_C_ASSERT(!symengine_have_component(s));
#endif
    // Checking flint builds
    s = "flint";
#ifdef HAVE_SYMENGINE_FLINT
    SYMENGINE_C_ASSERT(symengine_have_component(s));
#else
    SYMENGINE_C_ASSERT(!symengine_have_component(s));
#endif
    // Checking ecm builds
    s = "ecm";
#ifdef HAVE_SYMENGINE_ECM
    SYMENGINE_C_ASSERT(symengine_have_component(s));
#else
    SYMENGINE_C_ASSERT(!symengine_have_component(s));
#endif
    // Checking primesieve builds
    s = "primesieve";
#ifdef HAVE_SYMENGINE_PRIMESIEVE
    SYMENGINE_C_ASSERT(symengine_have_component(s));
#else
    SYMENGINE_C_ASSERT(!symengine_have_component(s));
#endif
    // Checking piranha builds
    s = "piranha";
#ifdef HAVE_SYMENGINE_PIRANHA
    SYMENGINE_C_ASSERT(symengine_have_component(s));
#else
    SYMENGINE_C_ASSERT(!symengine_have_component(s));
#endif
    // Checking boost builds
    s = "boost";
#ifdef HAVE_SYMENGINE_BOOST
    SYMENGINE_C_ASSERT(symengine_have_component(s));
#else
    SYMENGINE_C_ASSERT(!symengine_have_component(s));
#endif
    // Checking pthread builds
    s = "pthread";
#ifdef HAVE_SYMENGINE_PTHREAD
    SYMENGINE_C_ASSERT(symengine_have_component(s));
#else
    SYMENGINE_C_ASSERT(!symengine_have_component(s));
#endif
    // Checking llvm builds
    s = "llvm";
#ifdef HAVE_SYMENGINE_LLVM
    SYMENGINE_C_ASSERT(symengine_have_component(s));
#else
    SYMENGINE_C_ASSERT(!symengine_have_component(s));
#endif
    // Checking llvm builds with optional long double
    s = "llvm_long_double";
#ifdef HAVE_SYMENGINE_LLVM_LONG_DOUBLE
    SYMENGINE_C_ASSERT(symengine_have_component(s));
#else
    SYMENGINE_C_ASSERT(!symengine_have_component(s));
#endif

    basic_free_stack(custom);
    basic_free_stack(pi);
    basic_free_stack(e);
    basic_free_stack(euler_gamma);
    basic_free_stack(goldenratio);
    basic_free_stack(catalan);
}

void test_infinity()
{
    basic Inf, NegInf, ComplexInf;
    basic_new_stack(Inf);
    basic_new_stack(NegInf);
    basic_new_stack(ComplexInf);

    basic_const_infinity(Inf);
    basic_const_neginfinity(NegInf);
    basic_const_complex_infinity(ComplexInf);

    char *s;
    s = basic_str(Inf);
    SYMENGINE_C_ASSERT(strcmp(s, "oo") == 0);
    basic_str_free(s);
    s = basic_str_julia(Inf);
    SYMENGINE_C_ASSERT(strcmp(s, "Inf") == 0);
    basic_str_free(s);
    s = basic_str(NegInf);
    SYMENGINE_C_ASSERT(strcmp(s, "-oo") == 0);
    basic_str_free(s);
    s = basic_str_julia(NegInf);
    SYMENGINE_C_ASSERT(strcmp(s, "-Inf") == 0);
    basic_str_free(s);
    s = basic_str(ComplexInf);
    SYMENGINE_C_ASSERT(strcmp(s, "zoo") == 0);
    basic_str_free(s);
    s = basic_str_julia(ComplexInf);
    SYMENGINE_C_ASSERT(strcmp(s, "zoo") == 0);
    basic_str_free(s);

    basic_free_stack(Inf);
    basic_free_stack(NegInf);
    basic_free_stack(ComplexInf);
}

void test_nan()
{
    basic custom;
    basic_new_stack(custom);

    basic_const_nan(custom);

    char *s;
    s = basic_str(custom);
    SYMENGINE_C_ASSERT(strcmp(s, "nan") == 0);
    basic_str_free(s);
    s = basic_str_julia(custom);
    SYMENGINE_C_ASSERT(strcmp(s, "NaN") == 0);
    basic_str_free(s);

    basic_free_stack(custom);
}

void test_ascii_art()
{
    char *s = ascii_art_str();
    SYMENGINE_C_ASSERT(strlen(s) > 0);
    basic_str_free(s);
}

void test_functions()
{
    basic pi, e, complex_inf;
    basic minus_one, minus_half, zero, one, two, three, four, ten, twenty_four;
    basic pi_div_two, pi_div_four;
    basic e_minus_one;
    basic exp_minus_two;
    basic ans, res;

    basic_new_stack(pi);
    basic_new_stack(e);
    basic_new_stack(complex_inf);
    basic_new_stack(ans);
    basic_new_stack(res);
    basic_new_stack(two);
    basic_new_stack(three);
    basic_new_stack(pi_div_two);
    basic_new_stack(four);
    basic_new_stack(pi_div_four);
    basic_new_stack(one);
    basic_new_stack(minus_one);
    basic_new_stack(zero);
    basic_new_stack(e_minus_one);
    basic_new_stack(exp_minus_two);
    basic_new_stack(minus_half);
    basic_new_stack(ten);
    basic_new_stack(twenty_four);

    basic_const_pi(pi);
    basic_const_E(e);
    basic_const_complex_infinity(complex_inf);
    integer_set_si(two, 2);
    integer_set_si(four, 4);
    integer_set_si(three, 3);
    integer_set_si(one, 1);
    integer_set_si(minus_one, -1);
    integer_set_si(zero, 0);
    integer_set_si(ten, 10);
    integer_set_si(twenty_four, 24);

    CVecBasic *vec = vecbasic_new();

    vecbasic_push_back(vec, four);
    vecbasic_push_back(vec, two);
    vecbasic_push_back(vec, three);
    vecbasic_push_back(vec, one);

    basic_div(pi_div_two, pi, two);
    basic_div(pi_div_four, pi, four);
    basic_pow(e_minus_one, e, minus_one);
    basic_mul(e_minus_one, e_minus_one, minus_one);
    basic_div(minus_half, minus_one, two);
    basic_mul(exp_minus_two, minus_one, two);
    basic_exp(exp_minus_two, exp_minus_two);

    char *s;

    basic_erf(ans, zero);
    SYMENGINE_C_ASSERT(basic_eq(ans, zero));

    basic_erfc(ans, zero);
    SYMENGINE_C_ASSERT(basic_eq(ans, one));

    basic_sin(ans, pi);
    SYMENGINE_C_ASSERT(basic_eq(ans, zero));

    basic_cos(ans, pi);
    SYMENGINE_C_ASSERT(basic_eq(ans, minus_one));

    basic_tan(ans, pi);
    SYMENGINE_C_ASSERT(basic_eq(ans, zero));

    basic_csc(ans, pi_div_two);
    s = basic_str(ans);
    SYMENGINE_C_ASSERT(basic_eq(ans, one));
    basic_str_free(s);

    basic_sec(ans, pi);
    SYMENGINE_C_ASSERT(basic_eq(ans, minus_one));

    basic_cot(ans, pi_div_four);
    SYMENGINE_C_ASSERT(basic_eq(ans, one));

    basic_asin(ans, one);
    SYMENGINE_C_ASSERT(basic_eq(ans, pi_div_two));

    basic_acos(ans, one);
    SYMENGINE_C_ASSERT(basic_eq(ans, zero));

    basic_atan(ans, one);
    SYMENGINE_C_ASSERT(basic_eq(ans, pi_div_four));

    basic_acot(ans, one);
    SYMENGINE_C_ASSERT(basic_eq(ans, pi_div_four));

    basic_acsc(ans, one);
    SYMENGINE_C_ASSERT(basic_eq(ans, pi_div_two));

    basic_asec(ans, one);
    SYMENGINE_C_ASSERT(basic_eq(ans, zero));

    basic_sinh(ans, zero);
    SYMENGINE_C_ASSERT(basic_eq(ans, zero));

    basic_cosh(ans, zero);
    SYMENGINE_C_ASSERT(basic_eq(ans, one));

    basic_tanh(ans, zero);
    SYMENGINE_C_ASSERT(basic_eq(ans, zero));

    basic_csch(ans, one);
    s = basic_str(ans);
    SYMENGINE_C_ASSERT(strcmp(s, "csch(1)") == 0);
    basic_str_free(s);

    basic_sech(ans, zero);
    SYMENGINE_C_ASSERT(basic_eq(ans, one));

    basic_coth(ans, one);
    s = basic_str(ans);
    SYMENGINE_C_ASSERT(strcmp(s, "coth(1)") == 0);
    basic_str_free(s);

    basic_asinh(ans, zero);
    SYMENGINE_C_ASSERT(basic_eq(ans, zero));

    basic_acosh(ans, one);
    SYMENGINE_C_ASSERT(basic_eq(ans, zero));

    basic_atanh(ans, zero);
    SYMENGINE_C_ASSERT(basic_eq(ans, zero));

    basic_acsch(ans, one);
    s = basic_str(ans);
    SYMENGINE_C_ASSERT(strcmp(s, "log(1 + sqrt(2))") == 0);
    basic_str_free(s);

    basic_asech(ans, one);
    SYMENGINE_C_ASSERT(basic_eq(ans, zero));

    basic_acoth(ans, one);
    s = basic_str(ans);
    SYMENGINE_C_ASSERT(strcmp(s, "acoth(1)") == 0);
    basic_str_free(s);

    basic_lambertw(ans, e_minus_one);
    SYMENGINE_C_ASSERT(basic_eq(ans, minus_one));

    basic_zeta(ans, zero);
    SYMENGINE_C_ASSERT(basic_eq(ans, minus_half));

    basic_dirichlet_eta(ans, one);
    s = basic_str(ans);
    SYMENGINE_C_ASSERT(strcmp(s, "log(2)") == 0);
    basic_str_free(s);

    integer_set_ui(res, 2);
    basic_log(res, res);
    SYMENGINE_C_ASSERT(basic_eq(res, ans));

    real_double_set_d(res, 1.1);
    basic_floor(res, res);
    SYMENGINE_C_ASSERT(basic_eq(res, one));

    real_double_set_d(res, 0.8);
    basic_ceiling(res, res);
    SYMENGINE_C_ASSERT(basic_eq(res, one));

    basic_gamma(ans, one);
    SYMENGINE_C_ASSERT(basic_eq(ans, one));

    basic_loggamma(ans, one);
    SYMENGINE_C_ASSERT(basic_eq(ans, zero));

    basic_atan2(ans, one, one);
    basic_mul(ans, ans, four);
    SYMENGINE_C_ASSERT(basic_eq(ans, pi));

    basic_kronecker_delta(ans, two, two);
    SYMENGINE_C_ASSERT(basic_eq(ans, one));

    basic_lowergamma(ans, one, two);
    basic_add(ans, ans, exp_minus_two);
    SYMENGINE_C_ASSERT(basic_eq(ans, one));

    basic_div(ans, one, two);
    basic_beta(ans, ans, two);
    basic_mul(ans, ans, three);
    SYMENGINE_C_ASSERT(basic_eq(ans, four));

    basic_mul(ans, minus_one, two);
    basic_polygamma(ans, two, ans);
    SYMENGINE_C_ASSERT(basic_eq(ans, complex_inf));

    basic_max(ans, vec);
    SYMENGINE_C_ASSERT(basic_eq(ans, four));

    basic_min(ans, vec);
    SYMENGINE_C_ASSERT(basic_eq(ans, one));

    basic_add_vec(ans, vec);
    SYMENGINE_C_ASSERT(basic_eq(ans, ten));

    basic_mul_vec(ans, vec);
    SYMENGINE_C_ASSERT(basic_eq(ans, twenty_four));

    basic_free_stack(ans);
    basic_free_stack(res);
    basic_free_stack(pi);
    basic_free_stack(two);
    basic_free_stack(pi_div_two);
    basic_free_stack(four);
    basic_free_stack(pi_div_four);
    basic_free_stack(one);
    basic_free_stack(three);
    basic_free_stack(minus_one);
    basic_free_stack(zero);
    basic_free_stack(e);
    basic_free_stack(e_minus_one);
    basic_free_stack(minus_half);
    basic_free_stack(exp_minus_two);
    basic_free_stack(complex_inf);
    basic_free_stack(ten);
    basic_free_stack(twenty_four);
    vecbasic_free(vec);
}

void test_ntheory()
{
    int ret_val;
    basic x, y, z, a, b, c, i1, i2, i3, i4, i5, im1, im2, im3, im7;
    basic_new_stack(x);
    basic_new_stack(y);
    basic_new_stack(z);
    basic_new_stack(a);
    basic_new_stack(b);
    basic_new_stack(c);
    basic_new_stack(i1);
    basic_new_stack(i2);
    basic_new_stack(i3);
    basic_new_stack(i4);
    basic_new_stack(i5);
    basic_new_stack(im1);
    basic_new_stack(im2);
    basic_new_stack(im3);
    basic_new_stack(im7);

    integer_set_si(i1, 1);
    integer_set_si(i2, 2);
    integer_set_si(i3, 3);
    integer_set_si(i4, 4);
    integer_set_si(i5, 5);
    integer_set_si(im1, -1);
    integer_set_si(im2, -2);
    integer_set_si(im3, -3);
    integer_set_si(im7, -7);

    ntheory_gcd(x, i2, i4);
    SYMENGINE_C_ASSERT(basic_eq(x, i2));

    ntheory_lcm(x, i2, i4);
    SYMENGINE_C_ASSERT(basic_eq(x, i4));

    ntheory_gcd_ext(x, y, z, i2, i3);
    SYMENGINE_C_ASSERT(basic_eq(x, i1));
    basic_mul(a, i2, y);
    basic_mul(b, i3, z);
    basic_add(c, a, b);
    SYMENGINE_C_ASSERT(basic_eq(x, c));

    ntheory_nextprime(x, i4);
    SYMENGINE_C_ASSERT(basic_eq(x, i5));

    ntheory_mod(x, i5, i4);
    SYMENGINE_C_ASSERT(basic_eq(x, i1));

    ntheory_quotient(x, i5, i2);
    SYMENGINE_C_ASSERT(basic_eq(x, i2));

    ntheory_quotient_mod(x, y, im7, i4);
    SYMENGINE_C_ASSERT(basic_eq(x, im1));
    SYMENGINE_C_ASSERT(basic_eq(y, im3));

    ntheory_mod_f(x, im7, i4);
    SYMENGINE_C_ASSERT(basic_eq(x, i1));

    ntheory_quotient_f(x, im7, i4);
    SYMENGINE_C_ASSERT(basic_eq(x, im2));

    ntheory_quotient_mod_f(x, y, im7, i4);
    SYMENGINE_C_ASSERT(basic_eq(x, im2));
    SYMENGINE_C_ASSERT(basic_eq(y, i1));

    ret_val = ntheory_mod_inverse(x, i3, i5);
    SYMENGINE_C_ASSERT(ret_val != 0);
    SYMENGINE_C_ASSERT(basic_eq(x, i2));

    ntheory_fibonacci(x, 5);
    SYMENGINE_C_ASSERT(basic_eq(x, i5));

    ntheory_fibonacci2(x, y, 5);
    SYMENGINE_C_ASSERT(basic_eq(x, i5));
    SYMENGINE_C_ASSERT(basic_eq(y, i3));

    ntheory_lucas(x, 1);
    SYMENGINE_C_ASSERT(basic_eq(x, i1));

    ntheory_lucas2(x, y, 3);
    SYMENGINE_C_ASSERT(basic_eq(x, i4));
    SYMENGINE_C_ASSERT(basic_eq(y, i3));

    ntheory_binomial(x, i5, 1);
    SYMENGINE_C_ASSERT(basic_eq(x, i5));

    ntheory_factorial(x, 0);
    ntheory_factorial(y, 1);
    ntheory_factorial(z, 2);
    SYMENGINE_C_ASSERT(basic_eq(x, i1));
    SYMENGINE_C_ASSERT(basic_eq(y, i1));
    SYMENGINE_C_ASSERT(basic_eq(z, i2));

    basic_free_stack(x);
    basic_free_stack(y);
    basic_free_stack(z);
    basic_free_stack(a);
    basic_free_stack(b);
    basic_free_stack(c);
    basic_free_stack(i1);
    basic_free_stack(i2);
    basic_free_stack(i3);
    basic_free_stack(i4);
    basic_free_stack(i5);
    basic_free_stack(im1);
    basic_free_stack(im2);
    basic_free_stack(im3);
    basic_free_stack(im7);
}

void test_eval()
{
    basic sin2, eval;
    basic_new_stack(sin2);
    basic_new_stack(eval);

    integer_set_si(sin2, 2);
    basic_sin(sin2, sin2);
    basic_evalf(eval, sin2, 53, 1);
    SYMENGINE_C_ASSERT(basic_get_type(eval) == SYMENGINE_REAL_DOUBLE);
    double d = 0.909297;
    double d2 = real_double_get_d(eval);
    d = fabs(d - d2);
    d2 = 0.000001;

    SYMENGINE_C_ASSERT(d < d2);

    basic_free_stack(sin2);

#ifdef HAVE_SYMENGINE_MPFR
    basic s, t, r, eval2;
    basic_new_stack(s);
    basic_new_stack(t);
    basic_new_stack(r);
    basic_new_stack(eval2);

    basic_const_pi(s);
    integer_set_str(t, "1963319607");
    basic_mul(s, s, t);
    integer_set_str(t, "6167950454");
    basic_sub(r, s, t);
    // value of `r` is approximately 0.000000000149734291

    basic_evalf(eval2, r, 53, 1);
    SYMENGINE_C_ASSERT(basic_get_type(eval2) == SYMENGINE_REAL_DOUBLE);
    // With 53 bit precision, `s` and `t` have the same value.
    // Hence value of `r` was  rounded down to `0.000000000000000`
    SYMENGINE_C_ASSERT(real_double_get_d(eval2) == 0.0);

    basic_evalf(eval2, r, 100, 1);
    SYMENGINE_C_ASSERT(basic_get_type(eval2) == SYMENGINE_REAL_MPFR);
    // With 100 bit precision, `s` and `t` are not equal in value.
    // Value of `r` is a positive quantity with value 0.000000000149734291.....
    SYMENGINE_C_ASSERT(number_is_zero(eval2) == 0);

    basic_free_stack(s);
    basic_free_stack(t);
    basic_free_stack(r);
    basic_free_stack(eval2);
#endif // HAVE_SYMENGINE_MPFR

    basic imag, n1, n2, temp;
    basic_new_stack(imag);
    basic_new_stack(n1);
    basic_new_stack(n2);
    basic_new_stack(temp);

    basic_const_I(imag);
    integer_set_ui(n1, 4);
    basic_sin(n1, n1);
    integer_set_ui(temp, 3);
    basic_sin(temp, temp);
    basic_mul(temp, temp, imag);
    basic_add(n1, n1, temp);
    // n1 = sin(4) + sin(3)i

    integer_set_ui(n2, 2);
    basic_sin(n2, n2);
    integer_set_ui(temp, 7);
    basic_sin(temp, temp);
    basic_mul(temp, temp, imag);
    basic_add(n2, n2, temp);
    // n2 = sin(2) + sin(7)i

    basic_mul(n1, n1, n2);
    // n1 = (sin(4) + sin(3)i) * (sin(2) + sin(7)i)

    basic_evalf(eval, n1, 53, 0);
    SYMENGINE_C_ASSERT(basic_get_type(eval) == SYMENGINE_COMPLEX_DOUBLE);
    d = -0.780872515;
    complex_base_real_part(temp, eval);
    d2 = real_double_get_d(temp);
    complex_base_imaginary_part(temp, eval);
    double d3 = real_double_get_d(temp);
    double d4 = -0.3688890370;
    d = fabs(d - d2);
    d4 = fabs(d4 - d3);

    d2 = 0.000001;

    SYMENGINE_C_ASSERT(d < d2 && d4 < d2);

    basic_free_stack(eval);
    basic_free_stack(n1);
    basic_free_stack(n2);

#ifdef HAVE_SYMENGINE_MPC
    basic s1, t1, r1, eval3, com1, com2;
    basic_new_stack(s1);
    basic_new_stack(t1);
    basic_new_stack(r1);
    basic_new_stack(eval3);
    basic_new_stack(com1);
    basic_new_stack(com2);

    basic_const_pi(s1);
    integer_set_str(t1, "1963319607");
    basic_mul(s1, s1, t1);
    basic_mul(com1, s1, imag);
    basic_add(com1, com1, s1);
    integer_set_str(t1, "6167950454");
    basic_mul(com2, t1, imag);
    basic_add(com2, com2, t1);

    basic_sub(r1, com1, com2);
    // value of `r1` is approximately 0.000000000149734291 +
    // 0.000000000149734291i

    basic_evalf(eval3, r1, 53, 0);
    SYMENGINE_C_ASSERT(basic_get_type(eval3) == SYMENGINE_COMPLEX_DOUBLE);

    // With 53 bit precision, `com1` and `com2` have the same value.
    // Hence value of `r1` was  rounded down to `0.000000000000000`
    complex_base_real_part(temp, eval3);
    SYMENGINE_C_ASSERT(real_double_get_d(temp) == 0.0);
    complex_base_imaginary_part(temp, eval3);
    SYMENGINE_C_ASSERT(real_double_get_d(temp) == 0.0);

    basic_evalf(eval3, r1, 100, 0);
    SYMENGINE_C_ASSERT(basic_get_type(eval3) == SYMENGINE_COMPLEX_MPC);
    // With 100 bit precision, `com1` and `com2` are not equal in value.
    // Value of `r1` is a positive quantity with value 0.000000000149734291.....

    SYMENGINE_C_ASSERT(number_is_zero(eval3) == 0);

    basic_free_stack(s1);
    basic_free_stack(t1);
    basic_free_stack(r1);
    basic_free_stack(eval3);
    basic_free_stack(com1);
    basic_free_stack(com2);
#endif // HAVE_SYMENGINE_MPC

    basic_free_stack(temp);
    basic_free_stack(imag);
}

void test_matrix()
{
    CDenseMatrix *A = dense_matrix_new();
    SYMENGINE_C_ASSERT(is_a_DenseMatrix(A));
    dense_matrix_free(A);

    CSparseMatrix *SA = sparse_matrix_new();
    sparse_matrix_init(SA);
    SYMENGINE_C_ASSERT(is_a_SparseMatrix(SA));
    sparse_matrix_free(SA);

    CSparseMatrix *SB = sparse_matrix_new();
    sparse_matrix_init(SB);
    SYMENGINE_C_ASSERT(is_a_SparseMatrix(SB));
    sparse_matrix_free(SB);

    basic i1, i2, i3, i4;
    basic_new_stack(i1);
    basic_new_stack(i2);
    basic_new_stack(i3);
    basic_new_stack(i4);

    integer_set_ui(i1, 1);
    integer_set_ui(i2, 2);
    integer_set_ui(i3, 3);
    integer_set_ui(i4, 4);

    CVecBasic *vec = vecbasic_new();
    vecbasic_push_back(vec, i1);
    vecbasic_push_back(vec, i2);
    vecbasic_push_back(vec, i3);
    vecbasic_push_back(vec, i4);

    CDenseMatrix *B = dense_matrix_new_vec(2, 2, vec);
    SYMENGINE_C_ASSERT(is_a_DenseMatrix(B));
    vecbasic_free(vec);

    dense_matrix_get_basic(i4, B, 0, 0);
    SYMENGINE_C_ASSERT(is_a_Integer(i4));
    SYMENGINE_C_ASSERT(integer_get_ui(i4) == 1);

    dense_matrix_get_basic(i3, B, 0, 1);
    SYMENGINE_C_ASSERT(is_a_Integer(i3));
    SYMENGINE_C_ASSERT(integer_get_ui(i3) == 2);

    dense_matrix_get_basic(i2, B, 1, 0);
    SYMENGINE_C_ASSERT(is_a_Integer(i2));
    SYMENGINE_C_ASSERT(integer_get_ui(i2) == 3);

    dense_matrix_get_basic(i1, B, 1, 1);
    SYMENGINE_C_ASSERT(is_a_Integer(i1));
    SYMENGINE_C_ASSERT(integer_get_ui(i1) == 4);

    integer_set_ui(i1, 5);

    dense_matrix_set_basic(B, 0, 0, i1);

    dense_matrix_get_basic(i4, B, 0, 0);
    SYMENGINE_C_ASSERT(is_a_Integer(i4));
    SYMENGINE_C_ASSERT(integer_get_ui(i4) == 5);

    // Equality
    SYMENGINE_C_ASSERT(dense_matrix_eq(B, B) == 1);

    // Inverse

    vec = vecbasic_new();

    integer_set_ui(i4, 4);
    integer_set_ui(i3, 3);
    integer_set_ui(i2, 2);

    vecbasic_push_back(vec, i4);
    vecbasic_push_back(vec, i3);
    vecbasic_push_back(vec, i3);
    vecbasic_push_back(vec, i2);

    CDenseMatrix *C = dense_matrix_new_vec(2, 2, vec);
    vecbasic_free(vec);

    vec = vecbasic_new();

    integer_set_si(i4, -4);
    integer_set_si(i2, -2);

    vecbasic_push_back(vec, i2);
    vecbasic_push_back(vec, i3);
    vecbasic_push_back(vec, i3);
    vecbasic_push_back(vec, i4);

    CDenseMatrix *D = dense_matrix_new_vec(2, 2, vec);

    CDenseMatrix *E = dense_matrix_new();
    dense_matrix_inv(E, C);

    SYMENGINE_C_ASSERT(dense_matrix_eq(E, D) == 1);

    int r = dense_matrix_rows(E);
    int c = dense_matrix_cols(E);

    SYMENGINE_C_ASSERT(r == 2);
    SYMENGINE_C_ASSERT(c == 2);

    // matrix addition
    dense_matrix_add_matrix(C, E, D);

    char *result = dense_matrix_str(C);
    char *expected = "[-4, 6]\n[6, -8]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    dense_matrix_transpose(C, B);
    result = dense_matrix_str(C);
    expected = "[5, 3]\n[2, 4]\n";
    // Transpose of [[5, 2],[3, 4]]

    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    integer_set_ui(i1, 4);
    integer_set_ui(i2, 3);
    integer_set_ui(i3, 6);

    dense_matrix_set_basic(B, 0, 0, i1);
    dense_matrix_set_basic(B, 0, 1, i2);
    dense_matrix_set_basic(B, 1, 0, i3);
    dense_matrix_set_basic(B, 1, 1, i2);

    // LU decomposition
    dense_matrix_LU(C, D, B);

    result = dense_matrix_str(C);
    expected = "[1, 0]\n[3/2, 1]\n";

    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    result = dense_matrix_str(D);
    expected = "[4, 3]\n[0, -3/2]\n";

    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    // matrix multiplication
    dense_matrix_mul_matrix(E, C, D);
    SYMENGINE_C_ASSERT(dense_matrix_eq(E, B) == 1);

    // scalar multiplication
    dense_matrix_mul_scalar(E, D, i2);
    //"[[4, 3],[0, -3/2]] * 3

    result = dense_matrix_str(E);
    expected = "[12, 9]\n[0, -9/2]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    // scalar addition
    dense_matrix_add_scalar(E, D, i2);
    //"[[4, 3],[0, -3/2]] + 3

    result = dense_matrix_str(E);
    expected = "[7, 6]\n[3, 3/2]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    // LDL
    integer_set_ui(i1, 4);
    integer_set_ui(i2, 3);
    integer_set_ui(i3, 2);

    dense_matrix_set_basic(B, 0, 0, i1);
    dense_matrix_set_basic(B, 0, 1, i2);
    dense_matrix_set_basic(B, 1, 0, i2);
    dense_matrix_set_basic(B, 1, 1, i3);
    // B = [[4, 3],[3, 2]]

    // LDL decomposition
    // [[4, 3],[3, 2]] = [[1, 0],[3/4, 1]] * [[4, 0],[0, -1/4]] * [[1, 3/4],[0,
    // 1]]
    dense_matrix_LDL(C, D, B);
    // L : C, U : D

    result = dense_matrix_str(C);
    expected = "[1, 0]\n[3/4, 1]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    result = dense_matrix_str(D);
    expected = "[4, 0]\n[0, -1/4]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    // set matrix
    dense_matrix_set(C, D);
    SYMENGINE_C_ASSERT(dense_matrix_eq(D, C) == 1);

    dense_matrix_LDL(D, E, B);

    // now C should be equal to E, but different to D
    SYMENGINE_C_ASSERT(dense_matrix_eq(C, E) == 1);
    SYMENGINE_C_ASSERT(dense_matrix_eq(D, C) == 0);

    // submatrix
    dense_matrix_submatrix(C, B, 0, 0, 1, 0, 1, 1);
    result = dense_matrix_str(C);
    expected = "[4]\n[3]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    // LU_solve
    dense_matrix_set_basic(C, 1, 0, i1);
    dense_matrix_LU_solve(D, B, C);
    result = dense_matrix_str(D);
    expected = "[4]\n[-4]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    // Fractionfree LU
    dense_matrix_FFLU(C, B);
    result = dense_matrix_str(C);
    expected = "[4, 3]\n[3, -1]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    // FractionFree LDU
    dense_matrix_FFLDU(C, D, E, B);
    result = dense_matrix_str(C);
    expected = "[4, 0]\n[3, 1]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    result = dense_matrix_str(D);
    expected = "[4, 0]\n[0, 4]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    result = dense_matrix_str(E);
    expected = "[4, 3]\n[0, -1]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    // Num-py like functions

    // Ones
    dense_matrix_ones(D, 2, 3);
    result = dense_matrix_str(D);
    expected = "[1, 1, 1]\n[1, 1, 1]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    // Zeros
    dense_matrix_zeros(D, 3, 2);
    result = dense_matrix_str(D);
    expected = "[0, 0]\n[0, 0]\n[0, 0]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    // Diag
    dense_matrix_diag(D, vec, 0);
    result = dense_matrix_str(D);
    expected = "[-2, 0, 0, 0]\n[0, 3, 0, 0]\n[0, 0, 3, 0]\n[0, 0, 0, -4]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    // det
    dense_matrix_det(i1, D);
    SYMENGINE_C_ASSERT(integer_get_ui(i1) == 72);

    // eye
    dense_matrix_eye(D, 3, 4, 1);
    result = dense_matrix_str(D);
    expected = "[0, 1, 0, 0]\n[0, 0, 1, 0]\n[0, 0, 0, 1]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    // Num-py like functions

    // Ones
    dense_matrix_ones(D, 2, 3);
    result = dense_matrix_str(D);
    expected = "[1, 1, 1]\n[1, 1, 1]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    // Zeros
    dense_matrix_zeros(D, 3, 2);
    result = dense_matrix_str(D);
    expected = "[0, 0]\n[0, 0]\n[0, 0]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    // Diag
    dense_matrix_diag(D, vec, 0);
    result = dense_matrix_str(D);
    expected = "[-2, 0, 0, 0]\n[0, 3, 0, 0]\n[0, 0, 3, 0]\n[0, 0, 0, -4]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    // det
    dense_matrix_det(i1, D);
    SYMENGINE_C_ASSERT(integer_get_ui(i1) == 72);

    // eye
    dense_matrix_eye(D, 3, 4, 1);
    result = dense_matrix_str(D);
    expected = "[0, 1, 0, 0]\n[0, 0, 1, 0]\n[0, 0, 0, 1]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    // diff
    basic x;
    basic y;
    basic e;
    basic_new_stack(x);
    basic_new_stack(y);
    basic_new_stack(e);
    symbol_set(x, "x");
    symbol_set(y, "y");
    integer_set_si(i2, 2);
    dense_matrix_rows_cols(B, 2, 2);
    dense_matrix_set_basic(B, 0, 0, x);
    basic_mul(e, i2, x);
    dense_matrix_set_basic(B, 0, 1, e);
    dense_matrix_set_basic(B, 1, 0, i2);
    basic_mul(e, x, x);
    dense_matrix_set_basic(B, 1, 1, e);
    dense_matrix_rows_cols(D, 2, 2);
    dense_matrix_diff(D, B, x);
    result = dense_matrix_str(D);
    expected = "[1, 2]\n[0, 2*x]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    // jacobian
    dense_matrix_rows_cols(B, 2, 1);
    basic_add(e, x, y);
    dense_matrix_set_basic(B, 0, 0, e);
    basic_mul(e, x, y);
    dense_matrix_set_basic(B, 1, 0, e);
    dense_matrix_rows_cols(C, 2, 1);
    dense_matrix_set_basic(C, 0, 0, x);
    dense_matrix_set_basic(C, 1, 0, y);
    dense_matrix_rows_cols(D, 2, 2);
    dense_matrix_jacobian(D, B, C);
    result = dense_matrix_str(D);
    expected = "[1, 1]\n[y, x]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    basic_str_free(result);

    // row & col join
    symbol_set(x, "x");
    symbol_set(y, "y");
    integer_set_si(i2, 2);
    integer_set_si(i3, 3);
    dense_matrix_rows_cols(B, 2, 2);
    dense_matrix_set_basic(B, 0, 0, x);
    dense_matrix_set_basic(B, 0, 1, y);
    dense_matrix_set_basic(B, 1, 0, i2);
    dense_matrix_set_basic(B, 1, 1, i3);
    dense_matrix_rows_cols(C, 2, 2);
    dense_matrix_set_basic(C, 0, 0, y);
    dense_matrix_set_basic(C, 0, 1, x);
    dense_matrix_set_basic(C, 1, 0, i2);
    dense_matrix_set_basic(C, 1, 1, i3);
    dense_matrix_row_join(B, C);
    result = dense_matrix_str(B);
    expected = "[x, y, y, x]\n[2, 3, 2, 3]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    SYMENGINE_C_ASSERT(dense_matrix_rows(B) == 2);
    SYMENGINE_C_ASSERT(dense_matrix_cols(B) == 4);
    basic_str_free(result);
    dense_matrix_rows_cols(D, 2, 2);
    dense_matrix_set_basic(D, 0, 0, x);
    dense_matrix_set_basic(D, 0, 1, y);
    dense_matrix_set_basic(D, 1, 0, i2);
    dense_matrix_set_basic(D, 1, 1, i3);
    dense_matrix_col_join(D, C);
    result = dense_matrix_str(D);
    expected = "[x, y]\n[2, 3]\n[y, x]\n[2, 3]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    SYMENGINE_C_ASSERT(dense_matrix_rows(D) == 4);
    SYMENGINE_C_ASSERT(dense_matrix_cols(D) == 2);
    basic_str_free(result);
    dense_matrix_row_del(B, 1);
    result = dense_matrix_str(B);
    expected = "[x, y, y, x]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    SYMENGINE_C_ASSERT(dense_matrix_rows(B) == 1);
    SYMENGINE_C_ASSERT(dense_matrix_cols(B) == 4);
    basic_str_free(result);
    dense_matrix_col_del(D, 1);
    result = dense_matrix_str(D);
    expected = "[x]\n[2]\n[y]\n[2]\n";
    SYMENGINE_C_ASSERT(strcmp(result, expected) == 0);
    SYMENGINE_C_ASSERT(dense_matrix_rows(D) == 4);
    SYMENGINE_C_ASSERT(dense_matrix_cols(D) == 1);
    basic_str_free(result);

    basic_free_stack(x);
    basic_free_stack(y);
    basic_free_stack(e);

    vecbasic_free(vec);

    dense_matrix_free(B);
    dense_matrix_free(C);
    dense_matrix_free(D);
    dense_matrix_free(E);

    basic_free_stack(i1);
    basic_free_stack(i2);
    basic_free_stack(i3);
    basic_free_stack(i4);
}

void test_lambda_double()
{
    int perform_cse;
    double outs[2];
    double inps[3] = {1.5, 2.0, 3.0};

    basic two, x, y, z, r, s;
    basic_new_stack(two);
    basic_new_stack(x);
    basic_new_stack(y);
    basic_new_stack(z);
    basic_new_stack(r);
    basic_new_stack(s);
    CVecBasic *args = vecbasic_new();
    CVecBasic *exprs = vecbasic_new();

    integer_set_si(two, 2);
    symbol_set(x, "x");
    symbol_set(y, "y");
    symbol_set(z, "z");
    symbol_set(r, "r");
    symbol_set(s, "s");

    vecbasic_push_back(args, x);
    vecbasic_push_back(args, y);
    vecbasic_push_back(args, z);

    // r = x + y*z + (y*z)**2
    // s = 2*x + y*z + (y*z)**2
    basic_mul(y, y, z);
    basic_pow(z, y, two);
    basic_add(z, z, y);
    basic_add(r, x, z);
    basic_mul(x, two, x);
    basic_add(s, x, z);
    vecbasic_push_back(exprs, r);
    vecbasic_push_back(exprs, s);

    for (perform_cse = 0; perform_cse <= 1; ++perform_cse) {
        CLambdaRealDoubleVisitor *vis = lambda_real_double_visitor_new();
        lambda_real_double_visitor_init(vis, args, exprs, perform_cse);
        lambda_real_double_visitor_call(vis, outs, inps);
        lambda_real_double_visitor_free(vis);
        SYMENGINE_C_ASSERT(fabs(outs[0] - 43.5) < 1e-12);
        SYMENGINE_C_ASSERT(fabs(outs[1] - 45.0) < 1e-12);
#ifdef HAVE_SYMENGINE_LLVM
        // double
        int symbolic_cse = 1, opt_level = 2;
        CLLVMDoubleVisitor *vis2 = llvm_double_visitor_new();
        llvm_double_visitor_init(vis2, args, exprs, symbolic_cse, opt_level);
        llvm_double_visitor_call(vis2, outs, inps);
        llvm_double_visitor_free(vis2);
        SYMENGINE_C_ASSERT(fabs(outs[0] - 43.5) < 1e-12);
        SYMENGINE_C_ASSERT(fabs(outs[1] - 45.0) < 1e-12);

        // float
        float outs_f[2];
        float inps_f[3] = {1.5F, 2.0F, 3.0F};
        CLLVMFloatVisitor *vis2f = llvm_float_visitor_new();
        llvm_float_visitor_init(vis2f, args, exprs, symbolic_cse, opt_level);
        llvm_float_visitor_call(vis2f, outs_f, inps_f);
        llvm_float_visitor_free(vis2f);
        SYMENGINE_C_ASSERT(fabs(outs_f[0] - 43.5F) < 1e-6F);
        SYMENGINE_C_ASSERT(fabs(outs_f[1] - 45.0F) < 1e-6F);
#ifdef HAVE_SYMENGINE_LLVM_LONG_DOUBLE
        // long double
        long double outs_l[2];
        long double inps_l[3] = {1.5L, 2.0L, 3.0L};
        CLLVMLongDoubleVisitor *vis2l = llvm_long_double_visitor_new();
        llvm_long_double_visitor_init(vis2l, args, exprs, symbolic_cse,
                                      opt_level);
        llvm_long_double_visitor_call(vis2l, outs_l, inps_l);
        llvm_long_double_visitor_free(vis2l);
        SYMENGINE_C_ASSERT(fabs(outs_l[0] - 43.5L) < 1e-6L);
        SYMENGINE_C_ASSERT(fabs(outs_l[1] - 45.0L) < 1e-6L);
#endif
#endif
    }
    basic_free_stack(two);
    basic_free_stack(x);
    basic_free_stack(y);
    basic_free_stack(z);
    basic_free_stack(r);
    basic_free_stack(s);
    vecbasic_free(args);
    vecbasic_free(exprs);
}

void test_cse()
{
    basic x, y, z, r, s;
    basic_new_stack(x);
    basic_new_stack(y);
    basic_new_stack(z);
    basic_new_stack(r);
    basic_new_stack(s);

    symbol_set(x, "x");
    basic_sqrt(y, x);
    basic_sin(r, y);
    basic_cos(s, y);

    CVecBasic *exprs = vecbasic_new();
    CVecBasic *replacement_syms = vecbasic_new();
    CVecBasic *replacement_exprs = vecbasic_new();
    CVecBasic *reduced_exprs = vecbasic_new();
    vecbasic_push_back(exprs, r);
    vecbasic_push_back(exprs, s);

    basic_cse(replacement_syms, replacement_exprs, reduced_exprs, exprs);
    SYMENGINE_C_ASSERT(vecbasic_size(replacement_syms) == 1);
    vecbasic_get(replacement_exprs, 0, z);
    SYMENGINE_C_ASSERT(basic_eq(z, y));
    vecbasic_get(replacement_syms, 0, z);
    basic_sin(r, z);
    vecbasic_get(reduced_exprs, 0, s);
    SYMENGINE_C_ASSERT(basic_eq(r, s));
    basic_cos(r, z);
    vecbasic_get(reduced_exprs, 1, s);
    SYMENGINE_C_ASSERT(basic_eq(r, s));

    basic_free_stack(x);
    basic_free_stack(y);
    basic_free_stack(z);
    basic_free_stack(r);
    basic_free_stack(s);
    vecbasic_free(exprs);
    vecbasic_free(replacement_syms);
    vecbasic_free(replacement_exprs);
    vecbasic_free(reduced_exprs);
}

void test_sets()
{
    basic x, y, z, i, j, k;
    basic_new_stack(x);
    basic_new_stack(y);
    basic_new_stack(z);
    basic_new_stack(i);
    basic_new_stack(j);
    basic_new_stack(k);

    integer_set_ui(i, 1);
    integer_set_ui(j, 2);
    integer_set_ui(k, 3);
    basic_set_interval(x, i, j, 0, 0);
    basic_set_interval(y, j, k, 0, 0);
    basic_set_union(x, x, y);
    basic_set_interval(y, i, k, 0, 0);
    SYMENGINE_C_ASSERT(basic_eq(x, y));

    basic_set_emptyset(y);
    basic_set_intersection(x, x, y);
    SYMENGINE_C_ASSERT(basic_eq(x, y));

    basic_set_reals(x);
    basic_set_rationals(y);
    basic_set_union(y, x, y);
    SYMENGINE_C_ASSERT(basic_eq(x, y));

    basic_set_reals(x);
    basic_set_complexes(y);
    basic_set_union(x, x, y);
    SYMENGINE_C_ASSERT(basic_eq(x, y));

    basic_set_integers(x);
    basic_set_rationals(y);
    basic_set_union(x, x, y);
    SYMENGINE_C_ASSERT(basic_eq(x, y));

    basic_set_emptyset(x);
    basic_set_universalset(y);
    basic_set_union(x, x, y);
    SYMENGINE_C_ASSERT(basic_eq(x, y));

    basic_set_interval(x, i, j, 0, 0);
    basic_set_sup(y, x);
    SYMENGINE_C_ASSERT(basic_eq(y, j));
    basic_set_inf(y, x);
    SYMENGINE_C_ASSERT(basic_eq(y, i));

    basic_set_interval(x, i, j, 0, 0);
    SYMENGINE_C_ASSERT(is_a_Set(x));
    SYMENGINE_C_ASSERT(!is_a_Set(i));

    basic_set_interval(x, i, j, 0, 0);
    basic_set_interval(y, i, k, 0, 0);
    SYMENGINE_C_ASSERT(basic_set_is_subset(x, y));
    SYMENGINE_C_ASSERT(!basic_set_is_subset(y, x));
    SYMENGINE_C_ASSERT(basic_set_is_proper_subset(x, y));
    SYMENGINE_C_ASSERT(!basic_set_is_proper_subset(x, x));
    SYMENGINE_C_ASSERT(!basic_set_is_superset(x, y));
    SYMENGINE_C_ASSERT(basic_set_is_superset(y, x));
    SYMENGINE_C_ASSERT(basic_set_is_proper_superset(y, x));
    SYMENGINE_C_ASSERT(!basic_set_is_proper_superset(x, x));

    basic_set_rationals(x);
    basic_set_reals(y);
    basic_set_boundary(z, x);
    SYMENGINE_C_ASSERT(basic_eq(z, y));
    basic_set_closure(z, x);
    SYMENGINE_C_ASSERT(basic_eq(z, y));
    basic_set_interior(z, x);
    basic_set_emptyset(y);
    SYMENGINE_C_ASSERT(basic_eq(z, y));

    basic_set_interval(x, i, i, 0, 0);
    basic_set_interval(y, j, k, 0, 0);
    basic_set_complement(z, x, y);
    SYMENGINE_C_ASSERT(basic_eq(z, y));

    basic_set_interval(x, i, j, 0, 0);
    bool_set_true(y);
    basic_set_contains(z, x, i);
    SYMENGINE_C_ASSERT(basic_eq(z, y));
    basic_set_contains(z, x, k);
    bool_set_false(y);
    SYMENGINE_C_ASSERT(basic_eq(z, y));

    CSetBasic *set = setbasic_new();
    setbasic_insert(set, i);
    setbasic_insert(set, j);
    basic_set_finiteset(x, set);
    bool_set_true(y);
    basic_set_contains(z, x, i);
    SYMENGINE_C_ASSERT(basic_eq(z, y));
    bool_set_false(y);
    basic_set_contains(z, x, k);
    SYMENGINE_C_ASSERT(basic_eq(z, y));

    setbasic_free(set);
    basic_free_stack(k);
    basic_free_stack(j);
    basic_free_stack(i);
    basic_free_stack(z);
    basic_free_stack(y);
    basic_free_stack(x);
}

int main(int argc, char *argv[])
{
    symengine_print_stack_on_segfault();
    test_version();
    test_cwrapper();
    test_complex();
    test_complex_double();
    test_basic();
    test_CVectorInt1();
    test_CVectorInt2();
    test_CVecBasic();
    test_CSetBasic();
    test_CMapBasicBasic();
    test_get_args();
    test_free_symbols();
    test_function_symbols();
    test_function_symbol_get_name();
    test_get_type();
    test_hash();
    test_subs();
    test_subs2();
    test_coeff();
    test_linsolve();
    test_solve_poly();
    test_constants();
    test_infinity();
    test_nan();
    test_ascii_art();
    test_functions();
    test_ntheory();
    test_real_double();
    test_eval();
#ifdef HAVE_SYMENGINE_MPFR
    test_real_mpfr();
#endif // HAVE_SYMENGINE_MPFR
#ifdef HAVE_SYMENGINE_MPC
    test_complex_mpc();
#endif // HAVE_SYMENGINE_MPC
    test_matrix();
    test_lambda_double();
    test_cse();
    test_sets();
    return 0;
}
