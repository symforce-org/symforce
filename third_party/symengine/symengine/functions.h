/**
 *  \file functions.h
 *  Includes various trignometric functions
 *
 **/

#ifndef SYMENGINE_FUNCTIONS_H
#define SYMENGINE_FUNCTIONS_H

#include <symengine/basic.h>
#include <symengine/symengine_casts.h>
#include <symengine/constants.h>

namespace SymEngine
{

class Function : public Basic
{
};

class OneArgFunction : public Function
{
private:
    RCP<const Basic> arg_; //! The `arg` in `OneArgFunction(arg)`
public:
    //! Constructor
    OneArgFunction(const RCP<const Basic> &arg) : arg_{arg} {};
    //! \return the hash
    inline hash_t __hash__() const
    {
        hash_t seed = this->get_type_code();
        hash_combine<Basic>(seed, *arg_);
        return seed;
    }
    //! \return `arg_`
    inline RCP<const Basic> get_arg() const
    {
        return arg_;
    }
    virtual inline vec_basic get_args() const
    {
        return {arg_};
    }
    //! Method to construct classes with canonicalization
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const = 0;

    inline RCP<const Basic> create(const vec_basic &b) const
    {
        SYMENGINE_ASSERT(b.size() == 1);
        return create(b[0]);
    }

    /*! Equality comparator
     * \param o - Object to be compared with
     * \return whether the 2 objects are equal
     * */
    virtual inline bool __eq__(const Basic &o) const
    {
        return is_same_type(*this, o)
               and eq(*get_arg(),
                      *down_cast<const OneArgFunction &>(o).get_arg());
    }
    //! Structural equality comparator
    virtual inline int compare(const Basic &o) const
    {
        SYMENGINE_ASSERT(is_same_type(*this, o))
        return get_arg()->__cmp__(
            *(down_cast<const OneArgFunction &>(o).get_arg()));
    }
};

template <class BaseClass>
class TwoArgBasic : public BaseClass
{
private:
    RCP<const Basic> a_; //! `a` in `TwoArgBasic(a, b)`
    RCP<const Basic> b_; //! `b` in `TwoArgBasic(a, b)`
public:
    //! Constructor
    TwoArgBasic(const RCP<const Basic> &a, const RCP<const Basic> &b)
        : a_{a}, b_{b} {};
    //! \return the hash
    inline hash_t __hash__() const
    {
        hash_t seed = this->get_type_code();
        hash_combine<Basic>(seed, *a_);
        hash_combine<Basic>(seed, *b_);
        return seed;
    }
    //! \return `arg_`
    inline RCP<const Basic> get_arg1() const
    {
        return a_;
    }
    //! \return `arg_`
    inline RCP<const Basic> get_arg2() const
    {
        return b_;
    }
    virtual inline vec_basic get_args() const
    {
        return {a_, b_};
    }
    //! Method to construct classes with canonicalization
    virtual RCP<const Basic> create(const RCP<const Basic> &a,
                                    const RCP<const Basic> &b) const = 0;

    inline RCP<const Basic> create(const vec_basic &b) const
    {
        SYMENGINE_ASSERT(b.size() == 2);
        return create(b[0], b[1]);
    }

    /*! Equality comparator
     * \param o - Object to be compared with
     * \return whether the 2 objects are equal
     * */
    virtual inline bool __eq__(const Basic &o) const
    {
        return is_same_type(*this, o)
               and eq(*get_arg1(),
                      *down_cast<const TwoArgBasic &>(o).get_arg1())
               and eq(*get_arg2(),
                      *down_cast<const TwoArgBasic &>(o).get_arg2());
    }
    //! Structural equality comparator
    virtual inline int compare(const Basic &o) const
    {
        SYMENGINE_ASSERT(is_same_type(*this, o))
        const TwoArgBasic &t = down_cast<const TwoArgBasic &>(o);
        if (neq(*get_arg1(), *(t.get_arg1()))) {
            return get_arg1()->__cmp__(
                *(down_cast<const TwoArgBasic &>(o).get_arg1()));
        } else {
            return get_arg2()->__cmp__(
                *(down_cast<const TwoArgBasic &>(o).get_arg2()));
        }
    }
};

typedef TwoArgBasic<Function> TwoArgFunction;

class MultiArgFunction : public Function
{
private:
    vec_basic arg_;

public:
    //! Constructor
    MultiArgFunction(const vec_basic &arg) : arg_{arg} {};
    //! \return the hash
    inline hash_t __hash__() const
    {
        hash_t seed = this->get_type_code();
        for (const auto &a : arg_)
            hash_combine<Basic>(seed, *a);
        return seed;
    }
    virtual inline vec_basic get_args() const
    {
        return arg_;
    }
    inline const vec_basic &get_vec() const
    {
        return arg_;
    }
    //! Method to construct classes with canonicalization
    virtual RCP<const Basic> create(const vec_basic &v) const = 0;
    /*! Equality comparator
     * \param o - Object to be compared with
     * \return whether the 2 objects are equal
     * */
    virtual inline bool __eq__(const Basic &o) const
    {
        return is_same_type(*this, o)
               and unified_eq(get_vec(),
                              down_cast<const MultiArgFunction &>(o).get_vec());
    }
    //! Structural equality comparator
    virtual inline int compare(const Basic &o) const
    {
        SYMENGINE_ASSERT(is_same_type(*this, o))
        return unified_compare(
            get_vec(), down_cast<const MultiArgFunction &>(o).get_vec());
    }
};

class Sign : public OneArgFunction
{
public:
    IMPLEMENT_TYPEID(SYMENGINE_SIGN)
    //! Sign constructor
    Sign(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized sign
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize Sign
RCP<const Basic> sign(const RCP<const Basic> &arg);

class Floor : public OneArgFunction
{
public:
    IMPLEMENT_TYPEID(SYMENGINE_FLOOR)
    //! Floor Constructor
    Floor(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized floor
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize Floor:
RCP<const Basic> floor(const RCP<const Basic> &arg);

class Ceiling : public OneArgFunction
{
public:
    IMPLEMENT_TYPEID(SYMENGINE_CEILING)
    //! Ceiling Constructor
    Ceiling(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized ceiling
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize Ceiling:
RCP<const Basic> ceiling(const RCP<const Basic> &arg);

class Truncate : public OneArgFunction
{
public:
    IMPLEMENT_TYPEID(SYMENGINE_TRUNCATE)
    //! Truncate Constructor
    Truncate(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized truncate
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize Truncate:
RCP<const Basic> truncate(const RCP<const Basic> &arg);

class Conjugate : public OneArgFunction
{
public:
    IMPLEMENT_TYPEID(SYMENGINE_CONJUGATE)
    //! Conjugate constructor
    Conjugate(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized conjugate
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize Conjugate
RCP<const Basic> conjugate(const RCP<const Basic> &arg);

class TrigBase : public OneArgFunction
{
public:
    //! Constructor
    TrigBase(RCP<const Basic> arg) : OneArgFunction(arg){};
};

class TrigFunction : public TrigBase
{
public:
    //! Constructor
    TrigFunction(RCP<const Basic> arg) : TrigBase(arg){};
};

class InverseTrigFunction : public TrigBase
{
public:
    //! Constructor
    InverseTrigFunction(RCP<const Basic> arg) : TrigBase(arg){};
};

/*! \return `true` if `arg` is of form `m + n*pi` where `n` is a rational
 * */
bool get_pi_shift(const RCP<const Basic> &arg, const Ptr<RCP<const Number>> &n,
                  const Ptr<RCP<const Basic>> &m);

//! \return `true` if `arg` contains a negative sign.
bool could_extract_minus(const Basic &arg);

bool handle_minus(const RCP<const Basic> &arg,
                  const Ptr<RCP<const Basic>> &rarg);

/*! returns `true` if the given argument `t` is found in the
*   lookup table `d`. It also returns the value in `index`
**/
bool inverse_lookup(umap_basic_basic &d, const RCP<const Basic> &t,
                    const Ptr<RCP<const Basic>> &index);

// \return true of conjugate has to be returned finally else false
bool trig_simplify(const RCP<const Basic> &arg, unsigned period, bool odd,
                   bool conj_odd, // input
                   const Ptr<RCP<const Basic>> &rarg, int &index,
                   int &sign); // output

//! \return `sqrt` of the `arg`
RCP<const Basic> sqrt(const RCP<const Basic> &arg);

//! \return `cbrt` of the `arg`
RCP<const Basic> cbrt(const RCP<const Basic> &arg);

class Sin : public TrigFunction
{

public:
    IMPLEMENT_TYPEID(SYMENGINE_SIN)
    //! Sin Constructor
    Sin(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized sin
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize Sin:
RCP<const Basic> sin(const RCP<const Basic> &arg);

class Cos : public TrigFunction
{

public:
    IMPLEMENT_TYPEID(SYMENGINE_COS)
    //! Cos Constructor
    Cos(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized cos
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize Cos:
RCP<const Basic> cos(const RCP<const Basic> &arg);

class Tan : public TrigFunction
{

public:
    IMPLEMENT_TYPEID(SYMENGINE_TAN)
    //! Tan Constructor
    Tan(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized tan
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};
//! Canonicalize Tan:
RCP<const Basic> tan(const RCP<const Basic> &arg);

class Cot : public TrigFunction
{

public:
    IMPLEMENT_TYPEID(SYMENGINE_COT)
    //! Cot Constructor
    Cot(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized cot
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};
//! Canonicalize Cot:
RCP<const Basic> cot(const RCP<const Basic> &arg);

class Csc : public TrigFunction
{

public:
    IMPLEMENT_TYPEID(SYMENGINE_CSC)
    //! Csc Constructor
    Csc(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized csc
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};
//! Canonicalize Csc:
RCP<const Basic> csc(const RCP<const Basic> &arg);

class Sec : public TrigFunction
{

public:
    IMPLEMENT_TYPEID(SYMENGINE_SEC)
    //! Sec Constructor
    Sec(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized sec
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};
//! Canonicalize Sec:
RCP<const Basic> sec(const RCP<const Basic> &arg);

class ASin : public InverseTrigFunction
{

public:
    IMPLEMENT_TYPEID(SYMENGINE_ASIN)
    //! ASin Constructor
    ASin(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized asin
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize ASin:
RCP<const Basic> asin(const RCP<const Basic> &arg);

class ACos : public InverseTrigFunction
{

public:
    IMPLEMENT_TYPEID(SYMENGINE_ACOS)
    //! ACos Constructor
    ACos(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized acos
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize ACos:
RCP<const Basic> acos(const RCP<const Basic> &arg);

class ASec : public InverseTrigFunction
{

public:
    IMPLEMENT_TYPEID(SYMENGINE_ASEC)
    //! ASec Constructor
    ASec(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized asec
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize ASec:
RCP<const Basic> asec(const RCP<const Basic> &arg);

class ACsc : public InverseTrigFunction
{

public:
    IMPLEMENT_TYPEID(SYMENGINE_ACSC)
    //! ACsc Constructor
    ACsc(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized acsc
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize ACsc:
RCP<const Basic> acsc(const RCP<const Basic> &arg);

class ATan : public InverseTrigFunction
{

public:
    IMPLEMENT_TYPEID(SYMENGINE_ATAN)
    //! ATan Constructor
    ATan(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized atan
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize ATan:
RCP<const Basic> atan(const RCP<const Basic> &arg);

class ACot : public InverseTrigFunction
{

public:
    IMPLEMENT_TYPEID(SYMENGINE_ACOT)
    //! ACot Constructor
    ACot(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized acot
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize ACot:
RCP<const Basic> acot(const RCP<const Basic> &arg);

class ATan2 : public TwoArgFunction
{
public:
    IMPLEMENT_TYPEID(SYMENGINE_ATAN2)
    //! ATan2 Constructor
    ATan2(const RCP<const Basic> &num, const RCP<const Basic> &den);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &num,
                      const RCP<const Basic> &den) const;
    //! \return `y` in `atan2(y, x)`
    inline RCP<const Basic> get_num() const
    {
        return get_arg1();
    }
    //! \return `x` in `atan2(y, x)`
    inline RCP<const Basic> get_den() const
    {
        return get_arg2();
    }
    //! \return canonicalized `atan2`
    virtual RCP<const Basic> create(const RCP<const Basic> &a,
                                    const RCP<const Basic> &b) const;
};

//! Canonicalize ATan2:
RCP<const Basic> atan2(const RCP<const Basic> &num,
                       const RCP<const Basic> &den);

class Log : public OneArgFunction
{
    // Logarithms are taken with the natural base, `e`. To get
    // a logarithm of a different base `b`, use `log(x, b)`,
    // which is essentially short-hand for `log(x)/log(b)`.
public:
    IMPLEMENT_TYPEID(SYMENGINE_LOG)
    //! Log Constructor
    Log(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return canonicalized `log`
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Returns the Natural Logarithm from argument `arg`
RCP<const Basic> log(const RCP<const Basic> &arg);
//! \return Log from argument `arg` wrt base `b`
RCP<const Basic> log(const RCP<const Basic> &arg, const RCP<const Basic> &b);

class LambertW : public OneArgFunction
{
    // Lambert W function, defined as the inverse function of
    // x*exp(x). This function represents the principal branch
    // of this inverse function, which is multivalued.
    // For more information, see:
    // http://en.wikipedia.org/wiki/Lambert_W_function
public:
    IMPLEMENT_TYPEID(SYMENGINE_LAMBERTW)
    //! LambertW Constructor
    LambertW(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return canonicalized lambertw
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Create a new LambertW instance:
RCP<const Basic> lambertw(const RCP<const Basic> &arg);

class Zeta : public TwoArgFunction
{
    // Hurwitz zeta function (or Riemann zeta function).
    //
    // For `\operatorname{Re}(a) > 0` and `\operatorname{Re}(s) > 1`, this
    // function is defined as
    //
    // .. math:: \zeta(s, a) = \sum_{n=0}^\infty \frac{1}{(n + a)^s},
    //
    // where the standard choice of argument for :math:`n + a` is used.
    // If no value is passed for :math:`a`, by this function assumes a default
    // value
    // of :math:`a = 1`, yielding the Riemann zeta function.
public:
    using TwoArgFunction::create;
    IMPLEMENT_TYPEID(SYMENGINE_ZETA)
    //! Zeta Constructor
    Zeta(const RCP<const Basic> &s, const RCP<const Basic> &a);
    //! Zeta Constructor
    Zeta(const RCP<const Basic> &s);
    //! \return `s_`
    inline RCP<const Basic> get_s() const
    {
        return get_arg1();
    }
    //! \return `a_`
    inline RCP<const Basic> get_a() const
    {
        return get_arg2();
    }
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &s,
                      const RCP<const Basic> &a) const;
    //! \return canonicalized `zeta`
    virtual RCP<const Basic> create(const RCP<const Basic> &a,
                                    const RCP<const Basic> &b) const;
};

//! Create a new Zeta instance:
RCP<const Basic> zeta(const RCP<const Basic> &s, const RCP<const Basic> &a);
RCP<const Basic> zeta(const RCP<const Basic> &s);

class Dirichlet_eta : public OneArgFunction
{
    // See http://en.wikipedia.org/wiki/Dirichlet_eta_function
public:
    IMPLEMENT_TYPEID(SYMENGINE_DIRICHLET_ETA)
    //! Dirichlet_eta Constructor
    Dirichlet_eta(const RCP<const Basic> &s);
    //! \return `s`
    inline RCP<const Basic> get_s() const
    {
        return get_arg();
    }
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &s) const;
    //! Rewrites in the form of zeta
    RCP<const Basic> rewrite_as_zeta() const;
    //! \return Canonicalized zeta
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
    using OneArgFunction::create;
};

//! Create a new Dirichlet_eta instance:
RCP<const Basic> dirichlet_eta(const RCP<const Basic> &s);

class FunctionSymbol : public MultiArgFunction
{
protected:
    std::string name_; //! The `f` in `f(x+y, z)`

public:
    IMPLEMENT_TYPEID(SYMENGINE_FUNCTIONSYMBOL)
    //! FunctionSymbol Constructors
    FunctionSymbol(std::string name, const vec_basic &arg);
    FunctionSymbol(std::string name, const RCP<const Basic> &arg);
    //! \return Size of the hash
    virtual hash_t __hash__() const;
    /*! Equality comparator
     * \param o  Object to be compared with
     * \return whether the 2 objects are equal
     * */
    virtual bool __eq__(const Basic &o) const;
    virtual int compare(const Basic &o) const;
    //! \return `name_`
    inline const std::string &get_name() const
    {
        return name_;
    }
    //! \return `true` if canonical
    bool is_canonical(const vec_basic &arg) const;
    virtual RCP<const Basic> create(const vec_basic &x) const;
};

//! Create a new FunctionSymbol instance:
RCP<const Basic> function_symbol(std::string name, const RCP<const Basic> &arg);
RCP<const Basic> function_symbol(std::string name, const vec_basic &arg);

/*! Use this class to define custom functions by overriding
 *  the defaut behaviour for create, eval, diff, __eq__, compare etc.
* */

class FunctionWrapper : public FunctionSymbol
{
public:
    IMPLEMENT_TYPEID(SYMENGINE_FUNCTIONWRAPPER)
    FunctionWrapper(std::string name, const vec_basic &arg);
    FunctionWrapper(std::string name, const RCP<const Basic> &arg);
    virtual RCP<const Basic> create(const vec_basic &v) const = 0;
    virtual RCP<const Number> eval(long bits) const = 0;
    virtual RCP<const Basic> diff_impl(const RCP<const Symbol> &s) const = 0;
};

/*! Derivative operator
 *  Derivative(f, [x, y, ...]) represents a derivative of `f` with respect to
 *  `x`, `y`, and so on.
 * */
class Derivative : public Basic
{
private:
    RCP<const Basic> arg_; //! The expression to be differentiated
    // The symbols are declared as Basic (and checked by is_canonical() below),
    // to avoid issues with converting vector<RCP<Symbol>> to
    // vector<RCP<Basic>>, see [1], [2]. The problem is that even though Symbol
    // inherits from Basic, vector<RCP<Symbol>> does not inherit from
    // vector<RCP<Basic>>, so the compiler can't cast the derived type to the
    // base type when calling functions like unified_eq() that are only
    // defined for the base type vector<RCP<Basic>>.
    // [1]
    // http://stackoverflow.com/questions/14964909/how-to-cast-a-vector-of-shared-ptrs-of-a-derived-class-to-a-vector-of-share-ptrs
    // [2]
    // http://stackoverflow.com/questions/114819/getting-a-vectorderived-into-a-function-that-expects-a-vectorbase
    multiset_basic x_; //! x, y, ...

public:
    IMPLEMENT_TYPEID(SYMENGINE_DERIVATIVE)
    Derivative(const RCP<const Basic> &arg, const multiset_basic &x);

    static RCP<const Derivative> create(const RCP<const Basic> &arg,
                                        const multiset_basic &x)
    {
        return make_rcp<const Derivative>(arg, x);
    }

    virtual hash_t __hash__() const;
    virtual bool __eq__(const Basic &o) const;
    virtual int compare(const Basic &o) const;
    inline RCP<const Basic> get_arg() const
    {
        return arg_;
    }
    inline const multiset_basic &get_symbols() const
    {
        return x_;
    }
    virtual vec_basic get_args() const
    {
        vec_basic args = {arg_};
        args.insert(args.end(), x_.begin(), x_.end());
        return args;
    }
    bool is_canonical(const RCP<const Basic> &arg,
                      const multiset_basic &x) const;
};

/*! Subs operator
 *  Subs(f, {x1 : x2, y1: y2, ...}) represents `f` after substituting
 *  `x1` with `x2`, `y1` with `y2`, and so on.
 * */
class Subs : public Basic
{
private:
    RCP<const Basic> arg_;
    map_basic_basic dict_;

public:
    IMPLEMENT_TYPEID(SYMENGINE_SUBS)
    Subs(const RCP<const Basic> &arg, const map_basic_basic &x);

    static RCP<const Subs> create(const RCP<const Basic> &arg,
                                  const map_basic_basic &x)
    {
        return make_rcp<const Subs>(arg, x);
    }

    virtual hash_t __hash__() const;
    virtual bool __eq__(const Basic &o) const;
    virtual int compare(const Basic &o) const;
    inline const RCP<const Basic> &get_arg() const
    {
        return arg_;
    }
    inline const map_basic_basic &get_dict() const
    {
        return dict_;
    };
    virtual vec_basic get_variables() const;
    virtual vec_basic get_point() const;
    virtual vec_basic get_args() const;

    bool is_canonical(const RCP<const Basic> &arg,
                      const map_basic_basic &x) const;
};

class HyperbolicBase : public OneArgFunction
{
public:
    //! Constructor
    HyperbolicBase(RCP<const Basic> arg) : OneArgFunction{arg} {};
};

class HyperbolicFunction : public HyperbolicBase
{
public:
    //! Constructor
    HyperbolicFunction(RCP<const Basic> arg) : HyperbolicBase{arg} {};
};

class InverseHyperbolicFunction : public HyperbolicBase
{
public:
    //! Constructor
    InverseHyperbolicFunction(RCP<const Basic> arg) : HyperbolicBase{arg} {};
};

class Sinh : public HyperbolicFunction
{
    //! The hyperbolic sine function, `\frac{e^x - e^{-x}}{2}`.
public:
    IMPLEMENT_TYPEID(SYMENGINE_SINH)
    //! Sinh Constructor
    Sinh(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized sinh
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize Sinh:
RCP<const Basic> sinh(const RCP<const Basic> &arg);

class Csch : public HyperbolicFunction
{
    //! The hyperbolic cosecant function, `\frac{2}{e^x - e^{-x}}`.
public:
    IMPLEMENT_TYPEID(SYMENGINE_CSCH)
    //! Csch Constructor
    Csch(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized csch
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize Csch:
RCP<const Basic> csch(const RCP<const Basic> &arg);

class Cosh : public HyperbolicFunction
{
    //! The hyperbolic cosine function, `\frac{e^x + e^{-x}}{2}`.
public:
    IMPLEMENT_TYPEID(SYMENGINE_COSH)
    //! Cosh Constructor
    Cosh(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized cosh
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize Cosh:
RCP<const Basic> cosh(const RCP<const Basic> &arg);

class Sech : public HyperbolicFunction
{
    //! The hyperbolic secant function, `\frac{2}{e^x + e^{-x}}`.
public:
    IMPLEMENT_TYPEID(SYMENGINE_SECH)
    //! Sech Constructor
    Sech(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized sech
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize Sech:
RCP<const Basic> sech(const RCP<const Basic> &arg);

class Tanh : public HyperbolicFunction
{
    //! The hyperbolic tangent function, `\frac{\sinh(x)}{\cosh(x)}`.
public:
    IMPLEMENT_TYPEID(SYMENGINE_TANH)
    //! Tanh Constructor
    Tanh(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized tanh
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize Tanh:
RCP<const Basic> tanh(const RCP<const Basic> &arg);

class Coth : public HyperbolicFunction
{
    //! The hyperbolic tangent function, `\frac{\cosh(x)}{\sinh(x)}`.
public:
    IMPLEMENT_TYPEID(SYMENGINE_COTH)
    //! Coth Constructor
    Coth(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized coth
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize Coth:
RCP<const Basic> coth(const RCP<const Basic> &arg);

class ASinh : public InverseHyperbolicFunction
{
    //! The inverse hyperbolic sine function.
public:
    IMPLEMENT_TYPEID(SYMENGINE_ASINH)
    //! ASinh Constructor
    ASinh(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized asinh
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize ASinh:
RCP<const Basic> asinh(const RCP<const Basic> &arg);

class ACsch : public InverseHyperbolicFunction
{
    //! The inverse hyperbolic cosecant function.
public:
    IMPLEMENT_TYPEID(SYMENGINE_ACSCH)
    //! ACsch Constructor
    ACsch(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized acsch
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize ACsch:
RCP<const Basic> acsch(const RCP<const Basic> &arg);

class ACosh : public InverseHyperbolicFunction
{
    //! The inverse hyperbolic cosine function.
public:
    IMPLEMENT_TYPEID(SYMENGINE_ACOSH)
    //! ACosh Constructor
    ACosh(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized acosh
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize ACosh:
RCP<const Basic> acosh(const RCP<const Basic> &arg);

class ATanh : public InverseHyperbolicFunction
{
    //! The inverse hyperbolic tangent function.
public:
    IMPLEMENT_TYPEID(SYMENGINE_ATANH)
    //! ATanh Constructor
    ATanh(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized atanh
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize ATanh:
RCP<const Basic> atanh(const RCP<const Basic> &arg);

class ACoth : public InverseHyperbolicFunction
{
    //! The inverse hyperbolic cotangent function.
public:
    IMPLEMENT_TYPEID(SYMENGINE_ACOTH)
    //! ACoth Constructor
    ACoth(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized acoth
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize ACoth:
RCP<const Basic> acoth(const RCP<const Basic> &arg);

class ASech : public InverseHyperbolicFunction
{
    //! The inverse hyperbolic secant function.
public:
    IMPLEMENT_TYPEID(SYMENGINE_ASECH)
    //! ASech Constructor
    ASech(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized asech
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize ASech:
RCP<const Basic> asech(const RCP<const Basic> &arg);

class KroneckerDelta : public TwoArgFunction
{
    /*! The discrete, or Kronecker, delta function.
     * A function that takes in two integers `i` and `j`. It returns `0` if `i`
     *and `j` are
     * not equal or it returns `1` if `i` and `j` are equal.
     * http://en.wikipedia.org/wiki/Kronecker_delta
     **/
public:
    using TwoArgFunction::create;
    IMPLEMENT_TYPEID(SYMENGINE_KRONECKERDELTA)
    //! KroneckerDelta Constructor
    KroneckerDelta(const RCP<const Basic> &i, const RCP<const Basic> &j);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &i,
                      const RCP<const Basic> &j) const;
    //! \return canonicalized `KroneckerDelta`
    virtual RCP<const Basic> create(const RCP<const Basic> &a,
                                    const RCP<const Basic> &b) const;
};

//! Canonicalize KroneckerDelta:
RCP<const Basic> kronecker_delta(const RCP<const Basic> &i,
                                 const RCP<const Basic> &j);

class LeviCivita : public MultiArgFunction
{
    /*! Represent the Levi-Civita symbol.
     *  For even permutations of indices it returns 1, for odd permutations -1,
     *and
     *  for everything else (a repeated index) it returns 0.
     *
     *  Thus it represents an alternating pseudotensor.
     **/
public:
    IMPLEMENT_TYPEID(SYMENGINE_LEVICIVITA)
    //! LeviCivita Constructor
    LeviCivita(const vec_basic &&arg);
    //! \return `true` if canonical
    bool is_canonical(const vec_basic &arg) const;
    //! \return canonicalized Max
    virtual RCP<const Basic> create(const vec_basic &arg) const;
};

//! Canonicalize LeviCivita:
RCP<const Basic> levi_civita(const vec_basic &arg);

class Erf : public OneArgFunction
{
    /*   The Gauss error function. This function is defined as:
     *
     *   .. math::
     *      \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2}
     *\mathrm{d}t.
     **/
public:
    IMPLEMENT_TYPEID(SYMENGINE_ERF)
    //! Erf Constructor
    Erf(const RCP<const Basic> &arg) : OneArgFunction(arg)
    {
        SYMENGINE_ASSIGN_TYPEID()
        SYMENGINE_ASSERT(is_canonical(arg))
    }
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized erf
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize Erf:
RCP<const Basic> erf(const RCP<const Basic> &arg);

class Erfc : public OneArgFunction
{
    /*   The complementary error function. This function is defined as:
     *
     *   .. math::
     *      \mathrm{erfc}(x) = 1 - \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2}
     *	 	 \mathrm{d}t.
     **/
public:
    IMPLEMENT_TYPEID(SYMENGINE_ERFC)
    //! Erfc Constructor
    Erfc(const RCP<const Basic> &arg) : OneArgFunction(arg)
    {
        SYMENGINE_ASSIGN_TYPEID()
        SYMENGINE_ASSERT(is_canonical(arg))
    }
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized erfc
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize Erfc:
RCP<const Basic> erfc(const RCP<const Basic> &arg);

class Gamma : public OneArgFunction
{
    /*!    The gamma function
     *
     *   .. math::
     *      \Gamma(x) := \int^{\infty}_{0} t^{x-1} e^{t} \mathrm{d}t.
     *
     *  The ``gamma`` function implements the function which passes through the
     *  values of the factorial function, i.e. `\Gamma(n) = (n - 1)!` when n is
     *  an integer. More general, `\Gamma(z)` is defined in the whole complex
     *  plane except at the negative integers where there are simple poles.
     **/
public:
    IMPLEMENT_TYPEID(SYMENGINE_GAMMA)
    //! Gamma Constructor
    Gamma(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized gamma
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize Gamma:
RCP<const Basic> gamma(const RCP<const Basic> &arg);

class LowerGamma : public TwoArgFunction
{
    //! The lower incomplete gamma function.
public:
    using TwoArgFunction::create;
    IMPLEMENT_TYPEID(SYMENGINE_LOWERGAMMA)
    //! LowerGamma Constructor
    LowerGamma(const RCP<const Basic> &s, const RCP<const Basic> &x);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &s,
                      const RCP<const Basic> &x) const;
    //! \return canonicalized `LowerGamma`
    virtual RCP<const Basic> create(const RCP<const Basic> &a,
                                    const RCP<const Basic> &b) const;
};

//! Canonicalize LowerGamma:
RCP<const Basic> lowergamma(const RCP<const Basic> &s,
                            const RCP<const Basic> &x);

class UpperGamma : public TwoArgFunction
{
    //! The upper incomplete gamma function.
public:
    using TwoArgFunction::create;
    IMPLEMENT_TYPEID(SYMENGINE_UPPERGAMMA)
    //! UpperGamma Constructor
    UpperGamma(const RCP<const Basic> &s, const RCP<const Basic> &x);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &s,
                      const RCP<const Basic> &x) const;
    //! \return canonicalized `UpperGamma`
    virtual RCP<const Basic> create(const RCP<const Basic> &a,
                                    const RCP<const Basic> &b) const;
};

//! Canonicalize UpperGamma:
RCP<const Basic> uppergamma(const RCP<const Basic> &s,
                            const RCP<const Basic> &x);

class LogGamma : public OneArgFunction
{
    /*!    The loggamma function
        The `loggamma` function implements the logarithm of the
        gamma function i.e, `\log\Gamma(x)`.
     **/
public:
    IMPLEMENT_TYPEID(SYMENGINE_LOGGAMMA)
    //! LogGamma Constructor
    LogGamma(const RCP<const Basic> &arg) : OneArgFunction(arg)
    {
        SYMENGINE_ASSIGN_TYPEID()
        SYMENGINE_ASSERT(is_canonical(arg))
    }
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    RCP<const Basic> rewrite_as_gamma() const;
    //! \return canonicalized loggamma
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize LogGamma:
RCP<const Basic> loggamma(const RCP<const Basic> &arg);

class Beta : public TwoArgFunction
{
    /*!    The beta function, also called the Euler integral
     *     of the first kind, is a special function defined by
     *
     *   .. math::
     *      \Beta(x, y) := \int^{1}_{0} t^{x-1} (1-t)^{y-1} \mathrm{d}t.
     **/
public:
    using TwoArgFunction::create;
    IMPLEMENT_TYPEID(SYMENGINE_BETA)
    //! Beta Constructor
    Beta(const RCP<const Basic> &x, const RCP<const Basic> &y)
        : TwoArgFunction(x, y)
    {
        SYMENGINE_ASSIGN_TYPEID()
        SYMENGINE_ASSERT(is_canonical(x, y))
    }
    //! return `Beta` with ordered arguments
    static RCP<const Beta> from_two_basic(const RCP<const Basic> &x,
                                          const RCP<const Basic> &y);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &s, const RCP<const Basic> &x);
    RCP<const Basic> rewrite_as_gamma() const;
    //! \return canonicalized `Beta`
    virtual RCP<const Basic> create(const RCP<const Basic> &a,
                                    const RCP<const Basic> &b) const;
};

//! Canonicalize Beta:
RCP<const Basic> beta(const RCP<const Basic> &x, const RCP<const Basic> &y);

class PolyGamma : public TwoArgFunction
{
    /*!    The polygamma function
     *
     *     It is a meromorphic function on `\mathbb{C}` and defined as the
     *(n+1)-th
     *     derivative of the logarithm of the gamma function:
     *
     *  .. math::
     *  \psi^{(n)} (z) := \frac{\mathrm{d}^{n+1}}{\mathrm{d} z^{n+1}}
     *\log\Gamma(z).
     **/
public:
    using TwoArgFunction::create;
    IMPLEMENT_TYPEID(SYMENGINE_POLYGAMMA)
    //! PolyGamma Constructor
    PolyGamma(const RCP<const Basic> &n, const RCP<const Basic> &x)
        : TwoArgFunction(n, x)
    {
        SYMENGINE_ASSIGN_TYPEID()
        SYMENGINE_ASSERT(is_canonical(n, x))
    }
    bool is_canonical(const RCP<const Basic> &n, const RCP<const Basic> &x);
    RCP<const Basic> rewrite_as_zeta() const;
    //! \return canonicalized `PolyGamma`
    virtual RCP<const Basic> create(const RCP<const Basic> &a,
                                    const RCP<const Basic> &b) const;
};

//! Canonicalize PolyGamma
RCP<const Basic> polygamma(const RCP<const Basic> &n,
                           const RCP<const Basic> &x);

RCP<const Basic> digamma(const RCP<const Basic> &x);

RCP<const Basic> trigamma(const RCP<const Basic> &x);

class Abs : public OneArgFunction
{
    /*!    The absolute value function
     **/
public:
    IMPLEMENT_TYPEID(SYMENGINE_ABS)
    //! Abs Constructor
    Abs(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return canonicalized abs
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

//! Canonicalize Abs:
RCP<const Basic> abs(const RCP<const Basic> &arg);

class Max : public MultiArgFunction
{
public:
    IMPLEMENT_TYPEID(SYMENGINE_MAX)
    //! Max Constructor
    Max(const vec_basic &&arg);
    //! \return `true` if canonical
    bool is_canonical(const vec_basic &arg) const;
    //! \return canonicalized Max
    virtual RCP<const Basic> create(const vec_basic &arg) const;
};

//! Canonicalize Max:
RCP<const Basic> max(const vec_basic &arg);

class Min : public MultiArgFunction
{
public:
    IMPLEMENT_TYPEID(SYMENGINE_MIN)
    //! Min Constructor
    Min(const vec_basic &&arg);
    //! \return `true` if canonical
    bool is_canonical(const vec_basic &arg) const;
    //! \return canonicalized Max
    virtual RCP<const Basic> create(const vec_basic &arg) const;
};

//! Canonicalize Min:
RCP<const Basic> min(const vec_basic &arg);

//! \return simplified form if possible
RCP<const Basic> trig_to_sqrt(const RCP<const Basic> &arg);

class UnevaluatedExpr : public OneArgFunction
{
public:
    IMPLEMENT_TYPEID(SYMENGINE_UNEVALUATED_EXPR)
    //! UnevaluatedExpr Constructor
    UnevaluatedExpr(const RCP<const Basic> &arg);
    //! \return `true` if canonical
    bool is_canonical(const RCP<const Basic> &arg) const;
    //! \return Canonicalized UnevaluatedExpr
    virtual RCP<const Basic> create(const RCP<const Basic> &arg) const;
};

RCP<const Basic> unevaluated_expr(const RCP<const Basic> &arg);

} // SymEngine

#endif
