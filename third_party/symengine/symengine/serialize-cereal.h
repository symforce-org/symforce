#ifndef SYMENGINE_SERIALIZE_CEREAL_H
#define SYMENGINE_SERIALIZE_CEREAL_H
#include <symengine/basic.h>
#include <symengine/number.h>
#include <symengine/integer.h>
#include <symengine/symbol.h>
#include <symengine/visitor.h>
#include <symengine/utilities/stream_fmt.h>

#include <cereal/cereal.hpp>
#include <cereal/version.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>
#include <cereal/details/helpers.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/set.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/archives/portable_binary.hpp>

namespace SymEngine
{

template <class Archive>
inline void save_basic(Archive &ar, const Basic &b)
{
    const auto t_code = b.get_type_code();
    throw SerializationError(StreamFmt()
                             << __FILE__ << ":" << __LINE__
#ifndef _MSC_VER
                             << ": " << __PRETTY_FUNCTION__
#endif
                             << " not supported: " << type_code_name(t_code)
                             << " (" << t_code << ")"
#if !defined(NDEBUG)
                             << ", " << b.__str__()
#endif
    );
}
template <class Archive>
inline void save_basic(Archive &ar, const Symbol &b)
{
    ar(b.__str__());
}
template <class Archive>
inline void save_basic(Archive &ar, const Mul &b)
{
    ar(b.get_coef());
    ar(b.get_dict());
}
template <class Archive>
inline void save_basic(Archive &ar, const Add &b)
{
    ar(b.get_coef());
    ar(b.get_dict());
}
template <class Archive>
inline void save_basic(Archive &ar, const Pow &b)
{
    ar(b.get_base());
    ar(b.get_exp());
}
template <typename Archive>
void save_helper(Archive &ar, const integer_class &intgr)
{
    std::ostringstream s;
    s << intgr; // stream to string
    ar(s.str());
}
template <typename Archive>
void save_helper(Archive &ar, const rational_class &rat)
{
    integer_class num = get_num(rat);
    integer_class den = get_den(rat);
    save_helper(ar, num);
    save_helper(ar, den);
}
// Following is an ugly hack for templated integer classes
// Not sure why a direct version doesn't work
template <typename Archive>
void save_basic(Archive &ar, const URatPoly &b)
{
    ar(b.get_var());
    const URatDict &urd = b.get_poly();
    size_t l = urd.size();
    ar(l);
    for (auto &p : urd.dict_) {
        unsigned int first = p.first;
        const rational_class &second = p.second;
        ar(first);
        save_helper(ar, second);
    }
}
template <class Archive>
inline void save_basic(Archive &ar, const Integer &b)
{
    ar(b.__str__());
}
template <class Archive>
inline void save_basic(Archive &ar, const RealDouble &b)
{
    ar(b.i);
}
template <class Archive>
inline void save_basic(Archive &ar, const Rational &b)
{
    ar(b.get_num(), b.get_den());
}
template <class Archive>
inline void save_basic(Archive &ar, const ComplexBase &b)
{
    ar(b.real_part(), b.imaginary_part());
}
template <class Archive>
inline void save_basic(Archive &ar, const Interval &b)
{
    ar(b.get_left_open(), b.get_start(), b.get_right_open(), b.get_end());
}
template <class Archive>
inline void save_basic(Archive &ar, const BooleanAtom &b)
{
    ar(b.get_val());
}
template <class Archive>
inline void save_basic(Archive &ar, const Infty &b)
{
    ar(b.get_direction());
}

template <class Archive>
inline void save_basic(Archive &ar, const NaN &b)
{
}

template <class Archive>
inline void save_basic(Archive &ar, const Constant &b)
{
    ar(b.get_name());
}
template <class Archive>
inline void save_basic(Archive &ar, const OneArgFunction &b)
{
    ar(b.get_arg());
}
template <class Archive>
inline void save_basic(Archive &ar, const TwoArgFunction &b)
{
    ar(b.get_arg1(), b.get_arg2());
}

template <class Archive>
inline void save_basic(Archive &ar, const Relational &b)
{
    ar(b.get_arg1(), b.get_arg2());
}
template <class Archive>
inline void save_basic(Archive &ar, const And &b)
{
    ar(b.get_container());
}
template <class Archive>
inline void save_basic(Archive &ar, const Or &b)
{
    ar(b.get_container());
}
template <class Archive>
inline void save_basic(Archive &ar, const Xor &b)
{
    ar(b.get_container());
}
template <class Archive>
inline void save_basic(Archive &ar, const Not &b)
{
    ar(b.get_arg());
}
template <class Archive>
inline void save_basic(Archive &ar, const Contains &b)
{
    ar(b.get_expr(), b.get_set());
}
template <class Archive>
inline void save_basic(Archive &ar, const Piecewise &b)
{
    ar(b.get_vec());
}
template <class Archive>
inline void save_basic(Archive &ar, const Reals &b)
{
}
template <class Archive>
inline void save_basic(Archive &ar, const Rationals &b)
{
}
template <class Archive>
inline void save_basic(Archive &ar, const EmptySet &b)
{
}
template <class Archive>
inline void save_basic(Archive &ar, const Integers &b)
{
}
template <class Archive>
inline void save_basic(Archive &ar, const UniversalSet &b)
{
}
template <class Archive>
inline void save_basic(Archive &ar, const Union &b)
{
    ar(b.get_container());
}
template <class Archive>
inline void save_basic(Archive &ar, const Complement &b)
{
    ar(b.get_universe(), b.get_container());
}
template <class Archive>
inline void save_basic(Archive &ar, const ImageSet &b)
{
    ar(b.get_symbol(), b.get_expr(), b.get_baseset());
}
template <class Archive>
inline void save_basic(Archive &ar, const FiniteSet &b)
{
    ar(b.get_container());
}
template <class Archive>
inline void save_basic(Archive &ar, const ConditionSet &b)
{
    ar(b.get_symbol(), b.get_condition());
}
#ifdef HAVE_SYMENGINE_MPFR
template <class Archive>
inline void save_basic(Archive &ar, const RealMPFR &b)
{
    ar(b.__str__(), b.get_prec());
}
#endif
template <class Archive>
inline void save_basic(Archive &ar, const GaloisField &b)
{
    throw NotImplementedError("GaloisField saving is not implemented yet.");
}
template <class Archive>
inline void save_basic(Archive &ar, const SeriesCoeffInterface &)
{
    throw NotImplementedError("Series saving is not implemented yet.");
}
template <class Archive>
inline void save_basic(Archive &ar, const MultiArgFunction &b)
{
    ar(b.get_args());
}
template <class Archive>
inline void save_basic(Archive &ar, const FunctionSymbol &b)
{
    ar(b.get_name(), b.get_args());
}
template <class Archive>
inline void save_basic(Archive &ar, const Derivative &b)
{
    ar(b.get_arg(), b.get_symbols());
}
template <class Archive>
inline void save_basic(Archive &ar, const Subs &b)
{
    ar(b.get_arg(), b.get_dict());
}
template <class Archive>
inline void save_basic(Archive &ar, const NumberWrapper &b)
{
    throw NotImplementedError("NumberWrapper saving is not implemented yet.");
}
template <class Archive>
inline void save_basic(Archive &ar, const FunctionWrapper &b)
{
    throw NotImplementedError("FunctionWrapper saving is not implemented yet.");
}

template <class Archive>
inline void save_basic(Archive &ar, RCP<const Basic> const &ptr)
{
#if CEREAL_VERSION >= 10301
    std::shared_ptr<void> sharedPtr = std::static_pointer_cast<void>(
        std::make_shared<RCP<const Basic>>(ptr));
    uint32_t id = ar.registerSharedPointer(sharedPtr);
#else
    uint32_t id = ar.registerSharedPointer(ptr.get());
#endif
    ar(CEREAL_NVP(id));

    if (id & cereal::detail::msb_32bit) {
        ar(ptr->get_type_code());
        switch (ptr->get_type_code()) {
#define SYMENGINE_ENUM(type, Class)                                            \
    case type:                                                                 \
        save_basic(ar, static_cast<const Class &>(*ptr));                      \
        break;
#include "symengine/type_codes.inc"
#undef SYMENGINE_ENUM
            default:
                save_basic(ar, *ptr);
        }
    }
}

//! Saving for SymEngine::RCP
template <class Archive, class T>
inline void CEREAL_SAVE_FUNCTION_NAME(Archive &ar, RCP<const T> const &ptr)
{
    save_basic(ar, rcp_static_cast<const Basic>(ptr));
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const RealDouble> &)
{
    double val;
    ar(val);
    return real_double(val);
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const Infty> &)
{
    RCP<const Number> direction;
    ar(direction);
    return Infty::from_direction(direction);
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const NaN> &)
{
    return rcp_static_cast<const Basic>(Nan);
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const Symbol> &)
{
    std::string name;
    ar(name);
    return symbol(name);
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const Mul> &)
{
    RCP<const Number> coeff;
    map_basic_basic dict;
    ar(coeff);
    ar(dict);
    return make_rcp<const Mul>(coeff, std::move(dict));
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const Add> &)
{
    RCP<const Number> coeff;
    add_operands_map dict;
    ar(coeff);
    ar(dict);
    return make_rcp<const Add>(coeff, std::move(dict));
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const Pow> &)
{
    RCP<const Basic> base, exp;
    ar(base);
    ar(exp);
    return make_rcp<const Pow>(base, exp);
}
template <typename Archive>
void load_helper(Archive &ar, integer_class &intgr)
{
    std::string int_str;
    ar(int_str);
    intgr = integer_class(std::move(int_str));
}
template <typename Archive>
void load_helper(Archive &ar, const rational_class &rat)
{
    integer_class num, den;
    load_helper(ar, num);
    load_helper(ar, den);
}
// Following is an ugly hack for templated integer classes
// Not sure why the other clean version doesn't work
template <typename Archive>
RCP<const Basic> load_basic(Archive &ar, const URatPoly &b)
{
    RCP<const Basic> var;
    size_t l;
    ar(var);
    ar(l);
    std::map<unsigned, rational_class> d;
    auto hint = d.begin();
    for (size_t i = 0; i < l; i++) {
        unsigned int first;
        rational_class second;
        ar(first);
        load_helper(ar, second);
#if !defined(__clang__) && (__GNUC__ == 4 && __GNUC_MINOR__ <= 7)
        d.insert(hint, std::make_pair(std::move(first), std::move(second)));
#else
        d.emplace_hint(hint, std::move(first), std::move(second));
#endif
    }
    return make_rcp<const URatPoly>(var, URatDict(std::move(d)));
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const Integer> &)
{
    std::string int_str;
    ar(int_str);
    return integer(integer_class(int_str));
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const Constant> &)
{
    std::string name;
    ar(name);
    return constant(name);
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const Rational> &)
{
    RCP<const Integer> num, den;
    ar(num, den);
    return Rational::from_two_ints(*num, *den);
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const Complex> &)
{
    RCP<const Number> num, den;
    ar(num, den);
    return Complex::from_two_nums(*num, *den);
}
template <class Archive, class T>
RCP<const Basic>
load_basic(Archive &ar, RCP<const T> &,
           typename std::enable_if<std::is_base_of<ComplexBase, T>::value,
                                   int>::type * = nullptr)
{
    RCP<const Number> num, den;
    ar(num, den);
    return addnum(num, mulnum(I, den));
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const Interval> &)
{
    RCP<const Number> start, end;
    bool left_open, right_open;
    ar(left_open, start, right_open, end);
    return make_rcp<const Interval>(start, end, left_open, right_open);
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const BooleanAtom> &)
{
    bool val;
    ar(val);
    return boolean(val);
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const And> &)
{
    set_boolean container;
    ar(container);
    return make_rcp<const And>(std::move(container));
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const Or> &)
{
    set_boolean container;
    ar(container);
    return make_rcp<const Or>(std::move(container));
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const Xor> &)
{
    vec_boolean container;
    ar(container);
    return make_rcp<const Xor>(std::move(container));
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const Not> &)
{
    RCP<const Boolean> arg;
    ar(arg);
    return make_rcp<const Not>(arg);
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const Piecewise> &)
{
    PiecewiseVec vec;
    ar(vec);
    return make_rcp<const Piecewise>(std::move(vec));
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const Contains> &)
{
    RCP<const Basic> expr;
    RCP<const Set> contains_set;
    ar(expr, contains_set);
    return make_rcp<const Contains>(expr, contains_set);
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const Reals> &)
{
    return reals();
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const Rationals> &)
{
    return rationals();
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const EmptySet> &)
{
    return emptyset();
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const Integers> &)
{
    return integers();
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const UniversalSet> &)
{
    return universalset();
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const Union> &)
{
    set_set union_set;
    ar(union_set);
    return make_rcp<const Union>(std::move(union_set));
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const Complement> &)
{
    RCP<const Set> universe, container;
    ar(universe, container);
    return make_rcp<const Complement>(universe, container);
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const ImageSet> &)
{
    RCP<const Basic> sym, expr;
    RCP<const Set> base;
    ar(sym, expr, base);
    return make_rcp<const ImageSet>(sym, expr, base);
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const FiniteSet> &)
{
    set_basic set;
    ar(set);
    return make_rcp<const FiniteSet>(set);
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const ConditionSet> &)
{
    RCP<const Basic> sym;
    RCP<const Boolean> condition;
    ar(sym, condition);
    return make_rcp<const ConditionSet>(sym, condition);
}
#ifdef HAVE_SYMENGINE_MPFR
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const RealMPFR> &)
{
    std::string num;
    unsigned prec;
    ar(num, prec);
    return make_rcp<const RealMPFR>(mpfr_class(num, prec, 10));
}
#endif
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const Derivative> &)
{
    RCP<const Basic> arg;
    multiset_basic set;
    ar(arg, set);
    return make_rcp<const Derivative>(arg, std::move(set));
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const Subs> &)
{
    RCP<const Basic> arg;
    map_basic_basic dict;
    ar(arg, dict);
    return make_rcp<const Subs>(arg, std::move(dict));
}

template <class Archive, class T>
RCP<const Basic>
load_basic(Archive &ar, RCP<const T> &,
           typename std::enable_if<std::is_base_of<OneArgFunction, T>::value,
                                   int>::type * = nullptr)
{
    RCP<const Basic> arg;
    ar(arg);
    return make_rcp<const T>(arg);
}
template <class Archive, class T>
RCP<const Basic>
load_basic(Archive &ar, RCP<const T> &,
           typename std::enable_if<std::is_base_of<TwoArgFunction, T>::value,
                                   int>::type * = nullptr)
{
    RCP<const Basic> arg1, arg2;
    ar(arg1, arg2);
    return make_rcp<const T>(arg1, arg2);
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const FunctionSymbol> &)
{
    std::string name;
    vec_basic vec;
    ar(name, vec);
    return make_rcp<const FunctionSymbol>(name, std::move(vec));
}
template <class Archive>
RCP<const Basic> load_basic(Archive &ar, RCP<const FunctionWrapper> &)
{
    throw SerializationError(StreamFmt()
                             << __FILE__ << ":" << __LINE__
#ifndef _MSC_VER
                             << ": " << __PRETTY_FUNCTION__
#endif
                             << "Loading of this type is not implemented.");
}
template <class Archive, class T>
RCP<const Basic>
load_basic(Archive &ar, RCP<const T> &,
           typename std::enable_if<std::is_base_of<MultiArgFunction, T>::value,
                                   int>::type * = nullptr)
{
    vec_basic args;
    ar(args);
    return make_rcp<const T>(std::move(args));
}
template <class Archive, class T>
RCP<const Basic>
load_basic(Archive &ar, RCP<const T> &,
           typename std::enable_if<std::is_base_of<Relational, T>::value,
                                   int>::type * = nullptr)
{
    RCP<const Basic> arg1, arg2;
    ar(arg1, arg2);
    return make_rcp<const T>(arg1, arg2);
}
template <class Archive, class T>
RCP<const Basic> load_basic(
    Archive &ar, RCP<const T> &,
    typename std::enable_if<not(std::is_base_of<Relational, T>::value
                                or std::is_base_of<ComplexBase, T>::value
                                or std::is_base_of<OneArgFunction, T>::value
                                or std::is_base_of<MultiArgFunction, T>::value
                                or std::is_base_of<TwoArgFunction, T>::value),
                            int>::type * = nullptr)
{
    throw SerializationError(StreamFmt()
                             << __FILE__ << ":" << __LINE__
#ifndef _MSC_VER
                             << ": " << __PRETTY_FUNCTION__
#endif
                             << "Loading of this type is not implemented.");
}

//! Loading for SymEngine::RCP
template <class Archive, class T>
inline void CEREAL_LOAD_FUNCTION_NAME(Archive &ar, RCP<const T> &ptr)
{
    uint32_t id;
    ar(CEREAL_NVP(id));

    if (id & cereal::detail::msb_32bit) {
        TypeID type_code;
        ar(type_code);
        switch (type_code) {
#define SYMENGINE_ENUM(type_enum, Class)                                       \
    case type_enum: {                                                          \
        if (not std::is_base_of<T, Class>::value) {                            \
            throw std::runtime_error("Cannot convert to type.");               \
        } else {                                                               \
            RCP<const Class> dummy_ptr;                                        \
            ptr = rcp_static_cast<const T>(                                    \
                rcp_static_cast<const Basic>(load_basic(ar, dummy_ptr)));      \
            break;                                                             \
        }                                                                      \
    }
#include "symengine/type_codes.inc"
#undef SYMENGINE_ENUM
            default:
                throw std::runtime_error("Unknown type");
        }
        std::shared_ptr<void> sharedPtr = std::static_pointer_cast<void>(
            std::make_shared<RCP<const Basic>>(ptr));

        ar.registerSharedPointer(id, sharedPtr);
    } else {
        std::shared_ptr<RCP<const T>> sharedPtr
            = std::static_pointer_cast<RCP<const T>>(ar.getSharedPointer(id));
        ptr = *sharedPtr.get();
    }
}
} // namespace SymEngine
#endif // SYMENGINE_SERIALIZE_CEREAL_H
