#ifndef SYMENGINE_BASIC_INL_H
#define SYMENGINE_BASIC_INL_H

namespace SymEngine
{

inline hash_t Basic::hash() const
{
    if (hash_ == 0)
        hash_ = __hash__();
    return hash_;
}

//! \return true if not equal
inline bool Basic::__neq__(const Basic &o) const
{
    return not(eq(*this, o));
}

//! \return true if  `a` equal `b`
inline bool eq(const Basic &a, const Basic &b)
{
    if (&a == &b) {
        return true;
    }
    return a.__eq__(b);
}
//! \return true if  `a` not equal `b`
inline bool neq(const Basic &a, const Basic &b)
{
    return not(a.__eq__(b));
}

//! Templatised version to check is_a type
template <class T>
inline bool is_a(const Basic &b)
{
    return T::type_code_id == b.get_type_code();
}

template <class T>
inline bool is_a_sub(const Basic &b)
{
    return dynamic_cast<const T *>(&b) != nullptr;
}

inline bool is_same_type(const Basic &a, const Basic &b)
{
    return a.get_type_code() == b.get_type_code();
}

//! `<<` Operator
inline std::ostream &operator<<(std::ostream &out, const SymEngine::Basic &p)
{
    out << p.__str__();
    return out;
}

//! Templatised version to combine hash
template <typename T>
inline void hash_combine_impl(
    hash_t &seed, const T &v,
    typename std::enable_if<std::is_base_of<Basic, T>::value>::type * = nullptr)
{
    hash_combine(seed, v.hash());
}

template <typename T>
inline void hash_combine_impl(
    hash_t &seed, const T &v,
    typename std::enable_if<std::is_integral<T>::value>::type * = nullptr)
{
    seed ^= hash_t(v) + hash_t(0x9e3779b9) + (seed << 6) + (seed >> 2);
}

inline void hash_combine_impl(hash_t &seed, const std::string &s)
{
    for (const char &c : s) {
        hash_combine<hash_t>(seed, static_cast<hash_t>(c));
    }
}

inline void hash_combine_impl(hash_t &seed, const double &s)
{
    union {
        hash_t h;
        double d;
    } u;
    u.h = 0u;
    u.d = s;
    hash_combine(seed, u.h);
}

template <class T>
inline void hash_combine(hash_t &seed, const T &v)
{
    hash_combine_impl(seed, v);
}

template <typename T>
hash_t vec_hash<T>::operator()(const T &v) const
{
    hash_t h = 0;
    for (auto i : v)
        hash_combine<typename T::value_type>(h, i);
    return h;
}

//! workaround for MinGW bug
template <typename T>
std::string to_string(const T &value)
{
#ifdef HAVE_SYMENGINE_STD_TO_STRING
    return std::to_string(value);
#else
    std::ostringstream ss;
    ss << value;
    return ss.str();
#endif
}

} // namespace SymEngine

// std namespace functions
namespace std
{
//! Specialise std::hash for Basic. We just call Basic.__hash__()
template <>
struct hash<SymEngine::Basic> {
    std::size_t operator()(const SymEngine::Basic &b) const
    {
        return static_cast<std::size_t>(b.hash());
    }
};
} // namespace std

#endif
