#ifndef SYMENGINE_TRIBOOL_H
#define SYMENGINE_TRIBOOL_H

namespace SymEngine
{

enum class tribool { indeterminate = -1, trifalse = 0, tritrue = 1 };

inline bool is_true(tribool x)
{
    return x == tribool::tritrue;
}

inline bool is_false(tribool x)
{
    return x == tribool::trifalse;
}

inline bool is_indeterminate(tribool x)
{
    return x == tribool::indeterminate;
}

inline tribool tribool_from_bool(bool x)
{
    return static_cast<tribool>(x);
}

inline tribool and_tribool(tribool a, tribool b)
{
    if (!(static_cast<unsigned>(a) & static_cast<unsigned>(b))) {
        return tribool::trifalse;
    } else {
        return static_cast<tribool>(static_cast<unsigned>(a)
                                    | static_cast<unsigned>(b));
    }
}

inline tribool or_tribool(tribool a, tribool b)
{
    if (is_true(a) || is_true(b)) {
        return tribool::tritrue;
    } else if (is_indeterminate(a) || is_indeterminate(b)) {
        return tribool::indeterminate;
    } else {
        return tribool::trifalse;
    }
}

inline tribool not_tribool(tribool a)
{
    if (is_indeterminate(a)) {
        return a;
    } else {
        return static_cast<tribool>(!static_cast<unsigned>(a));
    }
}

// The weak kleene conjunction
// Indeterminate if any indeterminate otherwise like regular and
inline tribool andwk_tribool(tribool a, tribool b)
{
    if (is_indeterminate(a) || is_indeterminate(b)) {
        return tribool::indeterminate;
    } else {
        return static_cast<tribool>(static_cast<unsigned>(a)
                                    && static_cast<unsigned>(b));
    }
}

// The weak kleene disjunction
// Indeterminate if any indeterminate otherwise like regular or
inline tribool orwk_tribool(tribool a, tribool b)
{
    if (is_indeterminate(a) || is_indeterminate(b)) {
        return tribool::indeterminate;
    } else {
        return static_cast<tribool>(static_cast<unsigned>(a)
                                    || static_cast<unsigned>(b));
    }
}

} // namespace SymEngine

#endif
