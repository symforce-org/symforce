#include <symengine/expression.h>

namespace SymEngine
{

namespace
{
template <class T>
inline std::ostream &print_map(std::ostream &out, T &d)
{
    out << "{";
    for (auto p = d.begin(); p != d.end(); p++) {
        if (p != d.begin())
            out << ", ";
        out << (p->first) << ": " << (p->second);
    }
    out << "}";
    return out;
}

template <class T>
inline std::ostream &print_map_rcp(std::ostream &out, T &d)
{
    out << "{";
    for (auto p = d.begin(); p != d.end(); p++) {
        if (p != d.begin())
            out << ", ";
        out << *(p->first) << ": " << *(p->second);
    }
    out << "}";
    return out;
}

template <class T>
inline std::ostream &print_vec(std::ostream &out, T &d)
{
    out << "{";
    for (auto p = d.begin(); p != d.end(); p++) {
        if (p != d.begin())
            out << ", ";
        out << *p;
    }
    out << "}";
    return out;
}

template <class T>
inline std::ostream &print_vec_rcp(std::ostream &out, T &d)
{
    out << "{";
    for (auto p = d.begin(); p != d.end(); p++) {
        if (p != d.begin())
            out << ", ";
        out << **p;
    }
    out << "}";
    return out;
}

} // anonymous namespace

std::ostream &operator<<(std::ostream &out, const SymEngine::umap_basic_num &d)
{
    return SymEngine::print_map_rcp(out, d);
}

std::ostream &operator<<(std::ostream &out, const SymEngine::map_basic_num &d)
{
    return SymEngine::print_map_rcp(out, d);
}

std::ostream &operator<<(std::ostream &out, const SymEngine::map_basic_basic &d)
{
    return SymEngine::print_map_rcp(out, d);
}

std::ostream &operator<<(std::ostream &out,
                         const SymEngine::umap_basic_basic &d)
{
    return SymEngine::print_map_rcp(out, d);
}

std::ostream &operator<<(std::ostream &out, const SymEngine::vec_basic &d)
{
    return SymEngine::print_vec_rcp(out, d);
}

std::ostream &operator<<(std::ostream &out, const SymEngine::set_basic &d)
{
    return SymEngine::print_vec_rcp(out, d);
}

std::ostream &operator<<(std::ostream &out, const SymEngine::map_int_Expr &d)
{
    return SymEngine::print_map(out, d);
}

std::ostream &operator<<(std::ostream &out, const SymEngine::vec_pair &d)
{
    return SymEngine::print_map_rcp(out, d);
}

bool vec_basic_eq_perm(const vec_basic &a, const vec_basic &b)
{
    // Can't be equal if # of entries differ:
    if (a.size() != b.size())
        return false;
    // Loop over elements in "a"
    for (size_t i = 0; i < a.size(); i++) {
        // Find the element a[i] in "b"
        bool found = false;
        for (size_t j = 0; j < a.size(); j++) {
            if (eq(*a[i], *b[j])) {
                found = true;
                break;
            }
        }
        // If not found, then a != b
        if (not found)
            return false;
    }
    // If all elements were found, then a == b
    return true;
}
} // namespace SymEngine
