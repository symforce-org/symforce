#include <symengine/add.h>
#include <symengine/logic.h>
#include <symengine/functions.h>
#include <symengine/symengine_casts.h>
#include <iterator>

namespace SymEngine
{

Interval::Interval(const RCP<const Number> &start, const RCP<const Number> &end,
                   const bool left_open, const bool right_open)
    : start_(start), end_(end), left_open_(left_open), right_open_(right_open)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(
        Interval::is_canonical(start_, end_, left_open_, right_open_));
}

bool Interval::is_canonical(const RCP<const Number> &s,
                            const RCP<const Number> &e, bool left_open,
                            bool right_open)
{
    if (is_a<Complex>(*s) or is_a<Complex>(*e))
        throw NotImplementedError("Complex set not implemented");
    if (eq(*e, *s)) {
        return false;
    } else if (eq(*min({s, e}), *e)) {
        return false;
    }
    return true;
}

hash_t Interval::__hash__() const
{
    hash_t seed = SYMENGINE_INTERVAL;
    hash_combine<Basic>(seed, *start_);
    hash_combine<Basic>(seed, *end_);
    hash_combine<bool>(seed, left_open_);
    hash_combine<bool>(seed, right_open_);
    return seed;
}

bool Interval::__eq__(const Basic &o) const
{
    if (is_a<Interval>(o)) {
        const Interval &s = down_cast<const Interval &>(o);
        return ((this->left_open_ == s.left_open_)
                and (this->right_open_ == s.right_open_)
                and eq(*this->start_, *s.start_) and eq(*this->end_, *s.end_));
    }
    return false;
}

int Interval::compare(const Basic &s) const
{
    // compares two interval based on their length
    SYMENGINE_ASSERT(is_a<Interval>(s))
    const Interval &o = down_cast<const Interval &>(s);
    if (left_open_ and not o.left_open_) {
        return -1;
    } else if (not left_open_ and o.left_open_) {
        return 1;
    } else if (right_open_ and not o.right_open_) {
        return 1;
    } else if (not right_open_ and o.right_open_) {
        return -1;
    } else {
        auto temp = start_->__cmp__(*(o.start_));
        if (temp != 0) {
            return temp;
        } else {
            return end_->__cmp__(*(o.end_));
        }
    }
}

RCP<const Set> Interval::open() const
{
    return interval(start_, end_, true, true);
}

RCP<const Set> Interval::Lopen() const
{
    return interval(start_, end_, true, false);
}

RCP<const Set> Interval::Ropen() const
{
    return interval(start_, end_, false, true);
}

RCP<const Set> Interval::close() const
{
    return interval(start_, end_, false, false);
}

RCP<const Boolean> Interval::contains(const RCP<const Basic> &a) const
{
    if (not is_a_Number(*a)) {
        if (is_a_Set(*a)) {
            return boolean(false);
        } else {
            return make_rcp<Contains>(a, rcp_from_this_cast<const Set>());
        }
    }
    if (eq(*start_, *a))
        return boolean(not left_open_);
    if (eq(*end_, *a))
        return boolean(not right_open_);
    if (eq(*min({end_, a}), *end_) or eq(*max({start_, a}), *start_))
        return boolean(false);
    return boolean(true);
}

static RCP<const Set> make_set_union(const set_set &in)
{
    if (in.size() > 1) {
        return make_rcp<const Union>(in);
    }
    return *in.begin();
}

static RCP<const Set> make_set_intersection(const set_set &in)
{
    if (in.size() > 1) {
        return make_rcp<const Intersection>(in);
    }
    return *in.begin();
}

RCP<const Set> Interval::set_intersection(const RCP<const Set> &o) const
{
    if (is_a<Interval>(*o)) {
        const Interval &other = down_cast<const Interval &>(*o);
        RCP<const Number> start, end;
        bool left_open, right_open;
        RCP<const Basic> start_end, end_start;
        start_end = min({this->start_, other.end_});
        end_start = min({this->end_, other.start_});

        if (eq(*this->start_, *start_end) and eq(*other.start_, *end_start)) {
            RCP<const Basic> start_start, end_end;
            start_start = min({this->start_, other.start_});
            end_end = min({this->end_, other.end_});
            if (neq(*this->start_, *other.start_)) {
                if (eq(*this->start_, *start_start)) {
                    start = other.start_;
                    left_open = other.left_open_;
                } else {
                    start = this->start_;
                    left_open = this->left_open_;
                }
            } else {
                start = this->start_;
                left_open = this->left_open_ or other.left_open_;
            }

            if (neq(*this->end_, *other.end_)) {
                if (eq(*this->end_, *end_end)) {
                    end = this->end_;
                    right_open = this->right_open_;
                } else {
                    end = other.end_;
                    right_open = other.right_open_;
                }
            } else {
                end = this->end_;
                right_open = this->right_open_ or other.right_open_;
            }
            return interval(start, end, left_open, right_open);
        } else {
            return emptyset();
        }
    }
    if (is_a<Integers>(*o) or is_a<Naturals>(*o) or is_a<Naturals0>(*o)) {
        if (is_a_Number(*start_) and is_a_Number(*end_)) {
            auto first = SymEngine::ceiling(start_);
            auto last = SymEngine::floor(end_);
            if (is_a<Naturals>(*o)
                and not down_cast<const Integer &>(*first).is_positive()) {
                first = integer(1);
            } else if (is_a<Naturals0>(*o)
                       and down_cast<const Integer &>(*first).is_negative()) {
                first = integer(0);
            }
            if (eq(*first, *start_) and left_open_) {
                first = add(first, integer(1));
            }
            if (eq(*last, *end_) and right_open_) {
                last = add(last, integer(-1));
            }
            if (eq(*Lt(last, first), *boolTrue)) {
                return emptyset();
            }
            set_basic container;
            while (eq(*Ge(last, first), *boolTrue)) {
                container.insert(first);
                first = add(first, integer(1));
            }
            return finiteset(container);
        } else {
            return SymEngine::set_intersection(
                {rcp_from_this_cast<const Set>(), o});
        }
    }
    if (is_a<UniversalSet>(*o) or is_a<EmptySet>(*o) or is_a<FiniteSet>(*o)
        or is_a<Union>(*o) or is_a<Rationals>(*o) or is_a<Reals>(*o)
        or is_a<Complexes>(*o)) {
        return (*o).set_intersection(rcp_from_this_cast<const Set>());
    }
    return make_set_intersection({rcp_from_this_cast<const Set>(), o});
}

RCP<const Set> Interval::set_union(const RCP<const Set> &o) const
{
    if (is_a<Interval>(*o)) {
        const Interval &other = down_cast<const Interval &>(*o);
        RCP<const Basic> start_start, end_end, m;
        RCP<const Number> start, end;
        bool left_open, right_open;
        start_start = max({this->start_, other.start_});
        end_end = min({this->end_, other.end_});
        m = min({start_start, end_end});
        if ((eq(*end_end, *start_start) and eq(*end_end, *m)
             and ((eq(*end_end, *this->end_) and this->right_open_)
                  or (eq(*end_end, *other.end_) and other.right_open_)))
            or (eq(*end_end, *m) and not eq(*end_end, *start_start))) {
            return SymEngine::make_set_union(
                {rcp_from_this_cast<const Set>(), o});
        } else {
            if (eq(*min({this->start_, other.start_}), *this->start_)) {
                start = this->start_;
            } else {
                start = other.start_;
            }
            if (eq(*max({this->end_, other.end_}), *this->end_)) {
                end = this->end_;
            } else {
                end = other.end_;
            }
            left_open = ((neq(*this->start_, *start) or this->left_open_)
                         and (neq(*other.start_, *start) or other.left_open_));
            right_open = ((neq(*this->end_, *end) or this->right_open_)
                          and (neq(*other.end_, *end) or other.right_open_));
            return interval(start, end, left_open, right_open);
        }
    }
    if (is_a<UniversalSet>(*o) or is_a<EmptySet>(*o) or is_a<FiniteSet>(*o)
        or is_a<Union>(*o) or is_a<Complexes>(*o) or is_a<Reals>(*o)
        or is_a<Rationals>(*o) or is_a<Integers>(*o) or is_a<Naturals>(*o)
        or is_a<Naturals0>(*o)) {
        return (*o).set_union(rcp_from_this_cast<const Set>());
    }
    return SymEngine::make_set_union({rcp_from_this_cast<const Set>(), o});
}

RCP<const Set> Interval::set_complement(const RCP<const Set> &o) const
{
    if (is_a<Interval>(*o)) {
        set_set cont;
        const Interval &other = down_cast<const Interval &>(*o);
        if (eq(*max({start_, other.start_}), *start_)) {
            cont.insert(interval(other.get_start(), start_,
                                 other.get_left_open(), not left_open_));
        }
        if (eq(*min({end_, other.end_}), *end_)) {
            cont.insert(interval(end_, other.get_end(), not right_open_,
                                 other.get_right_open()));
        }
        return SymEngine::set_union(cont);
    }
    return SymEngine::set_complement_helper(rcp_from_this_cast<const Set>(), o);
}

vec_basic Interval::get_args() const
{
    return {start_, end_, boolean(left_open_), boolean(right_open_)};
}

RCP<const Set> Complexes::set_intersection(const RCP<const Set> &o) const
{
    if (is_a<Interval>(*o) or is_a<EmptySet>(*o) or is_a<Complexes>(*o)
        or is_a<Reals>(*o) or is_a<Rationals>(*o) or is_a<Integers>(*o)
        or is_a<Naturals>(*o) or is_a<Naturals0>(*o)) {
        return o;
    } else if (is_a<FiniteSet>(*o)) {
        return (*o).set_intersection(rcp_from_this_cast<const Set>());
    } else {
        return SymEngine::set_intersection(
            {rcp_from_this_cast<const Set>(), o});
    }
}

RCP<const Set> Complexes::set_union(const RCP<const Set> &o) const
{
    if (is_a<Interval>(*o) or is_a<EmptySet>(*o) or is_a<Complexes>(*o)
        or is_a<Reals>(*o) or is_a<Integers>(*o) or is_a<Rationals>(*o)
        or is_a<Naturals>(*o) or is_a<Naturals0>(*o)) {
        return complexes();
    } else if (is_a<FiniteSet>(*o)) {
        return (*o).set_union(rcp_from_this_cast<const Set>());
    } else {
        return SymEngine::set_union({rcp_from_this_cast<const Set>(), o});
    }
}

RCP<const Set> Complexes::set_complement(const RCP<const Set> &o) const
{
    if (is_a<EmptySet>(*o) or is_a<Complexes>(*o) or is_a<Reals>(*o)
        or is_a<Rationals>(*o) or is_a<Integers>(*o) or is_a<Interval>(*o)
        or is_a<Naturals>(*o) or is_a<Naturals0>(*o)) {
        return emptyset();
    }
    if (is_a<UniversalSet>(*o)) {
        return make_rcp<const Complement>(o, complexes());
    }
    return SymEngine::set_complement_helper(rcp_from_this_cast<const Set>(), o);
}

RCP<const Boolean> Complexes::contains(const RCP<const Basic> &a) const
{
    if (not is_a_Number(*a)) {
        if (is_a_Set(*a)) {
            return boolean(false);
        } else {
            return make_rcp<Contains>(a, rcp_from_this_cast<const Set>());
        }
    }
    return boolean(true);
}

hash_t Complexes::__hash__() const
{
    hash_t seed = SYMENGINE_COMPLEXES;
    return seed;
}

bool Complexes::__eq__(const Basic &o) const
{
    if (is_a<Complexes>(o))
        return true;
    return false;
}

int Complexes::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<Complexes>(o))
    return 0;
}

const RCP<const Complexes> &Complexes::getInstance()
{
    const static auto a = make_rcp<const Complexes>();
    return a;
}

RCP<const Set> Reals::set_intersection(const RCP<const Set> &o) const
{
    if (is_a<Interval>(*o) or is_a<EmptySet>(*o) or is_a<Reals>(*o)
        or is_a<Rationals>(*o) or is_a<Integers>(*o) or is_a<Naturals>(*o)
        or is_a<Naturals0>(*o)) {
        return o;
    } else if (is_a<FiniteSet>(*o) or is_a<Complexes>(*o)) {
        return (*o).set_intersection(rcp_from_this_cast<const Set>());
    } else {
        return SymEngine::set_intersection(
            {rcp_from_this_cast<const Set>(), o});
    }
}

RCP<const Set> Reals::set_union(const RCP<const Set> &o) const
{
    if (is_a<Interval>(*o) or is_a<EmptySet>(*o) or is_a<Reals>(*o)
        or is_a<Integers>(*o) or is_a<Rationals>(*o) or is_a<Naturals>(*o)
        or is_a<Naturals0>(*o)) {
        return reals();
    } else if (is_a<FiniteSet>(*o) or is_a<Complexes>(*o)) {
        return (*o).set_union(rcp_from_this_cast<const Set>());
    } else {
        return SymEngine::set_union({rcp_from_this_cast<const Set>(), o});
    }
}

RCP<const Set> Reals::set_complement(const RCP<const Set> &o) const
{
    if (is_a<EmptySet>(*o) or is_a<Reals>(*o) or is_a<Rationals>(*o)
        or is_a<Integers>(*o) or is_a<Interval>(*o) or is_a<Naturals>(*o)
        or is_a<Naturals0>(*o)) {
        return emptyset();
    }
    if (is_a<UniversalSet>(*o) or is_a<Complexes>(*o)) {
        return make_rcp<const Complement>(o, reals());
    }
    return SymEngine::set_complement_helper(rcp_from_this_cast<const Set>(), o);
}

RCP<const Boolean> Reals::contains(const RCP<const Basic> &a) const
{
    if (not is_a_Number(*a)) {
        if (is_a_Set(*a)) {
            return boolean(false);
        } else {
            return make_rcp<Contains>(a, rcp_from_this_cast<const Set>());
        }
    }
    if (is_a<Complex>(*a)) {
        return boolean(false);
    }
    return boolean(true);
}

hash_t Reals::__hash__() const
{
    hash_t seed = SYMENGINE_REALS;
    return seed;
}

bool Reals::__eq__(const Basic &o) const
{
    if (is_a<Reals>(o))
        return true;
    return false;
}

int Reals::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<Reals>(o))
    return 0;
}

const RCP<const Reals> &Reals::getInstance()
{
    const static auto a = make_rcp<const Reals>();
    return a;
}

RCP<const Set> Rationals::set_intersection(const RCP<const Set> &o) const
{
    if (is_a<EmptySet>(*o) or is_a<Rationals>(*o) or is_a<Integers>(*o)
        or is_a<Naturals>(*o) or is_a<Naturals0>(*o)) {
        return o;
    } else if (is_a<FiniteSet>(*o) or is_a<Reals>(*o) or is_a<Complexes>(*o)) {
        return (*o).set_intersection(rcp_from_this_cast<const Set>());
    } else {
        return SymEngine::set_intersection(
            {rcp_from_this_cast<const Set>(), o});
    }
}

RCP<const Set> Rationals::set_union(const RCP<const Set> &o) const
{
    if (is_a<EmptySet>(*o) or is_a<Integers>(*o) or is_a<Rationals>(*o)
        or is_a<Naturals>(*o) or is_a<Naturals0>(*o)) {
        return rationals();
    } else if (is_a<FiniteSet>(*o) or is_a<Reals>(*o) or is_a<Complexes>(*o)) {
        return (*o).set_union(rcp_from_this_cast<const Set>());
    } else {
        return SymEngine::set_union({rcp_from_this_cast<const Set>(), o});
    }
}

RCP<const Set> Rationals::set_complement(const RCP<const Set> &o) const
{
    if (is_a<EmptySet>(*o) or is_a<Rationals>(*o) or is_a<Integers>(*o)
        or is_a<Naturals>(*o) or is_a<Naturals0>(*o)) {
        return emptyset();
    }
    if (is_a<UniversalSet>(*o) or is_a<Complexes>(*o) or is_a<Reals>(*o)
        or is_a<Interval>(*o)) {
        return make_rcp<const Complement>(o, rationals());
    }
    return SymEngine::set_complement_helper(rcp_from_this_cast<const Set>(), o);
}

RCP<const Boolean> Rationals::contains(const RCP<const Basic> &a) const
{
    if (not is_a_Number(*a)) {
        if (is_a_Set(*a)) {
            return boolean(false);
        } else {
            return make_rcp<Contains>(a, rcp_from_this_cast<const Set>());
        }
    }
    if (is_a<Complex>(*a) or not down_cast<const Number &>(*a).is_exact()) {
        return boolean(false);
    }
    return boolean(true);
}

hash_t Rationals::__hash__() const
{
    hash_t seed = SYMENGINE_RATIONALS;
    return seed;
}

bool Rationals::__eq__(const Basic &o) const
{
    return (is_a<Rationals>(o));
}

int Rationals::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<Rationals>(o))
    return 0;
}

const RCP<const Rationals> &Rationals::getInstance()
{
    const static auto a = make_rcp<const Rationals>();
    return a;
}

RCP<const Set> Integers::set_intersection(const RCP<const Set> &o) const
{
    if (is_a<EmptySet>(*o) or is_a<Integers>(*o) or is_a<Naturals>(*o)
        or is_a<Naturals0>(*o)) {
        return o;
    } else if (is_a<Complexes>(*o) or is_a<Reals>(*o) or is_a<Rationals>(*o)) {
        return integers();
    } else if (is_a<FiniteSet>(*o) or is_a<Interval>(*o)) {
        return (*o).set_intersection(rcp_from_this_cast<const Set>());
    } else {
        return SymEngine::set_intersection(
            {rcp_from_this_cast<const Set>(), o});
    }
}

RCP<const Set> Integers::set_union(const RCP<const Set> &o) const
{
    if (is_a<Integers>(*o) or is_a<EmptySet>(*o) or is_a<Naturals>(*o)
        or is_a<Naturals0>(*o)) {
        return integers();
    } else if (is_a<Complexes>(*o)) {
        return complexes();
    } else if (is_a<Reals>(*o)) {
        return reals();
    } else if (is_a<Rationals>(*o)) {
        return rationals();
    } else if (is_a<FiniteSet>(*o)) {
        return (*o).set_union(rcp_from_this_cast<const Set>());
    } else if (is_a<UniversalSet>(*o)) {
        return universalset();
    } else {
        return SymEngine::make_set_union({rcp_from_this_cast<const Set>(), o});
    }
}

RCP<const Set> Integers::set_complement(const RCP<const Set> &o) const
{
    if (is_a<EmptySet>(*o) or is_a<Integers>(*o) or is_a<Naturals>(*o)
        or is_a<Naturals0>(*o)) {
        return emptyset();
    }
    if (is_a<UniversalSet>(*o) or is_a<Rationals>(*o) or is_a<Reals>(*o)
        or is_a<Complexes>(*o)) {
        return make_rcp<const Complement>(o, integers());
    }
    return SymEngine::set_complement_helper(rcp_from_this_cast<const Set>(), o);
}

RCP<const Boolean> Integers::contains(const RCP<const Basic> &a) const
{
    if (not is_a_Number(*a)) {
        if (is_a_Set(*a)) {
            return boolean(false);
        } else {
            return make_rcp<Contains>(a, rcp_from_this_cast<const Set>());
        }
    }
    if (is_a<Integer>(*a)) {
        return boolean(true);
    }
    return boolean(false);
}

hash_t Integers::__hash__() const
{
    hash_t seed = SYMENGINE_INTEGERS;
    return seed;
}

bool Integers::__eq__(const Basic &o) const
{
    if (is_a<Integers>(o))
        return true;
    return false;
}

int Integers::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<Integers>(o))
    return 0;
}

const RCP<const Integers> &Integers::getInstance()
{
    const static auto a = make_rcp<const Integers>();
    return a;
}

RCP<const Set> Naturals::set_intersection(const RCP<const Set> &o) const
{
    if (is_a<EmptySet>(*o) or is_a<Naturals>(*o)) {
        return o;
    } else if (is_a<Naturals0>(*o) or is_a<Integers>(*o) or is_a<Complexes>(*o)
               or is_a<Reals>(*o) or is_a<Rationals>(*o)) {
        return naturals();
    } else if (is_a<FiniteSet>(*o) or is_a<Interval>(*o)) {
        return (*o).set_intersection(rcp_from_this_cast<const Set>());
    } else {
        return SymEngine::set_intersection(
            {rcp_from_this_cast<const Set>(), o});
    }
}

RCP<const Set> Naturals::set_union(const RCP<const Set> &o) const
{
    if (is_a<EmptySet>(*o)) {
        return naturals();
    } else if (is_a<Naturals>(*o) or is_a<Naturals0>(*o) or is_a<Integers>(*o)
               or is_a<Complexes>(*o) or is_a<Reals>(*o) or is_a<Rationals>(*o)
               or is_a<UniversalSet>(*o)) {
        return o;
    } else if (is_a<FiniteSet>(*o)) {
        return (*o).set_union(rcp_from_this_cast<const Set>());
    } else {
        return SymEngine::make_set_union({rcp_from_this_cast<const Set>(), o});
    }
}

RCP<const Set> Naturals::set_complement(const RCP<const Set> &o) const
{
    if (is_a<EmptySet>(*o) or is_a<Naturals>(*o)) {
        return emptyset();
    }
    if (is_a<Naturals0>(*o)) {
        finiteset({zero});
    }
    if (is_a<UniversalSet>(*o) or is_a<Integers>(*o) or is_a<Rationals>(*o)
        or is_a<Reals>(*o) or is_a<Complexes>(*o)) {
        return make_rcp<const Complement>(o, naturals());
    }
    return SymEngine::set_complement_helper(rcp_from_this_cast<const Set>(), o);
}

RCP<const Boolean> Naturals::contains(const RCP<const Basic> &a) const
{
    if (not is_a_Number(*a)) {
        if (is_a_Set(*a)) {
            return boolean(false);
        } else {
            return make_rcp<Contains>(a, rcp_from_this_cast<const Set>());
        }
    } else if (is_a<Integer>(*a)
               and down_cast<const Integer &>(*a).is_positive()) {
        return boolean(true);
    } else {
        return boolean(false);
    }
}

hash_t Naturals::__hash__() const
{
    hash_t seed = SYMENGINE_NATURALS;
    return seed;
}

bool Naturals::__eq__(const Basic &o) const
{
    if (is_a<Naturals>(o))
        return true;
    return false;
}

int Naturals::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<Naturals>(o))
    return 0;
}

const RCP<const Naturals> &Naturals::getInstance()
{
    const static auto a = make_rcp<const Naturals>();
    return a;
}

RCP<const Set> Naturals0::set_intersection(const RCP<const Set> &o) const
{
    if (is_a<EmptySet>(*o) or is_a<Naturals>(*o) or is_a<Naturals0>(*o)) {
        return o;
    } else if (is_a<Integers>(*o) or is_a<Complexes>(*o) or is_a<Reals>(*o)
               or is_a<Rationals>(*o)) {
        return naturals0();
    } else if (is_a<FiniteSet>(*o) or is_a<Interval>(*o)) {
        return (*o).set_intersection(rcp_from_this_cast<const Set>());
    } else {
        return SymEngine::set_intersection(
            {rcp_from_this_cast<const Set>(), o});
    }
}

RCP<const Set> Naturals0::set_union(const RCP<const Set> &o) const
{
    if (is_a<EmptySet>(*o) or is_a<Naturals>(*o)) {
        return naturals0();
    } else if (is_a<Naturals0>(*o) or is_a<Integers>(*o) or is_a<Complexes>(*o)
               or is_a<Reals>(*o) or is_a<Rationals>(*o)
               or is_a<UniversalSet>(*o)) {
        return o;
    } else if (is_a<FiniteSet>(*o)) {
        return (*o).set_union(rcp_from_this_cast<const Set>());
    } else {
        return SymEngine::make_set_union({rcp_from_this_cast<const Set>(), o});
    }
}

RCP<const Set> Naturals0::set_complement(const RCP<const Set> &o) const
{
    if (is_a<EmptySet>(*o) or is_a<Naturals0>(*o) or is_a<Naturals>(*o)) {
        return emptyset();
    }
    if (is_a<UniversalSet>(*o) or is_a<Integers>(*o) or is_a<Rationals>(*o)
        or is_a<Reals>(*o) or is_a<Complexes>(*o)) {
        return make_rcp<const Complement>(o, naturals());
    }
    return SymEngine::set_complement_helper(rcp_from_this_cast<const Set>(), o);
}

RCP<const Boolean> Naturals0::contains(const RCP<const Basic> &a) const
{
    if (not is_a_Number(*a)) {
        if (is_a_Set(*a)) {
            return boolean(false);
        } else {
            return make_rcp<Contains>(a, rcp_from_this_cast<const Set>());
        }
    } else if (is_a<Integer>(*a)
               and (not down_cast<const Integer &>(*a).is_negative())) {
        return boolean(true);
    } else {
        return boolean(false);
    }
}

hash_t Naturals0::__hash__() const
{
    hash_t seed = SYMENGINE_NATURALS0;
    return seed;
}

bool Naturals0::__eq__(const Basic &o) const
{
    if (is_a<Naturals0>(o))
        return true;
    return false;
}

int Naturals0::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<Naturals0>(o))
    return 0;
}

const RCP<const Naturals0> &Naturals0::getInstance()
{
    const static auto a = make_rcp<const Naturals0>();
    return a;
}

RCP<const Set> EmptySet::set_intersection(const RCP<const Set> &o) const
{
    return emptyset();
}

RCP<const Set> EmptySet::set_union(const RCP<const Set> &o) const
{
    return o;
}

RCP<const Set> EmptySet::set_complement(const RCP<const Set> &o) const
{
    return o;
}

hash_t EmptySet::__hash__() const
{
    hash_t seed = SYMENGINE_EMPTYSET;
    return seed;
}

bool EmptySet::__eq__(const Basic &o) const
{
    if (is_a<EmptySet>(o))
        return true;
    return false;
}

int EmptySet::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<EmptySet>(o))
    return 0;
}

const RCP<const EmptySet> &EmptySet::getInstance()
{
    const static auto a = make_rcp<const EmptySet>();
    return a;
}

RCP<const Set> UniversalSet::set_intersection(const RCP<const Set> &o) const
{
    return o;
}

RCP<const Set> UniversalSet::set_union(const RCP<const Set> &o) const
{
    return universalset();
}

RCP<const Set> UniversalSet::set_complement(const RCP<const Set> &o) const
{
    return emptyset();
}

hash_t UniversalSet::__hash__() const
{
    hash_t seed = SYMENGINE_UNIVERSALSET;
    return seed;
}

bool UniversalSet::__eq__(const Basic &o) const
{
    if (is_a<UniversalSet>(o))
        return true;
    return false;
}

int UniversalSet::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<UniversalSet>(o))
    return 0;
}

const RCP<const UniversalSet> &UniversalSet::getInstance()
{
    const static auto a = make_rcp<const UniversalSet>();
    return a;
}

FiniteSet::FiniteSet(const set_basic &container) : container_(container)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(FiniteSet::is_canonical(container_));
}

bool FiniteSet::is_canonical(const set_basic &container)
{
    return container.size() != 0;
}

hash_t FiniteSet::__hash__() const
{
    hash_t seed = SYMENGINE_FINITESET;
    for (const auto &a : container_)
        hash_combine<Basic>(seed, *a);
    return seed;
}

bool FiniteSet::__eq__(const Basic &o) const
{
    if (is_a<FiniteSet>(o)) {
        const FiniteSet &other = down_cast<const FiniteSet &>(o);
        return unified_eq(container_, other.container_);
    }
    return false;
}

int FiniteSet::compare(const Basic &o) const
{
    // compares two FiniteSet based on their length
    SYMENGINE_ASSERT(is_a<FiniteSet>(o))
    const FiniteSet &other = down_cast<const FiniteSet &>(o);
    return unified_compare(container_, other.container_);
}

RCP<const Boolean> FiniteSet::contains(const RCP<const Basic> &a) const
{
    set_basic rest;
    for (const auto &elem : container_) {
        auto cont = Eq(elem, a);
        if (eq(*cont, *boolTrue))
            return boolTrue;
        if (not eq(*cont, *boolFalse))
            rest.insert(elem);
    }
    if (rest.empty()) {
        return boolFalse;
    } else {
        return make_rcp<Contains>(a, finiteset(rest));
    }
}

RCP<const Set> FiniteSet::set_union(const RCP<const Set> &o) const
{
    if (is_a<FiniteSet>(*o)) {
        const FiniteSet &other = down_cast<const FiniteSet &>(*o);
        set_basic container;
        std::set_union(container_.begin(), container_.end(),
                       other.container_.begin(), other.container_.end(),
                       std::inserter(container, container.begin()),
                       RCPBasicKeyLess{});
        return finiteset(container);
    }
    if (is_a<Interval>(*o)) {
        set_basic container;
        const Interval &other = down_cast<const Interval &>(*o);
        bool left = other.get_left_open(), right = other.get_right_open();
        for (const auto &a : container_) {
            auto contain = o->contains(a);
            if (eq(*contain, *boolFalse)) {
                if (left)
                    if (eq(*other.get_start(), *a)) {
                        left = false;
                        continue;
                    }
                if (right)
                    if (eq(*other.get_end(), *a)) {
                        right = false;
                        continue;
                    }
                container.insert(a);
            } else if (is_a<Contains>(*contain)) {
                container.insert(a);
            }
        }
        if (not container.empty()) {
            if (left == other.get_left_open()
                and right == other.get_right_open()) {
                return SymEngine::make_set_union({finiteset(container), o});
            } else {
                return SymEngine::make_set_union(
                    set_set({finiteset(container),
                             interval(other.get_start(), other.get_end(), left,
                                      right)}));
            }
        } else {
            if (left == other.get_left_open()
                and right == other.get_right_open()) {
                return o;
            } else {
                return interval(other.get_start(), other.get_end(), left,
                                right);
            }
        }
    }
    if (is_a<Complexes>(*o)) {
        set_basic container;
        for (const auto &elem : container_) {
            if (!is_a_Number(*elem)) {
                container.insert(elem);
            }
        }
        if (container.empty()) {
            return complexes();
        } else {
            return SymEngine::make_set_union(
                {complexes(), finiteset(container)});
        }
    }
    if (is_a<Reals>(*o)) {
        set_basic container;
        for (const auto &elem : container_) {
            if (!is_a_Number(*elem)
                || down_cast<const Number &>(*elem).is_complex()) {
                container.insert(elem);
            }
        }
        if (container.empty()) {
            return reals();
        } else {
            return SymEngine::make_set_union({reals(), finiteset(container)});
        }
    }
    if (is_a<Rationals>(*o)) {
        set_basic container;
        for (const auto &elem : container_) {
            if (!is_a_Number(*elem)
                || down_cast<const Number &>(*elem).is_complex()) {
                container.insert(elem);
            }
        }
        if (container.empty()) {
            return rationals();
        } else {
            return SymEngine::make_set_union(
                {rationals(), finiteset(container)});
        }
    }
    if (is_a<Integers>(*o) or is_a<Naturals>(*o) or is_a<Naturals0>(*o)) {
        set_basic container;
        for (const auto &elem : container_) {
            if (is_a<Integers>(*o)) {
                if (not is_a<Integer>(*elem)) {
                    container.insert(elem);
                }
            } else if (is_a<Naturals>(*o)) {
                if (not(is_a<Integer>(*elem)
                        and down_cast<const Integer &>(*elem).is_positive())) {
                    container.insert(elem);
                }
            } else {
                if (not(is_a<Integer>(*elem)
                        and not down_cast<const Integer &>(*elem)
                                    .is_negative())) {
                    container.insert(elem);
                }
            }
        }
        if (container.empty()) {
            return o;
        } else {
            return SymEngine::make_set_union({o, finiteset(container)});
        }
    }
    if (is_a<UniversalSet>(*o) or is_a<EmptySet>(*o) or is_a<Union>(*o)) {
        return (*o).set_union(rcp_from_this_cast<const Set>());
    }
    return SymEngine::make_set_union({rcp_from_this_cast<const Set>(), o});
}

RCP<const Set> FiniteSet::set_intersection(const RCP<const Set> &o) const
{
    if (is_a<FiniteSet>(*o)) {
        return SymEngine::set_intersection(
            {rcp_from_this_cast<const Set>(), o});
    }
    if (is_a<Interval>(*o)) {
        set_basic container;
        for (const auto &a : container_) {
            auto contain = o->contains(a);
            if (eq(*contain, *boolTrue))
                container.insert(a);
            if (is_a<Contains>(*contain))
                return make_set_intersection(
                    {rcp_from_this_cast<const Set>(), o});
        }
        return finiteset(container);
    }
    if (is_a<Complexes>(*o) or is_a<Reals>(*o) or is_a<Rationals>(*o)) {
        set_basic kept;
        set_basic others;
        for (const auto &elem : container_) {
            if (is_a_Number(*elem)) {
                if (!down_cast<const Number &>(*elem).is_complex()) {
                    if (!is_a<Rationals>(*o)
                        or down_cast<const Number &>(*elem).is_exact()) {
                        kept.insert(elem);
                    }
                } else {
                    if (is_a<Complexes>(*o)) {
                        kept.insert(elem);
                    }
                }
            } else {
                others.insert(elem);
            }
        }
        if (kept.empty()) {
            if (others.empty()) {
                return emptyset();
            } else {
                return SymEngine::make_set_intersection({o, finiteset(others)});
            }
        } else {
            if (others.empty()) {
                return finiteset(kept);
            } else {
                others.insert(kept.begin(), kept.end());
                return SymEngine::make_set_intersection({o, finiteset(others)});
            }
        }
    }
    if (is_a<Integers>(*o) or is_a<Naturals>(*o) or is_a<Naturals0>(*o)) {
        set_basic kept_integers;
        set_basic others;
        for (const auto &elem : container_) {
            if (is_a_Number(*elem)) {
                if (is_a<Integers>(*o) and is_a<Integer>(*elem)) {
                    kept_integers.insert(elem);
                } else if (is_a<Naturals>(*o) and is_a<Integer>(*elem)
                           and down_cast<const Integer &>(*elem)
                                   .is_positive()) {
                    kept_integers.insert(elem);
                } else if (is_a<Naturals0>(*o) and is_a<Integer>(*elem)
                           and (not down_cast<const Integer &>(*elem)
                                        .is_negative())) {
                    kept_integers.insert(elem);
                }
            } else {
                others.insert(elem);
            }
        }
        if (kept_integers.empty()) {
            if (others.empty()) {
                return emptyset();
            } else {
                return SymEngine::make_set_intersection(
                    {integers(), finiteset(others)});
            }
        } else {
            if (others.empty()) {
                return finiteset(kept_integers);
            } else {
                others.insert(kept_integers.begin(), kept_integers.end());
                return SymEngine::make_set_intersection(
                    {integers(), finiteset(others)});
            }
        }
    }
    if (is_a<UniversalSet>(*o) or is_a<EmptySet>(*o) or is_a<Union>(*o)) {
        return (*o).set_intersection(rcp_from_this_cast<const Set>());
    }
    return make_set_intersection({rcp_from_this_cast<const Set>(), o});
}

RCP<const Set> FiniteSet::set_complement(const RCP<const Set> &o) const
{
    if (is_a<FiniteSet>(*o)) {
        const FiniteSet &other = down_cast<const FiniteSet &>(*o);
        set_basic container;
        std::set_difference(other.container_.begin(), other.container_.end(),
                            container_.begin(), container_.end(),
                            std::inserter(container, container.begin()),
                            RCPBasicKeyLess{});
        return finiteset(container);
    }

    if (is_a<Interval>(*o)) {
        set_set intervals;
        auto &other = down_cast<const Interval &>(*o);
        RCP<const Number> last = other.get_start();
        RCP<const Number> a_num;
        set_basic rest;
        bool left_open = other.get_left_open(),
             right_open = other.get_right_open();
        for (auto it = container_.begin(); it != container_.end(); it++) {
            if (eq(*max({*it, other.get_start()}), *other.get_start())) {
                if (eq(**it, *other.get_start()))
                    left_open = true;
                continue;
            }
            if (eq(*max({*it, other.get_end()}), **it)) {
                if (eq(**it, *other.get_end()))
                    right_open = true;
                break;
            }
            if (is_a_Number(**it)) {
                a_num = rcp_static_cast<const Number>(*it);
                intervals.insert(interval(last, a_num, left_open, true));
                last = a_num;
                left_open = true;
            } else {
                rest.insert(*it);
            }
        }

        if (eq(*max({last, other.get_end()}), *other.get_end())) {
            intervals.insert(
                interval(last, other.get_end(), left_open, right_open));
        }
        if (rest.empty()) {
            return SymEngine::make_set_union(intervals);
        } else {
            return make_rcp<const Complement>(
                SymEngine::make_set_union(intervals), finiteset(rest));
        }
    }

    return SymEngine::set_complement_helper(rcp_from_this_cast<const Set>(), o);
}

RCP<const Set> FiniteSet::create(const set_basic &container) const
{
    return finiteset(container);
}

Union::Union(const set_set &in)
    : container_(in){SYMENGINE_ASSIGN_TYPEID()
                         SYMENGINE_ASSERT(Union::is_canonical(in))}

      hash_t Union::__hash__() const
{
    hash_t seed = SYMENGINE_UNION;
    for (const auto &a : container_)
        hash_combine<Basic>(seed, *a);
    return seed;
}

bool Union::__eq__(const Basic &o) const
{
    if (is_a<Union>(o)) {
        const Union &other = down_cast<const Union &>(o);
        return unified_eq(container_, other.container_);
    }
    return false;
}

bool Union::is_canonical(const set_set &in)
{
    if (in.size() <= 1)
        return false;
    int count = 0;
    for (const auto &s : in) {
        if (is_a<FiniteSet>(*s)) {
            count++;
        }
        if (count >= 2)
            return false;
    }
    return true;
}

int Union::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<Union>(o))
    const Union &other = down_cast<const Union &>(o);
    return unified_compare(container_, other.container_);
}

RCP<const Set> Union::set_union(const RCP<const Set> &o) const
{
    set_set container(container_);
    for (auto iter = container.begin(); iter != container.end(); ++iter) {
        auto temp = o->set_union(*iter);
        // If we are able to do union with `*iter`, we replace `*iter` with
        // the result of union.
        auto un = SymEngine::make_set_union({o, *iter});
        if (not eq(*temp, *un)) {
            iter = container.erase(iter);
            container.insert(temp);
            return SymEngine::set_union(container);
        }
    }
    container.insert(o);
    return SymEngine::make_set_union(container);
}

RCP<const Set> Union::set_intersection(const RCP<const Set> &o) const
{
    set_set container;
    for (auto &a : container_) {
        container.insert(a->set_intersection(o));
    }
    return SymEngine::set_union(container);
}

RCP<const Set> Union::set_complement(const RCP<const Set> &o) const
{
    set_set container;
    for (auto &a : container_) {
        container.insert(a->set_complement(o));
    }
    return SymEngine::set_intersection(container);
}

RCP<const Boolean> Union::contains(const RCP<const Basic> &o) const
{
    for (auto &a : container_) {
        auto contain = a->contains(o);
        if (eq(*contain, *boolTrue)) {
            return boolean(true);
        }
        if (is_a<Contains>(*contain))
            throw NotImplementedError("Not implemented");
    }
    return boolean(false);
}

RCP<const Set> Union::create(const set_set &in) const
{
    return SymEngine::set_union(in);
}

vec_basic Union::get_args() const
{
    vec_basic v(container_.begin(), container_.end());
    return v;
}

Intersection::Intersection(const set_set &in)
    : container_(in){SYMENGINE_ASSIGN_TYPEID()
                         SYMENGINE_ASSERT(Intersection::is_canonical(in))}

      hash_t Intersection::__hash__() const
{
    hash_t seed = SYMENGINE_INTERSECTION;
    for (const auto &a : container_)
        hash_combine<Basic>(seed, *a);
    return seed;
}

bool Intersection::__eq__(const Basic &o) const
{
    if (is_a<Intersection>(o)) {
        const Intersection &other = down_cast<const Intersection &>(o);
        return unified_eq(container_, other.container_);
    }
    return false;
}

bool Intersection::is_canonical(const set_set &in)
{
    if (in.size() <= 1)
        return false;
    return true;
}

int Intersection::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<Intersection>(o))
    const Intersection &other = down_cast<const Intersection &>(o);
    return unified_compare(container_, other.container_);
}

RCP<const Set> Intersection::set_union(const RCP<const Set> &o) const
{
    set_set container;
    for (auto &a : container_) {
        container.insert(a->set_union(o));
    }
    return SymEngine::set_intersection(container);
}

RCP<const Set> Intersection::set_intersection(const RCP<const Set> &o) const
{
    set_set container(container_);
    for (auto iter = container.begin(); iter != container.end(); ++iter) {
        auto temp = o->set_intersection(*iter);
        // If we are able to do intersection with `*iter`, we replace `*iter`
        // with the result of intersection.
        auto un = SymEngine::make_set_intersection({o, *iter});
        if (not eq(*temp, *un)) {
            iter = container.erase(iter);
            container.insert(temp);
            return SymEngine::set_intersection(container);
        }
    }
    container.insert(o);
    return SymEngine::make_set_intersection(container);
}

RCP<const Set> Intersection::set_complement(const RCP<const Set> &o) const
{
    set_set container;
    for (auto &a : container_) {
        container.insert(a->set_complement(o));
    }
    return SymEngine::set_intersection(container);
}

RCP<const Boolean> Intersection::contains(const RCP<const Basic> &o) const
{
    for (auto &a : container_) {
        auto contain = a->contains(o);
        if (eq(*contain, *boolTrue)) {
            return boolean(true);
        }
        if (is_a<Contains>(*contain))
            throw NotImplementedError("Not implemented");
    }
    return boolean(false);
}

RCP<const Set> Intersection::create(const set_set &in) const
{
    return SymEngine::set_intersection(in);
}

vec_basic Intersection::get_args() const
{
    vec_basic v(container_.begin(), container_.end());
    return v;
}

Complement::Complement(const RCP<const Set> &universe,
                       const RCP<const Set> &container)
    : universe_(universe), container_(container){SYMENGINE_ASSIGN_TYPEID()}

      hash_t Complement::__hash__() const
{
    hash_t seed = SYMENGINE_COMPLEMENT;
    hash_combine<Basic>(seed, *universe_);
    hash_combine<Basic>(seed, *container_);
    return seed;
}

bool Complement::__eq__(const Basic &o) const
{
    if (is_a<Complement>(o)) {
        const Complement &other = down_cast<const Complement &>(o);
        return unified_eq(universe_, other.universe_)
               and unified_eq(container_, other.container_);
    }
    return false;
}

int Complement::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<Complement>(o))
    const Complement &other = down_cast<const Complement &>(o);
    int c1 = unified_compare(universe_, other.universe_);
    if (c1 != 0) {
        return c1;
    } else {
        return unified_compare(container_, other.container_);
    }
}

RCP<const Boolean> Complement::contains(const RCP<const Basic> &a) const
{
    return logical_and(
        {universe_->contains(a), logical_not(container_->contains(a))});
}

RCP<const Set> Complement::set_union(const RCP<const Set> &o) const
{
    // A' U C = (A n C')'
    RCP<const Set> ocomplement = o->set_complement(universe_);
    RCP<const Set> intersect
        = SymEngine::set_intersection({container_, ocomplement});
    return intersect->set_complement(universe_);
}

RCP<const Set> Complement::set_intersection(const RCP<const Set> &o) const
{
    return SymEngine::set_intersection({rcp_from_this_cast<const Set>(), o});
}

RCP<const Set> Complement::set_complement(const RCP<const Set> &o) const
{
    auto newuniv = SymEngine::set_union({o, universe_});
    return container_->set_complement(newuniv);
}

ConditionSet::ConditionSet(const RCP<const Basic> &sym,
                           const RCP<const Boolean> &condition)
    : sym(sym), condition_(condition)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(ConditionSet::is_canonical(sym, condition))
}

bool ConditionSet::is_canonical(const RCP<const Basic> &sym,
                                const RCP<const Boolean> &condition)
{
    if (eq(*condition, *boolFalse) or eq(*condition, *boolTrue)
        or not is_a_sub<Symbol>(*sym)) {
        return false;
    } else if (is_a<Contains>(*condition)) {
        return false;
    }
    return true;
}

hash_t ConditionSet::__hash__() const
{
    hash_t seed = SYMENGINE_CONDITIONSET;
    hash_combine<Basic>(seed, *sym);
    hash_combine<Basic>(seed, *condition_);
    return seed;
}

bool ConditionSet::__eq__(const Basic &o) const
{
    if (is_a<ConditionSet>(o)) {
        const ConditionSet &other = down_cast<const ConditionSet &>(o);
        return unified_eq(sym, other.get_symbol())
               and unified_eq(condition_, other.get_condition());
    }
    return false;
}

int ConditionSet::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<ConditionSet>(o))
    const ConditionSet &other = down_cast<const ConditionSet &>(o);
    int c1 = unified_compare(sym, other.get_symbol());
    if (c1 != 0) {
        return c1;
    } else {
        return unified_compare(condition_, other.get_condition());
    }
}

RCP<const Set> ConditionSet::set_union(const RCP<const Set> &o) const
{
    return SymEngine::make_set_union({o, rcp_from_this_cast<const Set>()});
}

RCP<const Set> ConditionSet::set_intersection(const RCP<const Set> &o) const
{
    if (not is_a<ConditionSet>(*o)) {
        return conditionset(sym, logical_and({condition_, o->contains(sym)}));
    }
    return make_set_intersection({rcp_from_this_cast<const Set>(), o});
}

RCP<const Set> ConditionSet::set_complement(const RCP<const Set> &o) const
{
    return make_rcp<const Complement>(o, rcp_from_this_cast<const Set>());
}

RCP<const Boolean> ConditionSet::contains(const RCP<const Basic> &o) const
{
    map_basic_basic d;
    d[sym] = o;
    auto cond = condition_->subs(d);
    if (not is_a_Boolean(*cond)) {
        throw SymEngineException("expected an object of type Boolean");
    }
    return rcp_static_cast<const Boolean>(cond);
}

ImageSet::ImageSet(const RCP<const Basic> &sym, const RCP<const Basic> &expr,
                   const RCP<const Set> &base)
    : sym_(sym), expr_(expr), base_(base)
{
    SYMENGINE_ASSIGN_TYPEID()
    SYMENGINE_ASSERT(ImageSet::is_canonical(sym, expr, base));
}

hash_t ImageSet::__hash__() const
{
    hash_t seed = SYMENGINE_IMAGESET;
    hash_combine<Basic>(seed, *sym_);
    hash_combine<Basic>(seed, *expr_);
    hash_combine<Basic>(seed, *base_);
    return seed;
}

bool ImageSet::__eq__(const Basic &o) const
{
    if (is_a<ImageSet>(o)) {
        const ImageSet &other = down_cast<const ImageSet &>(o);
        return unified_eq(sym_, other.sym_) and unified_eq(expr_, other.expr_)
               and unified_eq(base_, other.base_);
    }
    return false;
}

int ImageSet::compare(const Basic &o) const
{
    SYMENGINE_ASSERT(is_a<ImageSet>(o))
    const ImageSet &other = down_cast<const ImageSet &>(o);
    int c1 = unified_compare(sym_, other.sym_);
    if (c1 != 0) {
        return c1;
    } else {
        int c2 = unified_compare(expr_, other.expr_);
        if (c2 != 0) {
            return c2;
        } else {
            return unified_compare(base_, other.base_);
        }
    }
}

bool ImageSet::is_canonical(const RCP<const Basic> &sym,
                            const RCP<const Basic> &expr,
                            const RCP<const Set> &base)
{
    if (not is_a_sub<Symbol>(*sym) or eq(*expr, *sym) or is_a_Number(*expr)
        or eq(*base, *emptyset()))
        return false;
    return true;
}

RCP<const Boolean> ImageSet::contains(const RCP<const Basic> &a) const
{
    throw SymEngineException("Not implemented");
}

RCP<const Set> ImageSet::set_union(const RCP<const Set> &o) const
{
    return make_set_union({rcp_from_this_cast<const Set>(), o});
}

RCP<const Set> ImageSet::set_intersection(const RCP<const Set> &o) const
{
    return SymEngine::set_intersection({rcp_from_this_cast<const Set>(), o});
}

RCP<const Set> ImageSet::set_complement(const RCP<const Set> &o) const
{
    return SymEngine::set_complement(rcp_from_this_cast<const Set>(), o);
}

RCP<const Set> ImageSet::create(const RCP<const Basic> &sym,
                                const RCP<const Basic> &expr,
                                const RCP<const Set> &base) const
{
    return imageset(sym, expr, base);
}

RCP<const Set> set_union(const set_set &in)
{
    set_set input;
    set_basic combined_FiniteSet;
    for (auto it = in.begin(); it != in.end(); ++it) {
        if (is_a<FiniteSet>(**it)) {
            const FiniteSet &other = down_cast<const FiniteSet &>(**it);
            combined_FiniteSet.insert(other.get_container().begin(),
                                      other.get_container().end());
        } else if (is_a<UniversalSet>(**it)) {
            return universalset();
        } else if (not is_a<EmptySet>(**it)) {
            input.insert(*it);
        }
    }
    if (input.empty()) {
        return finiteset(combined_FiniteSet);
    } else if (input.size() == 1 && combined_FiniteSet.empty()) {
        return *input.begin();
    }
    // Now we rely on respective containers' own rules
    // TODO: Improve it to O(log n)
    RCP<const Set> combined_Rest = finiteset(combined_FiniteSet);
    for (auto it = input.begin(); it != input.end(); ++it) {
        combined_Rest = combined_Rest->set_union(*it);
    }
    return combined_Rest;
}

RCP<const Set> set_intersection(const set_set &in)
{
    // https://en.wikipedia.org/wiki/Intersection_(set_theory)#Nullary_intersection
    if (in.empty())
        return universalset();

    // Global rules
    // If found any emptyset then return emptyset
    set_set incopy;
    for (const auto &input : in) {
        if (is_a<EmptySet>(*input)) {
            return emptyset();
        } else if (not is_a<UniversalSet>(*input)) {
            incopy.insert(input);
        }
    }

    if (incopy.empty())
        return universalset();
    if (incopy.size() == 1)
        return *incopy.begin();

    // Handle finite sets
    std::vector<RCP<const Set>> fsets, othersets;
    for (const auto &input : incopy) {
        if (is_a<FiniteSet>(*input)) {
            fsets.push_back(input);
        } else {
            othersets.push_back(input);
        }
    }
    if (fsets.size() != 0) {
        const FiniteSet &fs = down_cast<const FiniteSet &>(**fsets.begin());
        auto cont = fs.get_container();
        fsets.erase(fsets.begin());
        set_basic finalfs;
        for (const auto &fselement : cont) {
            bool present = true;
            for (const auto &fset : fsets) {
                auto contain = fset->contains(fselement);
                if (not(eq(*contain, *boolTrue) or eq(*contain, *boolFalse))) {
                    return make_set_intersection(incopy);
                }
                present = present and eq(*contain, *boolTrue);
            }
            if (!present)
                continue;
            for (const auto &oset : othersets) {
                auto contain = oset->contains(fselement);
                if (not(eq(*contain, *boolTrue) or eq(*contain, *boolFalse))) {
                    return make_set_intersection(incopy);
                }
                present = present and eq(*contain, *boolTrue);
            }
            if (present)
                finalfs.insert(fselement);
        }
        return finiteset(finalfs);
    }

    // If any of the sets is union, then return a Union of Intersections
    for (auto it = incopy.begin(); it != incopy.end(); ++it) {
        if (is_a<Union>(**it)) {
            auto container = down_cast<const Union &>(**it).get_container();
            incopy.erase(it);
            auto other = SymEngine::set_intersection(incopy);
            set_set usets;
            for (const auto &c : container) {
                usets.insert(SymEngine::set_intersection({c, other}));
            }
            return SymEngine::set_union(usets);
        }
    }

    // Simplify and return a `Complement` if any of the sets is a complement
    for (auto it = incopy.begin(); it != incopy.end(); ++it) {
        if (is_a<Complement>(**it)) {
            auto container
                = down_cast<const Complement &>(**it).get_container();
            auto universe = down_cast<const Complement &>(**it).get_universe();
            incopy.erase(it);
            incopy.insert(universe);
            auto other = SymEngine::set_intersection(incopy);
            return SymEngine::set_complement(other, container);
        }
    }

    // Pair-wise rules
    // TO-DO: needs the following improvement once Intersection
    // class is implemented.
    // input_oset if found to be not simplified, then skip this
    // pair.
    if (incopy.size() > 1) {
        auto temp = *incopy.begin();
        auto it = std::next(incopy.begin());
        for (; it != incopy.end(); ++it) {
            temp = temp->set_intersection(*it);
        }
        return temp;
    }

    return make_set_intersection(incopy);
}

// helper to avoid redundant code
RCP<const Set> set_complement_helper(const RCP<const Set> &container,
                                     const RCP<const Set> &universe)
{
    if (is_a<Union>(*universe)) {
        auto univ = down_cast<const Union &>(*universe).get_container();
        set_set container_;
        for (auto &a : univ) {
            container_.insert(container->set_complement(a));
        }
        return SymEngine::set_union(container_);
    } else if (is_a<EmptySet>(*universe)) {
        return emptyset();
    } else if (is_a<FiniteSet>(*universe)) {
        const FiniteSet &other = down_cast<const FiniteSet &>(*universe);
        set_basic container_;
        set_basic rest;
        for (const auto &a : other.get_container()) {
            auto contain = container->contains(a);
            if (eq(*contain, *boolFalse)) {
                container_.insert(a);
            } else if (is_a<Contains>(*contain)) {
                rest.insert(a);
            }
        }
        if (rest.empty()) {
            return finiteset(container_);
        } else {
            return SymEngine::set_union(
                {finiteset(container_),
                 make_rcp<const Complement>(finiteset(rest), container)});
        }
    }
    return make_rcp<const Complement>(universe, container);
}

RCP<const Set> set_complement(const RCP<const Set> &universe,
                              const RCP<const Set> &container)
{
    // represents universe - container
    return container->set_complement(universe);
}

RCP<const Set> conditionset(const RCP<const Basic> &sym,
                            const RCP<const Boolean> &condition)
{
    if (eq(*condition, *boolean(false))) {
        return emptyset();
    } else if (eq(*condition, *boolean(true))) {
        return universalset();
    }
    if (is_a<And>(*condition)) {
        auto cont = down_cast<const And &>(*condition).get_container();
        set_boolean newcont;
        set_basic present, others;
        for (auto it = cont.begin(); it != cont.end(); it++) {
            if (is_a<Contains>(**it)
                and eq(*down_cast<const Contains &>(**it).get_expr(), *sym)
                and is_a<FiniteSet>(
                    *down_cast<const Contains &>(**it).get_set())) {
                auto fset = down_cast<const Contains &>(**it).get_set();
                auto fcont
                    = down_cast<const FiniteSet &>(*fset).get_container();
                // use the result of simplification done in `logical_and()`
                for (const auto &elem : fcont) {
                    if (not(is_a_Number(*elem) or is_a<Constant>(*elem))) {
                        others.insert(elem);
                    } else {
                        // logical_and() doesn't guarantee that if element of a
                        // finiteset is a number, then it satisfies other
                        // conditions
                        // it only assures that there doesn't exist any such
                        // element of finiteset that surely fails in other
                        // conditions.
                        auto restCont = cont;
                        restCont.erase(*it);
                        auto restCond = logical_and(restCont);
                        map_basic_basic d;
                        d[sym] = elem;
                        auto contain = restCond->subs(d);
                        if (eq(*contain, *boolean(true))) {
                            present.insert(elem);
                        } else if (not eq(*contain, *boolean(false))) {
                            others.insert(elem);
                        } else {
                            throw SymEngineException("element should have "
                                                     "been removed within "
                                                     "logical_and()");
                        }
                    }
                }
            } else {
                newcont.insert(*it);
            }
        }
        if (not present.empty()) {
            newcont.insert(finiteset(others)->contains(sym));
            return SymEngine::set_union(
                {finiteset(present), conditionset(sym, logical_and(newcont))});
        }
    }
    if (is_a<Contains>(*condition)) {
        return down_cast<const Contains &>(*condition).get_set();
    }
    return make_rcp<const ConditionSet>(sym, condition);
}

} // namespace SymEngine
