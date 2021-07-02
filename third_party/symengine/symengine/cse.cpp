#include <symengine/basic.h>
#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/functions.h>
#include <symengine/visitor.h>

#include <queue>

namespace SymEngine
{
umap_basic_basic opt_cse(const vec_basic &exprs);
void tree_cse(vec_pair &replacements, vec_basic &reduced_exprs,
              const vec_basic &exprs, umap_basic_basic &opt_subs);

class FuncArgTracker
{

public:
    std::unordered_map<RCP<const Basic>, unsigned, RCPBasicHash, RCPBasicKeyEq>
        value_numbers;
    vec_basic value_number_to_value;
    std::vector<std::set<unsigned>> arg_to_funcset;
    std::vector<std::set<unsigned>> func_to_argset;

public:
    FuncArgTracker(
        const std::vector<std::pair<RCP<const Basic>, vec_basic>> &funcs)
    {
        arg_to_funcset.resize(funcs.size());
        for (unsigned func_i = 0; func_i < funcs.size(); func_i++) {
            std::set<unsigned> func_argset;
            for (auto &func_arg : funcs[func_i].second) {
                unsigned arg_number = get_or_add_value_number(func_arg);
                func_argset.insert(arg_number);
                arg_to_funcset[arg_number].insert(func_i);
            }
            func_to_argset.push_back(func_argset);
        }
    }

    template <typename Container>
    vec_basic get_args_in_value_order(Container &argset)
    {
        vec_basic v;
        for (unsigned i : argset) {
            v.push_back(value_number_to_value[i]);
        }
        return v;
    }

    unsigned get_or_add_value_number(RCP<const Basic> value)
    {
        unsigned nvalues = numeric_cast<unsigned>(value_numbers.size());
        auto ret = value_numbers.insert(std::make_pair(value, nvalues));
        bool inserted = ret.second;
        if (inserted) {
            value_number_to_value.push_back(value);
            arg_to_funcset.push_back(std::set<unsigned>());
            return nvalues;
        } else {
            return ret.first->second;
        }
    }

    void stop_arg_tracking(unsigned func_i)
    {
        for (unsigned arg : func_to_argset[func_i]) {
            arg_to_funcset[arg].erase(func_i);
        }
    }

    /*
       Return a dict whose keys are function numbers. The entries of the dict
       are the number of arguments said function has in common with `argset`.
       Entries have at least 2 items in common.
    */
    std::map<unsigned, unsigned>
    get_common_arg_candidates(std::set<unsigned> &argset, unsigned min_func_i)
    {
        std::map<unsigned, unsigned> count_map;
        std::vector<std::set<unsigned>> funcsets;
        for (unsigned arg : argset) {
            funcsets.push_back(arg_to_funcset[arg]);
        }
        // Sorted by size to make best use of the performance hack below.
        std::sort(funcsets.begin(), funcsets.end(),
                  [](const std::set<unsigned> &a, const std::set<unsigned> &b) {
                      return a.size() < b.size();
                  });

        for (unsigned i = 0; i < funcsets.size(); i++) {
            auto &funcset = funcsets[i];
            for (unsigned func_i : funcset) {
                if (func_i >= min_func_i) {
                    count_map[func_i] += 1;
                }
            }
        }

        /*auto &largest_funcset = funcsets[funcsets.size() - 1];

        // We pick the smaller of the two containers to iterate over to
        // reduce the number of items we have to look at.

        if (largest_funcset.size() < count_map.size()) {
            for (unsigned func_i : largest_funcset) {
                if (count_map[func_i] < 1) {
                    continue;
                }
                if (count_map.find(func_i) != count_map.end()) {
                    count_map[func_i] += 1;
                }
            }
        } else {
            for (auto &count_map_pair : count_map) {
                unsigned func_i = count_map_pair.first;
                if (count_map[func_i] < 1) {
                    continue;
                }
                if (largest_funcset.find(func_i) != largest_funcset.end()) {
                    count_map[func_i] += 1;
                }
            }
        }*/
        auto iter = count_map.begin();
        for (; iter != count_map.end();) {
            if (iter->second >= 2) {
                ++iter;
            } else {
                count_map.erase(iter++);
            }
        }
        return count_map;
    }

    template <typename Container1, typename Container2>
    std::vector<unsigned>
    get_subset_candidates(const Container1 &argset,
                          const Container2 &restrict_to_funcset)
    {
        std::vector<unsigned> indices;
        for (auto f : restrict_to_funcset) {
            indices.push_back(f);
        }
        std::sort(std::begin(indices), std::end(indices));
        std::vector<unsigned> intersect_result;
        for (const auto &arg : argset) {
            std::set_intersection(indices.begin(), indices.end(),
                                  arg_to_funcset[arg].begin(),
                                  arg_to_funcset[arg].end(),
                                  std::back_inserter(intersect_result));
            intersect_result.swap(indices);
            intersect_result.clear();
        }
        return indices;
    }

    void update_func_argset(unsigned func_i,
                            const std::vector<unsigned> &new_args)
    {
        // Update a function with a new set of arguments.
        auto &old_args = func_to_argset[func_i];

        std::set<unsigned> diff;
        std::set_difference(old_args.begin(), old_args.end(), new_args.begin(),
                            new_args.end(), std::inserter(diff, diff.begin()));

        for (auto &deleted_arg : diff) {
            arg_to_funcset[deleted_arg].erase(func_i);
        }

        diff.clear();
        std::set_difference(new_args.begin(), new_args.end(), old_args.begin(),
                            old_args.end(), std::inserter(diff, diff.begin()));

        for (auto &added_arg : diff) {
            arg_to_funcset[added_arg].insert(func_i);
        }

        func_to_argset[func_i].clear();
        func_to_argset[func_i].insert(new_args.begin(), new_args.end());
    }
};

std::vector<unsigned> set_diff(const std::set<unsigned> &a,
                               const std::vector<unsigned> &b)
{
    std::vector<unsigned> diff;
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                        std::inserter(diff, diff.begin()));
    return diff;
}

void add_to_sorted_vec(std::vector<unsigned> &vec, unsigned number)
{
    if (std::find(vec.begin(), vec.end(), number) == vec.end()) {
        // Add number if not found
        vec.insert(std::upper_bound(vec.begin(), vec.end(), number), number);
    }
}

void match_common_args(const std::string &func_class, const vec_basic &funcs_,
                       umap_basic_basic &opt_subs)
{
    std::vector<std::pair<RCP<const Basic>, vec_basic>> funcs;
    for (auto &b : funcs_) {
        funcs.push_back(std::make_pair(b, b->get_args()));
    }
    std::sort(funcs.begin(), funcs.end(),
              [](const std::pair<RCP<const Basic>, vec_basic> &a,
                 const std::pair<RCP<const Basic>, vec_basic> &b) {
                  return a.second.size() < b.second.size();
              });

    auto arg_tracker = FuncArgTracker(funcs);

    std::set<unsigned> changed;
    std::map<unsigned, unsigned> common_arg_candidates_counts;

    for (unsigned i = 0; i < funcs.size(); i++) {
        common_arg_candidates_counts = arg_tracker.get_common_arg_candidates(
            arg_tracker.func_to_argset[i], i + 1);

        std::deque<unsigned> common_arg_candidates;
        for (auto it = common_arg_candidates_counts.begin();
             it != common_arg_candidates_counts.end(); ++it) {
            common_arg_candidates.push_back(it->first);
        }

        // Sort the candidates in order of match size.
        // This makes us try combining smaller matches first.
        std::sort(common_arg_candidates.begin(), common_arg_candidates.end(),
                  [&](unsigned a, unsigned b) {
                      if (common_arg_candidates_counts[a]
                          == common_arg_candidates_counts[b]) {
                          return a < b;
                      }
                      return common_arg_candidates_counts[a]
                             < common_arg_candidates_counts[b];
                  });

        while (common_arg_candidates.size() > 0) {
            unsigned j = common_arg_candidates.front();
            common_arg_candidates.pop_front();
            std::vector<unsigned> com_args;

            std::set_intersection(arg_tracker.func_to_argset[i].begin(),
                                  arg_tracker.func_to_argset[i].end(),
                                  arg_tracker.func_to_argset[j].begin(),
                                  arg_tracker.func_to_argset[j].end(),
                                  std::back_inserter(com_args));

            if (com_args.size() <= 1) {
                // This may happen if a set of common arguments was already
                // combined in a previous iteration.
                continue;
            }

            std::vector<unsigned> diff_i
                = set_diff(arg_tracker.func_to_argset[i], com_args);

            unsigned com_func_number;

            if (diff_i.size() > 0) {
                // com_func needs to be unevaluated to allow for recursive
                // matches.
                auto com_func = function_symbol(
                    func_class, arg_tracker.get_args_in_value_order(com_args));
                com_func_number = arg_tracker.get_or_add_value_number(com_func);
                add_to_sorted_vec(diff_i, com_func_number);
                arg_tracker.update_func_argset(i, diff_i);
                changed.insert(i);

            } else {
                // Treat the whole expression as a CSE.
                //
                // The reason this needs to be done is somewhat subtle. Within
                // tree_cse(), to_eliminate only contains expressions that are
                // seen more than once. The problem is unevaluated expressions
                // do not compare equal to the evaluated equivalent. So
                // tree_cse() won't mark funcs[i] as a CSE if we use an
                // unevaluated version.
                com_func_number
                    = arg_tracker.get_or_add_value_number(funcs[i].first);
            }

            std::vector<unsigned> diff_j
                = set_diff(arg_tracker.func_to_argset[j], com_args);
            add_to_sorted_vec(diff_j, com_func_number);
            arg_tracker.update_func_argset(j, diff_j);
            changed.insert(j);

            for (unsigned k : arg_tracker.get_subset_candidates(
                     com_args, common_arg_candidates)) {
                std::vector<unsigned> diff_k
                    = set_diff(arg_tracker.func_to_argset[k], com_args);
                add_to_sorted_vec(diff_k, com_func_number);
                arg_tracker.update_func_argset(k, diff_k);
                changed.insert(k);
            }
        }
        if (std::find(changed.begin(), changed.end(), i) != changed.end()) {
            opt_subs[funcs[i].first] = function_symbol(
                func_class, arg_tracker.get_args_in_value_order(
                                arg_tracker.func_to_argset[i]));
        }
        arg_tracker.stop_arg_tracking(i);
    }
}

class OptsCSEVisitor : public BaseVisitor<OptsCSEVisitor>
{
public:
    umap_basic_basic &opt_subs;
    set_basic adds;
    set_basic muls;
    set_basic seen_subexp;
    OptsCSEVisitor(umap_basic_basic &opt_subs_) : opt_subs(opt_subs_)
    {
    }
    bool is_seen(const Basic &expr)
    {
        return (seen_subexp.find(expr.rcp_from_this()) != seen_subexp.end());
    }
    void bvisit(const Derivative &x)
    {
        return;
    }
    void bvisit(const Subs &x)
    {
        return;
    }
    void bvisit(const Add &x)
    {
        if (not is_seen(x)) {
            seen_subexp.insert(x.rcp_from_this());
            for (const auto &p : x.get_args()) {
                p->accept(*this);
            }
            adds.insert(x.rcp_from_this());
        }
    }
    void bvisit(const Pow &x)
    {
        if (not is_seen(x)) {
            auto expr = x.rcp_from_this();
            seen_subexp.insert(expr);
            for (const auto &p : x.get_args()) {
                p->accept(*this);
            }
            auto ex = x.get_exp();
            if (is_a<Mul>(*ex)) {
                ex = static_cast<const Mul &>(*ex).get_coef();
            }
            if (is_a_Number(*ex)
                and static_cast<const Number &>(*ex).is_negative()) {
                vec_basic v({pow(x.get_base(), neg(x.get_exp())), integer(-1)});
                opt_subs[expr] = function_symbol("pow", v);
            }
        }
    }
    void bvisit(const Mul &x)
    {
        if (not is_seen(x)) {
            auto expr = x.rcp_from_this();
            seen_subexp.insert(expr);
            for (const auto &p : x.get_args()) {
                p->accept(*this);
            }
            if (x.get_coef()->is_negative()) {
                auto neg_expr = neg(x.rcp_from_this());
                if (not is_a<Symbol>(*neg_expr)) {
                    opt_subs[expr]
                        = function_symbol("mul", {integer(-1), neg_expr});
                    seen_subexp.insert(neg_expr);
                    expr = neg_expr;
                }
            }
            if (is_a<Mul>(*expr)) {
                muls.insert(expr);
            }
        }
    }
    void bvisit(const Basic &x)
    {
        auto v = x.get_args();
        if (v.size() > 0 and not is_seen(x)) {
            auto expr = x.rcp_from_this();
            seen_subexp.insert(expr);
            for (const auto &p : v) {
                p->accept(*this);
            }
        }
    }
};

vec_basic set_as_vec(const set_basic &s)
{
    vec_basic result;
    for (auto &u : s) {
        result.push_back(u);
    }
    return result;
}

umap_basic_basic opt_cse(const vec_basic &exprs)
{
    // Find optimization opportunities in Adds, Muls, Pows and negative
    // coefficient Muls
    umap_basic_basic opt_subs;
    OptsCSEVisitor visitor(opt_subs);
    for (auto &e : exprs) {
        e->accept(visitor);
    }

    match_common_args("add", set_as_vec(visitor.adds), opt_subs);
    match_common_args("mul", set_as_vec(visitor.muls), opt_subs);

    return opt_subs;
}

class RebuildVisitor : public BaseVisitor<RebuildVisitor, TransformVisitor>
{
private:
    umap_basic_basic &subs;
    umap_basic_basic &opt_subs;
    set_basic &to_eliminate;
    set_basic &excluded_symbols;
    vec_pair &replacements;
    unsigned next_symbol_index = 0;

public:
    using TransformVisitor::result_;
    using TransformVisitor::bvisit;
    RebuildVisitor(umap_basic_basic &subs_, umap_basic_basic &opt_subs_,
                   set_basic &to_eliminate_, set_basic &excluded_symbols_,
                   vec_pair &replacements_)
        : subs(subs_), opt_subs(opt_subs_), to_eliminate(to_eliminate_),
          excluded_symbols(excluded_symbols_), replacements(replacements_)
    {
    }
    virtual RCP<const Basic> apply(const RCP<const Basic> &orig_expr)
    {
        RCP<const Basic> expr = orig_expr;
        if (is_a_Atom(*expr)) {
            return expr;
        }

        auto iter = subs.find(expr);
        if (iter != subs.end()) {
            return iter->second;
        }
        auto iter2 = opt_subs.find(expr);
        if (iter2 != opt_subs.end()) {
            expr = iter2->second;
        }
        expr->accept(*this);
        auto new_expr = result_;
        if (to_eliminate.find(orig_expr) != to_eliminate.end()) {
            auto sym = next_symbol();
            subs[orig_expr] = sym;
            replacements.push_back(
                std::pair<RCP<const Basic>, RCP<const Basic>>(sym, new_expr));
            return sym;
        }
        return new_expr;
    }
    RCP<const Basic> next_symbol()
    {
        RCP<const Basic> sym = symbol("x" + to_string(next_symbol_index));
        next_symbol_index++;
        if (excluded_symbols.find(sym) == excluded_symbols.end()) {
            return sym;
        } else {
            return next_symbol();
        }
    };
    void bvisit(const FunctionSymbol &x)
    {
        auto &fargs = x.get_vec();
        vec_basic newargs;
        for (const auto &a : fargs) {
            newargs.push_back(apply(a));
        }
        if (x.get_name() == "add") {
            result_ = add(newargs);
        } else if (x.get_name() == "mul") {
            result_ = mul(newargs);
        } else if (x.get_name() == "pow") {
            result_ = pow(newargs[0], newargs[1]);
        } else {
            result_ = x.create(newargs);
        }
    }
};

void tree_cse(vec_pair &replacements, vec_basic &reduced_exprs,
              const vec_basic &exprs, umap_basic_basic &opt_subs)
{
    set_basic to_eliminate;
    set_basic seen_subexp;
    set_basic excluded_symbols;

    std::function<void(RCP<const Basic> & expr)> find_repeated;
    find_repeated = [&](RCP<const Basic> expr) -> void {

        if (is_a_Number(*expr)) {
            return;
        }

        if (is_a<Symbol>(*expr)) {
            excluded_symbols.insert(expr);
        }

        if (seen_subexp.find(expr) != seen_subexp.end()) {
            to_eliminate.insert(expr);
            return;
        }

        seen_subexp.insert(expr);

        auto iter = opt_subs.find(expr);
        if (iter != opt_subs.end()) {
            expr = iter->second;
        }

        vec_basic args = expr->get_args();

        for (auto &arg : args) {
            find_repeated(arg);
        }
    };

    for (auto e : exprs) {
        find_repeated(e);
    }

    umap_basic_basic subs;

    RebuildVisitor rebuild_visitor(subs, opt_subs, to_eliminate,
                                   excluded_symbols, replacements);

    for (auto &e : exprs) {
        auto reduced_e = rebuild_visitor.apply(e);
        reduced_exprs.push_back(reduced_e);
    }
}

void cse(vec_pair &replacements, vec_basic &reduced_exprs,
         const vec_basic &exprs)
{
    // Find other optimization opportunities.
    umap_basic_basic opt_subs = opt_cse(exprs);

    // Main CSE algorithm.
    tree_cse(replacements, reduced_exprs, exprs, opt_subs);
}
}
