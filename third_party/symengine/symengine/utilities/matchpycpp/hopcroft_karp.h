#ifndef SYMENGINE_UTILITIES_MATCHPYCPP_HOPCROFT_KARP_H_
#define SYMENGINE_UTILITIES_MATCHPYCPP_HOPCROFT_KARP_H_

#include <vector>
#include <map>
#include <iostream>
#include <tuple>
#include <deque>
#include <set>
#include <climits>

using namespace std;

/*
 * Implementation of the Hopcroft-Karp algorithm on a bipartite graph.
 *
 * The bipartite graph has types TLeft and TRight on the two partitions.
 *
 * The constructor accepts a `map` mapping the left vertices to the set of
 * connected right vertices.
 *
 * The method `.hopcroft_karp()` finds the maximum cardinality matching,
 * returning its cardinality. The matching will be stored in the file
 * `pair_left`
 * and `pair_right` after the matching is found.
 */
template <typename TLeft, typename TRight>
class HopcroftKarp
{
public:
    HopcroftKarp(map<TLeft, set<TRight>> &_graph_left)
        : _graph_left(_graph_left)
    {
        reference_distance = INT_MAX;
        get_left_indices_vector(_graph_left);
    }

    int hopcroft_karp()
    {
        pair_left.clear();
        pair_right.clear();
        dist_left.clear();
        for (const TLeft &left : _left) {
            dist_left[left] = INT_MAX;
        }
        int matchings = 0;
        while (true) {
            if (!_bfs_hopcroft_karp())
                break;
            for (const TLeft &left : _left) {
                if (pair_left.find(left) != pair_left.end()) {
                    continue;
                }
                if (_dfs_hopcroft_karp(left)) {
                    matchings++;
                }
            }
        }
        return matchings;
    }

    map<TLeft, TRight> pair_left;
    map<TRight, TLeft> pair_right;

private:
    vector<TLeft> _left;
    map<TLeft, set<TRight>> _graph_left;
    map<TLeft, int> dist_left;
    int reference_distance;

    void get_left_indices_vector(const map<TLeft, set<TRight>> &m)
    {
        _left.reserve(m.size());
        for (const pair<const TLeft, set<TRight>> &p : m) {
            _left.push_back(p.first);
        }
    }

    bool _bfs_hopcroft_karp()
    {
        deque<TLeft> vertex_queue;
        for (const TLeft &left_vert : _left) {
            if (pair_left.find(left_vert) == pair_left.end()) {
                vertex_queue.push_back(left_vert);
                dist_left[left_vert] = 0;
            } else {
                dist_left[left_vert] = INT_MAX;
            }
        }
        reference_distance = INT_MAX;
        while (true) {
            if (vertex_queue.empty())
                break;
            TLeft &left_vertex = vertex_queue.front();
            vertex_queue.pop_front();
            if (dist_left.at(left_vertex) >= reference_distance)
                continue;
            for (const TRight &right_vertex : _graph_left.at(left_vertex)) {
                if (pair_right.find(right_vertex) == pair_right.end()) {
                    if (reference_distance == INT_MAX) {
                        reference_distance = dist_left[left_vertex] + 1;
                    }
                } else {
                    TLeft &other_left = pair_right.at(right_vertex);
                    if (dist_left.at(other_left) == INT_MAX) {
                        dist_left[other_left] = dist_left[left_vertex] + 1;
                        vertex_queue.push_back(other_left);
                    }
                }
            }
        }
        return reference_distance < INT_MAX;
    }

    inline void swap_lr(const TLeft &left, const TRight &right)
    {
        pair_left[left] = right;
        pair_right[right] = left;
    }

    bool _dfs_hopcroft_karp(const TLeft &left)
    {
        for (const TRight &right : _graph_left.at(left)) {
            if (pair_right.find(right) == pair_right.end()) {
                if (reference_distance == dist_left.at(left) + 1) {
                    swap_lr(left, right);
                    return true;
                }
            } else {
                TLeft &other_left = pair_right.at(right);
                if (dist_left.at(other_left) == dist_left.at(left) + 1) {
                    if (_dfs_hopcroft_karp(other_left)) {
                        swap_lr(left, right);
                        return true;
                    }
                }
            }
        }
        dist_left[left] = INT_MAX;
        return false;
    }
};

#endif /* SYMENGINE_UTILITIES_MATCHPYCPP_HOPCROFT_KARP_H_ */
