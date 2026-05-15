#ifndef MATCHPY_BIPARTITE_H
#define MATCHPY_BIPARTITE_H

#include <deque>
#include <map>
#include <set>

#include <symengine/utilities/matchpycpp/hopcroft_karp.h>
#include <symengine/utilities/matchpycpp/common.h>

using namespace std;

/*
 * A bipartite graph representation.
 *
 * Only TLeft = int, TRight = int are supported.
 */
template <typename TLeft, typename TRight, typename TEdgeValue>
class BipartiteGraph
{
public:
    TYPES_DERIVED_FROM_TLEFT_TRIGHT

    map<Edge, TEdgeValue> _edges;
    set<TLeft> _left;
    set<TRight> _right;
    map<TLeft, set<TRight>> _graph_left;
    map<TRight, set<TLeft>> _graph_right;

    BipartiteGraph()
    {
    }

    BipartiteGraph(map<Edge, TEdgeValue> &edges)
    {
        _edges = edges;
        for (const pair<const Edge, TEdgeValue> &p : edges) {
            TLeft nl = get<0>(p.first);
            TRight nr = get<1>(p.first);
            _left.insert(nl);
            _right.insert(nr);
            _graph_left[nl].insert(nr);
            _graph_right[nr].insert(nl);
        }
    }

    void __setitem__(const Edge &key, const TEdgeValue &value)
    {
        _edges[key] = value;
        _left.insert(get<0>(key));
        _right.insert(get<1>(key));

        TLeft k1 = get<0>(key);
        if (_graph_left.find(k1) == _graph_left.end()) {
            _graph_left[k1] = set<TRight>();
        }
        _graph_left[k1].insert(get<1>(key));

        TRight k2 = get<1>(key);
        if (_graph_right.find(k2) == _graph_right.end()) {
            _graph_right[k2] = set<TLeft>();
        }
        _graph_right[k2].insert(get<0>(key));
    }

    TEdgeValue setdefault(const Edge &key, const TEdgeValue &value)
    {
        if (_edges.find(key) != _edges.end()) {
            return _edges[key];
        } else {
            __setitem__(key, value);
            return value;
        }
    }

    TEdgeValue &__getitem__(const Edge &key)
    {
        return _edges.at(key);
    }

    void __delitem__(const Edge &key)
    {
        _edges.erase(key);
        for (const pair<const Edge, TEdgeValue> &p : _edges) {
            TLeft l = get<0>(p.first);
            if (l == get<0>(key)) {
                _left.erase(get<0>(key));
                break;
            }
        }
        for (const pair<const Edge, TEdgeValue> &p : _edges) {
            TRight l = get<1>(p.first);
            if (l == get<1>(key)) {
                _right.erase(get<0>(key));
                break;
            }
        }
        _graph_right[get<0>(key)].erase(get<1>(key));
        _graph_left[get<1>(key)].erase(get<0>(key));
    }

    //    """Returns a copy of this bipartite graph with the given edge and its
    //    adjacent nodes removed."""
    BipartiteGraph<TLeft, TRight, TEdgeValue>
    without_nodes(const Edge &edge) const
    {
        BipartiteGraph<TLeft, TRight, TEdgeValue> new_graph;
        for (const pair<const Edge, TEdgeValue> &p : _edges) {
            Edge node = p.first;
            TEdgeValue v = p.second;
            TLeft &n1 = get<0>(node);
            TRight &n2 = get<1>(node);
            if ((n1 == get<0>(edge)) || (n2 == get<1>(edge))) {
                continue;
            }
            new_graph.__setitem__(node, v);
        }
        return new_graph;
    }

    // Returns a copy of this bipartite graph with the given edge
    // removed
    BipartiteGraph<TLeft, TRight, TEdgeValue>
    without_edge(const Edge &edge) const
    {
        BipartiteGraph<TLeft, TRight, TEdgeValue> new_graph;
        for (const pair<const Edge, TEdgeValue> &p : _edges) {
            Edge e2 = p.first;
            TEdgeValue v = p.second;
            if (edge == e2) {
                continue;
            }
            new_graph.__setitem__(e2, v);
        }
        return new_graph;
    }

    map<TLeft, TRight> find_matching() const
    {
        map<TLeft, set<TRight>> directed_graph;

        for (const pair<const Edge, TEdgeValue> &p : _edges) {
            TLeft left = get<0>(p.first);
            TRight right = get<1>(p.first);
            auto elem = directed_graph.find(left);
            if (elem == directed_graph.end()) {
                directed_graph[left] = {right};
            } else {
                elem->second.insert(right);
            }
        }

        HopcroftKarp<TLeft, TRight> hk(directed_graph);
        hk.hopcroft_karp();
        return hk.pair_left;
    }

    void clear()
    {
        _edges.clear();
        _left.clear();
        _right.clear();
        _graph_left.clear();
        _graph_right.clear();
    }
};

template <typename TLeft, typename TRight>
class _DirectedMatchGraph
{
public:
    TYPES_DERIVED_FROM_TLEFT_TRIGHT

    map<Node, NodeSet> _map;

    _DirectedMatchGraph()
    {
    }

    template <typename TEdgeValue>
    _DirectedMatchGraph(BipartiteGraph<TLeft, TRight, TEdgeValue> graph,
                        map<TLeft, TRight> matching)
    {
        for (const pair<const Edge, TEdgeValue> &p : graph._edges) {
            TLeft tail = get<0>(p.first);
            TRight head = get<1>(p.first);
            auto elem = matching.find(tail);
            if ((elem != matching.end()) && (elem->second == head)) {
                set<tuple<int, TRight>> s;
                s.insert(make_tuple(RIGHT, head));
                _map[make_tuple(LEFT, tail)] = s;
            } else {
                Node right_head = make_tuple(RIGHT, head);
                auto elem_map = _map.find(right_head);
                if (elem_map == _map.end()) {
                    _map[right_head] = {make_tuple(LEFT, tail)};
                } else {
                    elem_map->second.insert(make_tuple(LEFT, tail));
                }
            }
        }
    }

    NodeList find_cycle() const
    {
        set<Node> visited;
        for (const pair<const Node, NodeSet> &n : _map) {
            NodeList node_list;
            NodeList cycle;
            cycle = _find_cycle(n.first, node_list, visited);
            if (!cycle.empty()) {
                return cycle;
            }
        }
        return NodeList();
    }

    NodeList _find_cycle(const Node &node, NodeList &path,
                         set<Node> &visited) const
    {
        if (visited.find(node) != visited.end()) {
            typename NodeList::iterator found_end;
            found_end = find_if(path.begin(), path.end(),
                                [&node](const Node &p) { return p == node; });
            if (found_end != path.end()) {
                return NodeList(path.begin(), found_end + 1);
            } else {
                return NodeList();
            }
        }
        visited.insert(node);
        if (_map.find(node) == _map.end()) {
            return NodeList();
        }
        for (const Node &other : _map.at(node)) {
            NodeList new_path(path.begin(), path.end());
            new_path.push_back(node);
            NodeList cycle = _find_cycle(other, new_path, visited);
            if (!cycle.empty()) {
                return cycle;
            }
        }
        return NodeList();
    }
};

/*
 * Algorithm described in "Algorithms for Enumerating All Perfect, Maximum and
 * Maximal Matchings in Bipartite Graphs"
 * By Takeaki Uno in "Algorithms and Computation: 8th International Symposium,
 * ISAAC '97 Singapore,
 * December 17-19, 1997 Proceedings"
 * See http://dx.doi.org/10.1007/3-540-63890-3_11
 */

template <typename TLeft, typename TRight, typename TEdgeValue>
generator<map<TLeft, TRight>> _enum_maximum_matchings_iter(
    BipartiteGraph<TLeft, TRight, TEdgeValue> &graph,
    map<TLeft, TRight> &matching,
    const _DirectedMatchGraph<TLeft, TRight> &directed_match_graph)
{
    TYPES_DERIVED_FROM_TLEFT_TRIGHT

    vector<map<TLeft, TRight>> result;
    map<TLeft, TRight> new_match;
    _DirectedMatchGraph<TLeft, TRight> directed_match_graph_minus,
        directed_match_graph_plus;

    BipartiteGraph<TLeft, TRight, TEdgeValue> graph_plus, graph_minus;
    _DirectedMatchGraph<TLeft, TRight> dgm_plus, dgm_minus;

    // Step 1
    if (graph._edges.empty()) {
        return result;
    }

    // Step 2
    // Find a circle in the directed matching graph
    // Note that this circle alternates between nodes from the left and the
    NodeList raw_cycle = directed_match_graph.find_cycle();
    vector<TLeft> cycle;
    if (!raw_cycle.empty()) {
        if (get<0>(raw_cycle[0]) != LEFT) {
            cycle.push_back(get<1>(*raw_cycle.end()));
            for (size_t i = 1; i < raw_cycle.size(); i++) {
                cycle.push_back(get<1>(raw_cycle[i]));
            }
        } else {
            for (Node &i : raw_cycle) {
                cycle.push_back(get<1>(i));
            }
        }

        // Step 3 - TODO: Properly find right edge? (to get complexity bound)
        Edge edge = make_tuple(cycle[0], cycle[1]);

        // Step 4
        // already done because we are not really finding the optimal edge

        // Step 5
        // Construct new matching M' by flipping edges along the cycle, i.e.
        // change the direction of all the
        // edges in the circle
        new_match = matching;
        for (size_t i = 0; i < cycle.size(); i += 2) {
            new_match[(TLeft)cycle[i]] = cycle[i - 1];
        }
        result.push_back(new_match);

        // Construct G+(e) and G-(e)
        TEdgeValue old_value = graph._edges.at(edge);
        graph.__delitem__(edge);

        // Step 7
        // Recurse with the new matching M' but without the edge e
        // directed_match_graph_minus = _DirectedMatchGraph(graph, new_match)
        directed_match_graph_minus
            = _DirectedMatchGraph<TLeft, TRight>(graph, new_match);
        generator<map<TLeft, TRight>> g = _enum_maximum_matchings_iter(
            graph, new_match, directed_match_graph_minus);
        result.insert(result.end(), g.begin(), g.end());

        graph.__setitem__(edge, old_value);

        // Step 6
        // Recurse with the old matching M but without the edge e
        graph_plus = graph;

        vector<tuple<TLeft, TRight, TEdgeValue>> edges;

        for (const pair<const Edge, TEdgeValue> &p : graph_plus._edges) {
            TLeft left = get<0>(p.first);
            TRight right = get<1>(p.first);
            if ((left == get<0>(edge)) && (right == get<1>(edge))) {
                Edge lredge = make_tuple(left, right);
                edges.push_back(
                    make_tuple(left, right, graph_plus.__getitem__(lredge)));
                graph_plus.__delitem__(lredge);
            }
        }
        directed_match_graph_plus
            = _DirectedMatchGraph<TLeft, TRight>(graph_plus, matching);
        g = _enum_maximum_matchings_iter(graph_plus, matching,
                                         directed_match_graph_plus);

        result.insert(result.end(), g.begin(), g.end());

        for (const tuple<TLeft, TRight, TEdgeValue> &p : edges) {
            Edge edge0 = make_tuple(get<0>(p), get<1>(p));
            graph_plus.__setitem__(edge0, get<2>(p));
        }
    } else {
        // Step 8
        // Find feasible path of length 2 in D(graph, matching)
        // This path has the form left1 -> right -> left2
        // left1 must be in the left part of the graph and in matching
        // right must be in the right part of the graph
        // left2 is also in the left part of the graph and but must not be in
        // matching
        TLeft left1;
        TLeft left2;
        bool left2found = false;
        TRight right;

        for (const pair<const Node, NodeSet> &p : directed_match_graph._map) {
            int part1 = get<0>(p.first);
            TLeft node1 = get<1>(p.first);
            if ((part1 == LEFT)
                && (matching.find((TLeft)node1) != matching.end())) {
                left1 = (TLeft)node1;
                right = matching[left1];
                if (directed_match_graph._map.find(make_tuple(RIGHT, right))
                    != directed_match_graph._map.end()) {
                    for (const tuple<int, TLeft> &p2 :
                         directed_match_graph._map.at(
                             make_tuple(RIGHT, right))) {
                        TLeft node2 = get<1>(p2);
                        if (matching.find((TLeft)node2) == matching.end()) {
                            left2 = node2;
                            left2found = true;
                            break;
                        }
                    }
                    if (left2found) {
                        break;
                    }
                }
            }
        }
        if (!left2found) {
            return result;
        }
        // Construct M'
        // Exchange the direction of the path left1 -> right -> left2
        // to left1 <- right <- left2 in the new matching
        new_match = matching;
        new_match.erase(left1);
        new_match[left2] = right;

        result.push_back(new_match);

        Edge edge = make_tuple(left2, right);

        // Construct G+(e) and G-(e)
        graph_plus = graph.without_nodes(edge);
        graph_minus = graph.without_edge(edge);

        dgm_plus = _DirectedMatchGraph<TLeft, TRight>(graph_plus, new_match);
        dgm_minus = _DirectedMatchGraph<TLeft, TRight>(graph_minus, matching);

        // Step 9
        generator<map<TLeft, TRight>> g
            = _enum_maximum_matchings_iter(graph_plus, new_match, dgm_plus);

        result.insert(result.end(), g.begin(), g.end());

        // Step 10
        g = _enum_maximum_matchings_iter(graph_minus, matching, dgm_minus);

        result.insert(result.end(), g.begin(), g.end());
    }
    return result;
}

template <typename TLeft, typename TRight, typename TEdgeValue>
generator<map<TLeft, TRight>>
enum_maximum_matchings_iter(BipartiteGraph<TLeft, TRight, TEdgeValue> &graph)
{
    vector<map<TLeft, TRight>> result;
    map<TLeft, TRight> matching = graph.find_matching();
    if (!matching.empty()) {
        result.push_back(matching);
        // graph = graph.__copy__();
        generator<map<TLeft, TRight>> extension = _enum_maximum_matchings_iter(
            graph, matching,
            _DirectedMatchGraph<TLeft, TRight>(graph, matching));
        result.insert(result.end(), extension.begin(), extension.end());
    }
    return result;
}

#endif
