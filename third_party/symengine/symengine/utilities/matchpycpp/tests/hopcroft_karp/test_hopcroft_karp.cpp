#include "catch.hpp"
#include <chrono>
#include <string>
#include <symengine/utilities/matchpycpp/hopcroft_karp.h>

using namespace std;

TEST_CASE("Hopcroft Karp algorithm",
          "Testing the implementation of the Hopcroft Karp algorithm.")
{
    {
        map<int, set<string>> graph = {{0, {"v0", "v1"}},
                                       {1, {"v0", "v4"}},
                                       {2, {"v2", "v3"}},
                                       {3, {"v0", "v4"}},
                                       {4, {"v0", "v3"}}};
        map<int, string> expected
            = {{0, "v1"}, {1, "v4"}, {2, "v2"}, {3, "v0"}, {4, "v3"}};
        HopcroftKarp<int, string> hk(graph);
        int matchings = hk.hopcroft_karp();
        REQUIRE(hk.pair_left == expected);
        REQUIRE(matchings == 5);
    }
    {
        map<char, set<int>> graph
            = {{'A', {1, 2}}, {'B', {2, 3}}, {'C', {2}}, {'D', {3, 4, 5, 6}},
               {'E', {4, 7}}, {'F', {7}},    {'G', {7}}};
        map<char, int> expected
            = {{'A', 1}, {'B', 3}, {'C', 2}, {'D', 5}, {'E', 4}, {'F', 7}};
        HopcroftKarp<char, int> hk(graph);
        int matchings = hk.hopcroft_karp();
        REQUIRE(hk.pair_left == expected);
        REQUIRE(matchings == 6);
    }
    {
        map<int, set<char>> graph
            = {{1, {'a', 'c'}}, {2, {'a', 'c'}}, {3, {'c', 'b'}}, {4, {'e'}}};
        map<int, char> expected = {{1, 'a'}, {2, 'c'}, {3, 'b'}, {4, 'e'}};
        HopcroftKarp<int, char> hk(graph);
        int matchings = hk.hopcroft_karp();
        REQUIRE(hk.pair_left == expected);
        REQUIRE(matchings == 4);
    }
    {
        map<char, set<int>> graph
            = {{'A', {3, 4}},    {'B', {3, 4}}, {'C', {3}}, {'D', {1, 5, 7}},
               {'E', {1, 2, 7}}, {'F', {2, 8}}, {'G', {6}}, {'H', {2, 4, 8}}};
        map<char, int> expected = {{'A', 3}, {'B', 4}, {'D', 1}, {'E', 7},
                                   {'F', 8}, {'G', 6}, {'H', 2}};
        HopcroftKarp<char, int> hk(graph);
        int matchings = hk.hopcroft_karp();
        REQUIRE(hk.pair_left == expected);
        REQUIRE(matchings == 7);
    }
}
