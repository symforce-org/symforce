#include "catch.hpp"
#include <symengine/utilities/matchpycpp/bipartite.h>
#include <map>

using namespace std;

TEST_CASE("BipartiteGraph", "")
{
    {
        map<tuple<int, int>, bool> m = {{make_tuple(0, 1), true},
                                        {make_tuple(1, 0), true},
                                        {make_tuple(1, 1), true},
                                        {make_tuple(2, 0), true},
                                        {make_tuple(2, 1), true}};
        BipartiteGraph<int, int, bool> bg(m);

        vector<map<int, int>> expected = {{{1, 0}, {0, 1}},
                                          {{1, 0}, {2, 1}},
                                          {{0, 1}, {2, 0}},
                                          {{2, 0}, {1, 1}}};
        generator<map<int, int>> result = enum_maximum_matchings_iter(bg);
        REQUIRE(result.size() == expected.size());
    }

    {
        map<tuple<int, int>, bool> m = {{make_tuple(0, 0), true},
                                        {make_tuple(1, 1), true},
                                        {make_tuple(2, 0), true}};
        BipartiteGraph<int, int, bool> bg(m);

        vector<map<int, int>> expected = {{{0, 0}, {1, 1}}, {{1, 1}, {2, 0}}};
        generator<map<int, int>> result = enum_maximum_matchings_iter(bg);
        REQUIRE(result.size() == expected.size());
    }

    {
        map<tuple<int, int>, bool> m = {{make_tuple(0, 0), true},
                                        {make_tuple(1, 1), true},
                                        {make_tuple(2, 0), true},
                                        {make_tuple(2, 1), true}};
        BipartiteGraph<int, int, bool> bg(m);

        vector<map<int, int>> expected
            = {{{0, 0}, {1, 1}}, {{1, 1}, {2, 0}}, {{0, 0}, {2, 1}}};
        generator<map<int, int>> result = enum_maximum_matchings_iter(bg);
        REQUIRE(result.size() == expected.size());
    }

    {
        map<tuple<int, int>, bool> m = {};
        BipartiteGraph<int, int, bool> bg(m);

        generator<map<int, int>> result = enum_maximum_matchings_iter(bg);
        REQUIRE(result.empty());
    }
}
